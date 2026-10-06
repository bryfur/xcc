#include "../../config.h"
#include "optimize.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>  // free
#include <string.h>

#include "ir.h"
#include "regalloc.h"
#include "ssa.h"
#include "table.h"
#include "util.h"

bool keep_phi;
bool apply_ssa;

static IR *is_last_jmp(BB *bb) {
  int len;
  IR *ir;
  if ((len = bb->irs->len) > 0 && (ir = bb->irs->data[len - 1])->kind == IR_JMP)
    return ir;
  return NULL;
}

static IR *is_last_jtable(BB *bb) {
  int len;
  IR *ir;
  if ((len = bb->irs->len) > 0 && (ir = bb->irs->data[len - 1])->kind == IR_TJMP)
    return ir;
  return NULL;
}

static void replace_jmp_destination(BBContainer *bbcon, BB *src, BB *dst) {
  for (int i = 0; i < bbcon->len; ++i) {
    BB *bb = bbcon->data[i];
    if (bb == src)
      continue;

    IR *ir = is_last_jmp(bb);
    if (ir != NULL && ir->jmp.bb == src)
      ir->jmp.bb = dst;

    IR *tjmp = is_last_jtable(bb);
    if (tjmp != NULL) {
      BB **bbs = tjmp->tjmp.bbs;
      for (size_t j = 0, len = tjmp->tjmp.len; j < len; ++j) {
        if (bbs[j] == src)
          bbs[j] = dst;
      }
    }
  }
}

static void clear_unused_bbs(BBContainer *bbcon) {
  for (int i = 1; i < bbcon->len; ++i) {
    BB *bb = bbcon->data[i];
    if (bb->from_bbs->len == 0)
      vec_clear(bb->irs);
  }
}

static void remove_unnecessary_bb(BBContainer *bbcon) {
  Table keeptbl;
  for (;;) {
    table_init(&keeptbl);
    assert(bbcon->len > 0);

    for (int i = 0; i < bbcon->len; ++i) {
      BB *bb = bbcon->data[i];
      bool remove = false;
      IR *ir_jmp = is_last_jmp(bb);
      if (ir_jmp != NULL && ir_jmp->jmp.bb == bb->next) {  // Remove jmp to next instruction.
        vec_pop(bb->irs);
        ir_jmp = NULL;
      }
      if (bb->irs->len == 0 && bb->next != NULL) {  // Empty BB.
        replace_jmp_destination(bbcon, bb, bb->next);
        remove = true;
      } else if (bb->irs->len == 1 && ir_jmp != NULL && ir_jmp->jmp.cond == COND_ANY &&
                 bb != ir_jmp->jmp.bb) {  // jmp only.
        replace_jmp_destination(bbcon, bb, ir_jmp->jmp.bb);
        if (i > 0) {
          BB *pbb = bbcon->data[i - 1];
          IR *ir0 = is_last_jmp(pbb);
          if (ir0 != NULL && ir0->jmp.cond != COND_ANY &&  // Fallthrough pass exists.
              ir0->jmp.bb == bb->next &&                   // Skip jmp: Fix bb connection.
              !(ir0->opr1->flag & VRF_FLONUM)) {
            // Invert prev jmp condition and change jmp destination.
            ir0->jmp.cond = invert_cond(ir0->jmp.cond);
            ir0->jmp.bb = ir_jmp->jmp.bb;
            remove = true;
          }
        }
      }

      if (ir_jmp != NULL)
        table_put(&keeptbl, ir_jmp->jmp.bb->label, bb);
      if ((ir_jmp == NULL || ir_jmp->jmp.cond != COND_ANY) && bb->next != NULL)
        table_put(&keeptbl, bb->next->label, bb);

      IR *tjmp = is_last_jtable(bb);
      if (tjmp != NULL) {
        BB **bbs = tjmp->tjmp.bbs;
        for (size_t j = 0, len = tjmp->tjmp.len; j < len; ++j)
          table_put(&keeptbl, bbs[j]->label, bb);
      }

      if (remove)
        table_delete(&keeptbl, bb->label);
    }

    bool again = false;
    for (int i = 1; i < bbcon->len; ++i) {
      BB *bb = bbcon->data[i];
      if (!table_try_get(&keeptbl, bb->label, NULL)) {
        if (i > 0) {
          BB *pbb = bbcon->data[i - 1];
          pbb->next = bb->next;
        }

        vec_remove_at(bbcon, i);
        --i;
        again = true;
      }
    }
    if (!again)
      break;
  }
}

//

static void remove_unused_vregs(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  unsigned char *vreg_read = malloc_or_die(vreg_count);
  for (;;) {
    for (int i = 0; i < vreg_count; ++i) {
      VReg *vreg = ra->vregs->data[i];
      // Must keep function parameter and `&` taken one.
      vreg_read[i] = vreg != NULL && (vreg->flag & (VRF_PARAM | VRF_REF)) != 0;
    }

    // Check VReg usage.
    for (int i = 0; i < bbcon->len; ++i) {
      BB *bb = bbcon->data[i];
      Vector *phis = bb->phis;
      if (phis != NULL) {
        for (int j = 0; j < phis->len; ++j) {
          Phi *phi = phis->data[j];
          for (int k = 0; k < phi->params->len; ++k) {
            VReg *vreg = phi->params->data[k];
            assert(vreg != NULL);
            if (!(vreg->flag & VRF_CONST))
              vreg_read[vreg->virt] = true;
          }
        }
      }

      for (int j = 0; j < bb->irs->len; ++j) {
        IR *ir = bb->irs->data[j];
        VReg *operands[] = {ir->opr1, ir->opr2};
        const int N = ARRAY_SIZE(operands);
        int n = N;
        Vector *additional = ir->additional_operands;
        if (additional != NULL)
          n += additional->len;
        for (int k = 0; k < n; ++k) {
          VReg *vreg = k < N ? operands[k] : additional->data[k - N];
          if (vreg != NULL && !(vreg->flag & VRF_CONST))
            vreg_read[vreg->virt] = true;
        }
      }
    }

    // Remove instruction if the destination is unread.
    for (int i = 0; i < bbcon->len; ++i) {
      BB *bb = bbcon->data[i];
      Vector *phis = bb->phis;
      if (phis != NULL) {
        for (int j = 0; j < phis->len; ++j) {
          Phi *phi = phis->data[j];
          if (vreg_read[phi->dst->virt])
            continue;
          vec_remove_at(phis, j);
          --j;
        }
      }

      for (int j = 0; j < bb->irs->len; ++j) {
        IR *ir = bb->irs->data[j];
        if (ir->dst == NULL || vreg_read[ir->dst->virt])
          continue;
        if (ir->kind == IR_CALL) {
          // Function must be CALLed even if the result is unused.
          ir->dst = NULL;
        } else {
          vec_remove_at(bb->irs, j);
          --j;
        }
      }
    }

    // Mark unused VRegs.
    bool again = false;
    for (int i = 0; i < vreg_count; ++i) {
      VReg *vreg;
      if (!vreg_read[i] && (vreg = ra->vregs->data[i]) != NULL) {
        ra->vregs->data[i] = NULL;
        if (vreg->original != vreg) {
          assert(vreg->original->virt < ra->original_vreg_count);
          Vector *vt = ra->vreg_table[vreg->original->virt];
          int j;
          for (j = 0; j < vt->len; ++j) {
            if (vt->data[j] == vreg)
              break;
          }
          assert(j < vt->len);
          vec_remove_at(vt, j);
        }
        vreg->flag |= VRF_UNUSED;
        again = true;
      }
    }
    if (!again)
      break;
  }

  free(vreg_read);
}

//

static int replace_register_in_bb(BB *bb, VReg *target, VReg *alternation, int start, int ip) {
  int first = INT_MAX;
  for (int iir = start; iir < bb->irs->len; ++iir, ++ip) {
    IR *ir = bb->irs->data[iir];

    if (ir->opr1 == target) {
      ir->opr1 = alternation;
      first = MIN(first, ip);
    }
    if (ir->opr2 == target) {
      ir->opr2 = alternation;
      first = MIN(first, ip);
    }

    Vector *additional = ir->additional_operands;
    if (additional != NULL) {
      for (int i = 0; i < additional->len; ++i) {
        VReg *vreg = additional->data[i];
        if (vreg == target)
          additional->data[i] = alternation;
      }
    }

    if (ir->kind == IR_CALL) {
      int n = ir->call->total_arg_count;
      VReg **operands = ir->call->args;
      for (int i = 0; i < n; ++i) {
        VReg *opr = operands[i];
        if (opr == target) {
          operands[i] = alternation;
          first = MIN(first, ip);
        }
      }
    }

    if (ir->dst == alternation || (ir->dst == target && ir->opr1 != alternation))
      break;
  }
  return first;
}

static bool calc_const_cond(enum ConditionKind cond, VReg *opr1, VReg *opr2) {
  assert(opr1->flag & VRF_CONST);
  assert(opr2->flag & VRF_CONST);

#define JUDGE(cond, v1, v2) \
    do { \
      switch (cond) { \
      case COND_EQ:  return v1 == v2; \
      case COND_NE:  return v1 != v2; \
      case COND_LT:  return v1 < v2; \
      case COND_GT:  return v1 > v2; \
      case COND_LE:  return v1 <= v2; \
      case COND_GE:  return v1 >= v2; \
      default: assert(false); return false; \
      } \
    } while (0)

#ifndef __NO_FLONUM
  if (opr1->flag & VRF_FLONUM) {
    double f1 = opr1->flonum.value;
    double f2 = opr2->flonum.value;
    JUDGE(cond, f1, f2);
  } else
#endif
  if (opr1->flag & VRF_UNSIGNED) {
    uint64_t u1 = wrap_value(opr1->fixnum, 1 << opr1->vsize, true);
    uint64_t u2 = wrap_value(opr2->fixnum, 1 << opr2->vsize, true);
    JUDGE(cond, u1, u2);
  } else {
    int64_t s1 = wrap_value(opr1->fixnum, 1 << opr1->vsize, false);
    int64_t s2 = wrap_value(opr2->fixnum, 1 << opr2->vsize, false);
    JUDGE(cond, s1, s2);
  }
#undef JUDGE
}

static bool replace_const_jmp(IR *ir) {
  assert(ir->opr1 != NULL);
  assert(ir->opr2 != NULL);
  assert(ir->jmp.cond != COND_ANY && ir->jmp.cond != COND_NONE);
  if (ir->opr1->flag & VRF_CONST) {
    if (ir->opr2->flag & VRF_CONST) {
      ir->jmp.cond = calc_const_cond(ir->jmp.cond, ir->opr1, ir->opr2) ? COND_ANY : COND_NONE;
      ir->opr1 = ir->opr2 = NULL;
      return true;
    } else {
      ir->jmp.cond = swap_cond(ir->jmp.cond);
      VReg *tmp = ir->opr1;
      ir->opr1 = ir->opr2;
      ir->opr2 = tmp;
    }
  }
  return false;
}

static bool replace_const_cond(RegAlloc *ra, IR *ir) {
  assert(ir->opr1 != NULL);
  assert(ir->opr2 != NULL);
  assert(ir->cond.kind != COND_ANY && ir->cond.kind != COND_NONE);
  if (ir->opr1->flag & VRF_CONST) {
    if (ir->opr2->flag & VRF_CONST) {
      // Replace COND to MOV.
      bool result = calc_const_cond(ir->cond.kind, ir->opr1, ir->opr2);
      ir->kind = IR_MOV;
      ir->opr1 = reg_alloc_spawn_const(ra, result, ir->dst->vsize, ir->dst->flag & VRF_MASK);
      ir->opr2 = NULL;
      return true;
    } else {
      ir->cond.kind = swap_cond(ir->cond.kind);
      VReg *tmp = ir->opr1;
      ir->opr1 = ir->opr2;
      ir->opr2 = tmp;
    }
  }
  return false;
}

static int64_t calc_const_expr(IR *ir) {
  assert((ir->opr1->flag & (VRF_FLONUM | VRF_CONST)) == VRF_CONST);
  assert(ir->opr2 == NULL || ir->opr2->flag & VRF_CONST);

#define CALC_CONST(kind) \
  switch (kind) { \
  case IR_ADD: value = opr1 + opr2; break; \
  case IR_SUB: value = opr1 - opr2; break; \
  case IR_MUL: value = opr1 * opr2; break; \
  case IR_DIV: assert(opr2 != 0); value = opr1 / opr2; break; \
  case IR_MOD: assert(opr2 != 0); value = opr1 % opr2; break; \
  case IR_BITAND: value = opr1 & opr2; break; \
  case IR_BITOR: value = opr1 | opr2; break; \
  case IR_BITXOR: value = opr1 ^ opr2; break; \
  case IR_LSHIFT: value = opr1 << opr2; break; \
  case IR_RSHIFT: value = opr1 >> opr2; break; \
  case IR_NEG: value = -opr1; break; \
  case IR_BITNOT: value = ~opr1; break; \
  default: assert(false); break; \
  }

  int64_t value = 0;
  if (ir->opr1->flag & VRF_UNSIGNED) {
    uint64_t opr1 = ir->opr1->fixnum;
    uint64_t opr2 = ir->opr2 != NULL ? ir->opr2->fixnum : 0;
    CALC_CONST(ir->kind);
  } else {
    int64_t opr1 = ir->opr1->fixnum;
    int64_t opr2 = ir->opr2 != NULL ? ir->opr2->fixnum : 0;
    CALC_CONST(ir->kind);
  }
#undef CALC

  return value;
}

#ifndef __NO_FLONUM
static double calc_fconst_expr(IR *ir) {
  assert((ir->opr1->flag & (VRF_FLONUM | VRF_CONST)) == (VRF_FLONUM | VRF_CONST));
  assert(ir->opr2 == NULL || ir->opr2->flag & VRF_CONST);

  double value = 0;
  double opr1 = ir->opr1->flonum.value;
  double opr2 = ir->opr2 != NULL ? ir->opr2->flonum.value : 0;
  switch (ir->kind) {
  case IR_ADD: value = opr1 + opr2; break;
  case IR_SUB: value = opr1 - opr2; break;
  case IR_MUL: value = opr1 * opr2; break;
  case IR_DIV: assert(opr2 != 0); value = opr1 / opr2; break;
  case IR_NEG: value = -opr1; break;
  default: assert(false); break;
  }

  return value;
}
#endif

static bool constant_folding(RegAlloc *ra, IR *ir) {
  switch (ir->kind) {
  case IR_DIV: case IR_MOD:
    if ((ir->opr2->flag & VRF_CONST) &&
        (   (!(ir->opr2->flag & VRF_FLONUM) && ir->opr2->fixnum == 0)
#ifndef __NO_FLONUM
         || ( (ir->opr2->flag & VRF_FLONUM) && ir->opr2->flonum.value == 0)
#endif
        )) {
      // Stay as it is, DIV or MOD instruction accepts even if both operands are constant.
      // Zero division exception will be thrown on runtime.
      // TODO: warning message?
      break;
    }
    // Fallthrough
  case IR_ADD: case IR_SUB: case IR_MUL: case IR_BITAND: case IR_BITOR: case IR_BITXOR:
  case IR_LSHIFT: case IR_RSHIFT: case IR_NEG: case IR_BITNOT:
    if ((ir->opr1->flag & VRF_CONST) && (ir->opr2 == NULL || ir->opr2->flag & VRF_CONST)) {
#ifndef __NO_FLONUM
      if (ir->opr1->flag & VRF_FLONUM) {
        double value = calc_fconst_expr(ir);
        // Replace to MOV.
        ir->kind = IR_MOV;
        ir->opr1 = reg_alloc_spawn_fconst(ra, value, ir->dst->vsize);
        ir->opr2 = NULL;
      } else
#endif
      {
        int64_t value = wrap_value(calc_const_expr(ir), 1 << ir->dst->vsize,
                                   ir->opr1->flag & VRF_UNSIGNED);
        // Replace to MOV.
        ir->kind = IR_MOV;
        ir->opr1 = reg_alloc_spawn_const(ra, value, ir->dst->vsize, ir->dst->flag & VRF_MASK);
        ir->opr2 = NULL;
      }
      return true;
    }
    break;
  case IR_COND:
    assert(ir->cond.kind != COND_ANY && ir->cond.kind != COND_NONE);
    return replace_const_cond(ra, ir);
  case IR_CAST:
    if (ir->opr1->flag & VRF_CONST) {
#ifndef __NO_FLONUM
      if (ir->dst->flag & VRF_FLONUM) {
        double value;
        if (!(ir->opr1->flag & VRF_FLONUM)) {
          value = (ir->opr1->flag & VRF_UNSIGNED) ? (double)(uint64_t)ir->opr1->fixnum
                                                  : (double)ir->opr1->fixnum;
        } else {
          value = ir->opr1->flonum.value;
        }
        // Replace to MOV.
        ir->kind = IR_MOV;
        ir->opr1 = reg_alloc_spawn_fconst(ra, value, ir->dst->vsize);
      } else
#endif
      {
        int64_t value;
#ifndef __NO_FLONUM
        if (ir->opr1->flag & VRF_FLONUM) {
          double d = ir->opr1->flonum.value;
          bool u = ir->opr1->flag & VRF_UNSIGNED;
          value = u ? (int64_t)(uint64_t)d : (int64_t)d;
          value = wrap_value(value, 1 << ir->dst->vsize, u);
        } else
#endif
        {
          value = ir->opr1->fixnum;
          if (ir->dst->vsize > ir->opr1->vsize)
            value = wrap_value(value, 1 << ir->opr1->vsize, ir->opr1->flag & VRF_UNSIGNED);
        }
        // Replace to MOV.
        ir->kind = IR_MOV;
        ir->opr1 = reg_alloc_spawn_const(ra, value, ir->dst->vsize, ir->dst->flag & VRF_MASK);
      }
      return true;
    }
    break;
  case IR_JMP:
    if (ir->jmp.cond != COND_ANY) {
      assert(ir->jmp.cond != COND_NONE);
      return replace_const_jmp(ir);
    }
    break;
  case IR_TJMP:
    if (ir->opr1->flag & VRF_CONST) {
      assert(!(ir->opr1->flag & VRF_FLONUM));
      size_t value = ir->opr1->fixnum;
      if (value >= ir->tjmp.len) {
        // Out of range, skipped by earlier check, so no need to handle here.
        break;
      }

      // Replace to unconditional jump.
      BB *target = ir->tjmp.bbs[value];
      ir->kind = IR_JMP;
      ir->jmp.bb = target;
      ir->jmp.cond = COND_ANY;
      ir->opr1 = ir->opr2 = NULL;
    }
    break;

  default: break;
  }
  return false;
}

//

static inline void fold_biofs_addition(BB *bb, int i) {
  IR *ir = bb->irs->data[i];
  if (i < bb->irs->len - 1) {
    IR *next = bb->irs->data[i + 1];
    if ((next->kind == IR_ADD || next->kind == IR_SUB) &&
        next->opr1 == ir->dst && next->opr2->flag & VRF_CONST) {
      assert(!(next->opr2->flag & VRF_FLONUM));
      // Overwrite next IR. Current IR should be eliminated because of dst is unused.
      VReg *dst = next->dst;
      int64_t offset = next->opr2->fixnum;
      if (next->kind == IR_SUB)
        offset = -offset;
      *next = *ir;
      next->dst = dst;
      if (ir->kind == IR_BOFS)
        next->bofs.offset += offset;
      else
        next->iofs.offset += offset;
    }
  }
}

static inline void fold_addition(RegAlloc *ra, BB *bb, int i) {
  IR *ir = bb->irs->data[i];
  if ((ir->opr2->flag & (VRF_FLONUM | VRF_CONST)) == VRF_CONST && i < bb->irs->len - 1) {
    IR *next = bb->irs->data[i + 1];
    if ((next->kind == IR_ADD || next->kind == IR_SUB) &&
        next->opr1 == ir->dst && next->opr2->flag & VRF_CONST) {
      assert(!(next->opr2->flag & VRF_FLONUM));
      // Overwrite next IR. Current IR should be eliminated because of dst is unused.
      VReg *dst = next->dst;
      VReg *opr2 = ir->opr2;
      int64_t value1 = opr2->fixnum;
      if (ir->kind == IR_SUB)
        value1 = -value1;
      int64_t value2 = next->opr2->fixnum;
      if (next->kind == IR_SUB)
        value2 = -value2;
      *next = *ir;
      next->kind = IR_ADD;
      next->dst = dst;
      next->opr2 = reg_alloc_spawn_const(ra, value1 + value2, opr2->vsize, opr2->flag & VRF_MASK);
    }
  }
}

static inline int muldiv_to_shift(RegAlloc *ra, BB *bb, int i) {
  IR *ir = bb->irs->data[i];
  if ((ir->opr2->flag & (VRF_FLONUM | VRF_CONST)) == VRF_CONST &&
      IS_POWER_OF_2(ir->opr2->fixnum)) {
    int shift = most_significant_bit(ir->opr2->fixnum);
    if (ir->kind == IR_DIV && !(ir->opr1->flag & VRF_UNSIGNED)) {
      // Patch for signed right shift:
      enum VRegSize vsize = ir->opr1->vsize;
      int bits = TARGET_CHAR_BIT << vsize;
      VReg *tmps = reg_alloc_spawn(ra, vsize, ir->opr1->flag & VRF_MASK);
      IR *mov_to_tmps = new_ir_mov(tmps, ir->opr1);
      IR *sign_bit = new_ir_bop_raw(
          IR_RSHIFT, tmps, tmps, reg_alloc_spawn_const(ra, bits - 1, vsize, tmps->flag & VRF_MASK));
      IR *mov_to_tmpu = new_ir_cast(tmps, tmps->vsize, (tmps->flag & VRF_MASK) | VRF_UNSIGNED);
      VReg *tmpu = mov_to_tmpu->dst;
      IR *addend = new_ir_bop_raw(
          IR_RSHIFT, tmpu, tmpu,
          reg_alloc_spawn_const(ra, bits - shift, vsize, tmpu->flag & VRF_MASK));
      VReg *patched = reg_alloc_spawn(ra, vsize, ir->opr1->flag & VRF_MASK);
      IR *add = new_ir_bop_raw(IR_ADD, patched, ir->opr1, tmpu);
      vec_insert(bb->irs, i++, mov_to_tmps);
      vec_insert(bb->irs, i++, sign_bit);
      vec_insert(bb->irs, i++, mov_to_tmpu);
      vec_insert(bb->irs, i++, addend);
      vec_insert(bb->irs, i++, add);
      ir->opr1 = patched;
    }
    ir->kind = ir->kind + (IR_LSHIFT - IR_MUL);
    ir->opr2 = reg_alloc_spawn_const(ra, shift, ir->opr2->vsize, ir->opr2->flag & VRF_MASK);
  }
  return i;
}

static void peephole(RegAlloc *ra, BB *bb) {
  for (int i = 0; i < bb->irs->len; ++i) {
    IR *ir = bb->irs->data[i];
    switch (ir->kind) {
    case IR_BOFS:
    case IR_IOFS:
      fold_biofs_addition(bb, i);
      break;
    case IR_ADD:
    case IR_SUB:
      fold_addition(ra, bb, i);
      break;
    case IR_MUL:
    case IR_DIV:
      i = muldiv_to_shift(ra, bb, i);
      break;
    default:
      break;
    }
  }
}

// Depends on SSA.

static int replace_register(BBContainer *bbcon, VReg *target, VReg *alternation) {
  if (target->flag & (VRF_FORCEMEMORY | VRF_VOLATILEREG) ||
      alternation->flag & (VRF_FORCEMEMORY | VRF_VOLATILEREG))
    return INT_MAX;

  int first = INT_MAX;
  int ip = 0;
  for (int ibb = 0; ibb < bbcon->len; ++ibb) {
    BB *bb = bbcon->data[ibb];

    Vector *phis = bb->phis;
    if (phis != NULL) {
      for (int iphi = 0; iphi < phis->len; ++iphi) {
        Phi *phi = phis->data[iphi];
        for (int i = 0; i < phi->params->len; ++i) {
          VReg *opr = phi->params->data[i];
          if (opr == target) {
            phi->params->data[i] = alternation;
            first = MIN(first, ip);
          }
        }
      }
    }

    int at = replace_register_in_bb(bb, target, alternation, 0, ip);
    first = MIN(first, at);
    ip += bb->irs->len;
  }
  return first;
}

static void copy_propagation(RegAlloc *ra, BBContainer *bbcon) {
  bool again;
  do {
    again = false;
    int ip = 0;
    for (int ibb = 0; ibb < bbcon->len; ++ibb) {
      BB *bb = bbcon->data[ibb];
      if (bb->from_bbs->len == 0 && ibb > 0)
        continue;

      Vector *phis = bb->phis;
      if (phis != NULL) {
        for (int iphi = 0; iphi < phis->len; ++iphi) {
          Phi *phi = phis->data[iphi];
          int n = phi->params->len;
          assert(n > 0);
          VReg *dst = phi->dst;
          VReg *value = phi->params->data[0];
          int i;
          for (i = 1; i < n; ++i) {
            VReg *vreg = phi->params->data[i];
            if (vreg != value && vreg != dst)
              break;
          }
          if (i >= n) {  // All values are same.
            IR *ir = new_ir_mov(dst, value);
            vec_insert(bb->irs, 0, ir);
            vec_remove_at(phis, iphi--);
          }
        }
      }

      for (int iir = 0; iir < bb->irs->len; ++iir, ++ip) {
        IR *ir = bb->irs->data[iir];
        if (constant_folding(ra, ir)) {
          if (ir->kind == IR_JMP && ir->jmp.cond == COND_NONE) {
            vec_remove_at(bb->irs, iir);
            --iir;
            --ip;
            continue;
          }
        }

        switch (ir->kind) {
        case IR_RESULT:
          // Inlined function call uses RESULT with `dst`.
          if (ir->dst == NULL)
            break;
          // Fallthrough
        case IR_MOV:
          if (ir->dst->flag & VRF_VOLATILE)
            break;
          if (replace_register(bbcon, ir->dst, ir->opr1) < ip)  // Former instruction is replaced:
            again = true;  // Try again.
          break;
        default: break;
        }
      }
    }
  } while (again);
}

// FPC: move coalescing on the non-SSA IR, before register allocation.
//
//   (A)  t = OP ...; v = MOV t        ->  v = OP ...      (t used only there)
//   (B)  t = MOV v; ... X t ...       ->  ... X v ...     (t defined and used once,
//                                                          in the same block, v unchanged)
//
// What a loop counter's `i++` and an expression assigned to a variable
// leave: a temporary and a register-to-register move each.

static void count_vreg_uses(BBContainer *bbcon, int *uses, int *defs) {
  for (int i = 0; i < bbcon->len; ++i) {
    BB *bb = bbcon->data[i];
    if (bb->phis != NULL) {
      for (int j = 0; j < bb->phis->len; ++j) {
        Phi *phi = bb->phis->data[j];
        ++defs[phi->dst->virt];
        for (int k = 0; k < phi->params->len; ++k) {
          VReg *vreg = phi->params->data[k];
          if (!(vreg->flag & VRF_CONST))
            ++uses[vreg->virt];
        }
      }
    }
    for (int j = 0; j < bb->irs->len; ++j) {
      IR *ir = bb->irs->data[j];
      VReg *operands[] = {ir->opr1, ir->opr2};
      const int N = ARRAY_SIZE(operands);
      int n = N;
      Vector *additional = ir->additional_operands;
      if (additional != NULL)
        n += additional->len;
      for (int k = 0; k < n; ++k) {
        VReg *vreg = k < N ? operands[k] : additional->data[k - N];
        if (vreg != NULL && !(vreg->flag & VRF_CONST))
          ++uses[vreg->virt];
      }
      if (ir->dst != NULL && !(ir->dst->flag & VRF_CONST))
        ++defs[ir->dst->virt];
    }
  }
}

static bool coalescable(VReg *t, VReg *v) {
  // t: a temporary; v: any register variable (a parameter too)
  const int NG = VRF_CONST | VRF_STACK_PARAM | VRF_FORCEMEMORY | VRF_VOLATILEREG | VRF_NO_SPILL;
  return t != v && !(t->flag & (NG | VRF_PARAM)) && !(v->flag & NG) && t->vsize == v->vsize &&
         (t->flag & (VRF_MASK | VRF_UNALIGNED)) == (v->flag & (VRF_MASK | VRF_UNALIGNED));
}

// FPC: whether `v = (v's type)(a OP b)` is `v = a OP b` with a, b as they
// are (kept extended to 32 bits in their registers: ARM32): a bitwise op or
// a right shift of values of v's type or narrower
static bool narrow_operand_fits(VReg *x, VReg *v) {
  int bits = 8 << v->vsize;
  if (x->flag & VRF_CONST) {
    if (x->flag & VRF_FLONUM)
      return false;
    int64_t k = x->fixnum;
    return (v->flag & VRF_UNSIGNED) ? k >= 0 && k < ((int64_t)1 << bits)
                                    : k >= -((int64_t)1 << (bits - 1)) && k < ((int64_t)1 << (bits - 1));
  }
  if (x->flag & VRF_FLONUM || x->vsize > v->vsize)
    return false;
  if (v->flag & VRF_UNSIGNED)
    return (x->flag & VRF_UNSIGNED) != 0;
  return !(x->flag & VRF_UNSIGNED) || x->vsize < v->vsize;
}

static bool narrow_op_fits(IR *ir, VReg *v) {
#if XCC_TARGET_ARCH == XCC_ARCH_ARM32
  if (v->flag & (VRF_FLONUM | VRF_CONST) || v->vsize >= VRegSize4 || ir->dst->vsize > VRegSize4 ||
      (ir->dst->flag & VRF_FLONUM))
    return false;
  switch (ir->kind) {
  case IR_BITAND: case IR_BITOR: case IR_BITXOR:
    return narrow_operand_fits(ir->opr1, v) && narrow_operand_fits(ir->opr2, v);
  case IR_RSHIFT:
    return !(ir->opr1->flag & VRF_CONST) && narrow_operand_fits(ir->opr1, v) &&
           (ir->opr1->flag & VRF_UNSIGNED) == (v->flag & VRF_UNSIGNED) &&
           !(ir->opr2->flag & VRF_FLONUM);
  default:
    return false;
  }
#else
  return false;
#endif
}

static bool coalescable_narrow(VReg *t, VReg *v) {
  const int NG = VRF_CONST | VRF_STACK_PARAM | VRF_FORCEMEMORY | VRF_VOLATILEREG | VRF_NO_SPILL;
  return t != v && !(t->flag & (NG | VRF_PARAM)) && !(v->flag & NG);
}

static bool ir_reads(IR *ir, VReg *v) {
  if (ir->opr1 == v || ir->opr2 == v)
    return true;
  Vector *additional = ir->additional_operands;
  if (additional != NULL) {
    for (int k = 0; k < additional->len; ++k) {
      if (additional->data[k] == v)
        return true;
    }
  }
  return false;
}

static void coalesce_moves(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  int *uses = malloc_or_die(sizeof(*uses) * vreg_count * 2);
  int *defs = uses + vreg_count;
  for (int pass = 0; pass < 4; ++pass) {
    for (int i = 0; i < vreg_count * 2; ++i)
      uses[i] = 0;
    count_vreg_uses(bbcon, uses, defs);
    bool changed = false;
    for (int ibb = 0; ibb < bbcon->len; ++ibb) {
      BB *bb = bbcon->data[ibb];
      Vector *irs = bb->irs;
      for (int j = 0; j + 1 < irs->len; ++j) {
        IR *ir = irs->data[j];
        IR *next = irs->data[j + 1];
        VReg *t = ir->dst;
        if (t != NULL && next->kind == IR_CAST && next->opr1 == t && uses[t->virt] == 1 &&
            defs[t->virt] == 1 && narrow_op_fits(ir, next->dst) &&
            coalescable_narrow(t, next->dst)) {
          // t = a OP b; v = (narrow)t  ->  v = a OP b: the result fits v
          ir->dst = next->dst;
          vec_remove_at(irs, j + 1);
          --uses[t->virt];
          --defs[t->virt];
          changed = true;
          continue;
        }
        if (t == NULL || next->kind != IR_MOV || next->opr1 != t)
          continue;
        switch (ir->kind) {
        case IR_ADD: case IR_SUB: case IR_MUL: case IR_DIV: case IR_MOD:
        case IR_BITAND: case IR_BITOR: case IR_BITXOR: case IR_LSHIFT: case IR_RSHIFT:
        case IR_COND: case IR_NEG: case IR_BITNOT: case IR_CAST: case IR_MOV:
        case IR_LOAD: case IR_BOFS: case IR_IOFS: case IR_CALL:
          break;
        default:
          continue;
        }
        if (uses[t->virt] != 1 || defs[t->virt] != 1 || !coalescable(t, next->dst))
          continue;
        ir->dst = next->dst;
        vec_remove_at(irs, j + 1);
        --uses[t->virt];
        --defs[t->virt];
        changed = true;
      }

      for (int j = 0; j < irs->len; ++j) {
        IR *ir = irs->data[j];
        VReg *t = ir->dst, *v = ir->opr1;
        if (t == NULL || v == NULL || uses[t->virt] != 1 || defs[t->virt] != 1)
          continue;
        if (ir->kind != IR_MOV &&
            !(ir->kind == IR_CAST && t->vsize >= v->vsize && t->vsize <= VRegSize4 &&
              !((t->flag | v->flag) & VRF_FLONUM)))
          continue;
        // Only the signedness differs (a cast), or a widening cast (a narrow
        // value is kept extended to 32 bits in its register: ARM32): fine
        // where it doesn't matter
        bool signonly = false;
        if (t->vsize != v->vsize) {
#if XCC_TARGET_ARCH == XCC_ARCH_ARM32
          VReg tmp = *v;
          tmp.vsize = t->vsize;
          tmp.flag = (tmp.flag & ~VRF_MASK) | (t->flag & VRF_MASK);
          if (!coalescable(t, &tmp) || t->vsize != VRegSize4)
            continue;
          signonly = true;
#else
          continue;
#endif
        } else if (!coalescable(t, v)) {
          int f = t->flag ^ v->flag;
          v->flag ^= f & VRF_UNSIGNED;
          signonly = coalescable(t, v);
          v->flag ^= f & VRF_UNSIGNED;
          if (!signonly)
            continue;
        }
        for (int k = j + 1; k < irs->len; ++k) {
          IR *use = irs->data[k];
          if (ir_reads(use, t)) {
            if (signonly) {
              switch (use->kind) {
              case IR_ADD: case IR_SUB: case IR_MUL: case IR_BITAND: case IR_BITOR:
              case IR_BITXOR: case IR_LSHIFT:
                if (use->dst != NULL && use->dst->vsize == t->vsize)
                  break;
                goto next_mov;
              case IR_LOAD:
                break;
              case IR_STORE:  // (the value's size is the store's)
                if (use->opr1 == t && t->vsize != v->vsize)
                  goto next_mov;
                break;
              default:
                goto next_mov;
              }
            }
            if (use->opr1 == t)
              use->opr1 = v;
            if (use->opr2 == t)
              use->opr2 = v;
            Vector *additional = use->additional_operands;
            if (additional != NULL) {
              for (int m = 0; m < additional->len; ++m) {
                if (additional->data[m] == t)
                  additional->data[m] = v;
              }
            }
            vec_remove_at(irs, j);
            --j;
            changed = true;
            break;
          }
          if (use->dst == v || use->kind == IR_ASM)
            break;
        }
      next_mov:;
      }
    }
    if (!changed)
      break;
  }
  free(uses);
}

// FPC: a conditional jump on a register given a constant earlier in the same
// block (a loop's first test after its counter's initialization, `while (1)`
// through a variable...) decided at compile time.
static bool fold_known_jumps(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  VReg **known = calloc_or_die(sizeof(*known) * vreg_count);
  bool changed = false;
  for (int ibb = 0; ibb < bbcon->len; ++ibb) {
    BB *bb = bbcon->data[ibb];
    Vector *irs = bb->irs;
    for (int j = 0; j < irs->len; ++j) {
      IR *ir = irs->data[j];
      if (ir->kind == IR_JMP && ir->jmp.cond != COND_ANY && ir->jmp.cond != COND_NONE &&
          ir->opr1 != NULL && ir->opr2 != NULL) {
        VReg *a = ir->opr1, *b = ir->opr2;
        if (!(a->flag & VRF_CONST) && known[a->virt] != NULL)
          a = known[a->virt];
        if (!(b->flag & VRF_CONST) && known[b->virt] != NULL)
          b = known[b->virt];
        if ((a->flag & VRF_CONST) && (b->flag & VRF_CONST) &&
            !((a->flag | b->flag) & VRF_FLONUM)) {
          // (The comparison is signed or not as its first operand is)
          bool taken = calc_const_cond(ir->jmp.cond, a, b);
          if (taken) {
            ir->jmp.cond = COND_ANY;
            ir->opr1 = ir->opr2 = NULL;
          } else {
            vec_remove_at(irs, j);
            --j;
          }
          changed = true;
          continue;
        }
      }
      VReg *dst = ir->dst;
      if (dst == NULL || (dst->flag & VRF_CONST))
        continue;
      if (ir->kind == IR_MOV && (ir->opr1->flag & VRF_CONST) &&
          !(dst->flag & (VRF_FORCEMEMORY | VRF_VOLATILEREG | VRF_FLONUM)) &&
          ir->opr1->vsize == dst->vsize && (ir->opr1->flag & VRF_UNSIGNED) == (dst->flag & VRF_UNSIGNED))
        known[dst->virt] = ir->opr1;
      else
        known[dst->virt] = NULL;
    }
    // Only within the block
    for (int j = 0; j < irs->len; ++j) {
      IR *ir = irs->data[j];
      if (ir->dst != NULL && !(ir->dst->flag & VRF_CONST))
        known[ir->dst->virt] = NULL;
    }
  }
  free(known);
  return changed;
}

// FPC: common subexpressions within a block: a pure integer computation
// (an address of a global, a row offset...) made again from the same
// operands gives way to the first one's result (both temporaries).
static bool same_operand(VReg *a, VReg *b) {
  if (a == b)
    return true;
  if (a == NULL || b == NULL)
    return false;
  return (a->flag & VRF_CONST) && (b->flag & VRF_CONST) && !((a->flag | b->flag) & VRF_FLONUM) &&
         a->fixnum == b->fixnum && a->vsize == b->vsize &&
         (a->flag & VRF_MASK) == (b->flag & VRF_MASK);
}

static bool cse_kind(IR *ir) {
  switch (ir->kind) {
  case IR_LOAD:
    return ir->additional_operands == NULL && ir->opr2 == NULL;
  case IR_IOFS: case IR_BOFS:
  case IR_ADD: case IR_SUB: case IR_MUL: case IR_BITAND: case IR_BITOR: case IR_BITXOR:
  case IR_LSHIFT: case IR_RSHIFT: case IR_NEG: case IR_BITNOT: case IR_CAST:
    return ir->additional_operands == NULL;
  default:
    return false;
  }
}

static bool same_expr(IR *a, IR *b) {
  if (a->kind != b->kind || a->dst->vsize != b->dst->vsize ||
      (a->dst->flag & (VRF_MASK | VRF_UNALIGNED)) != (b->dst->flag & (VRF_MASK | VRF_UNALIGNED)) ||
      !same_operand(a->opr1, b->opr1) || !same_operand(a->opr2, b->opr2))
    return false;
  switch (a->kind) {
  case IR_IOFS:
    return equal_name(a->iofs.label, b->iofs.label) && a->iofs.offset == b->iofs.offset &&
           a->iofs.global == b->iofs.global;
  case IR_BOFS:
    return a->bofs.frameinfo == b->bofs.frameinfo && a->bofs.offset == b->bofs.offset;
  case IR_LOAD:
    return a->load.offset == b->load.offset && a->load.unaligned == b->load.unaligned;
  default:
    return true;
  }
}

// What may write memory: a load made before is not known after it
static bool clobbers_memory(IR *ir) {
  switch (ir->kind) {
  case IR_STORE: case IR_STORE_S: case IR_CALL: case IR_ASM: case IR_SUBSP:
    return true;
  default:
    // (a variable in memory, its address taken: written by a MOV)
    return ir->dst != NULL && (ir->dst->flag & (VRF_FORCEMEMORY | VRF_VOLATILEREG));
  }
}

static VReg *cse_subst(VReg **repl, VReg *v) {
  while (v != NULL && !(v->flag & VRF_CONST) && repl[v->virt] != NULL)
    v = repl[v->virt];
  return v;
}

static void local_cse(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  int *uses = malloc_or_die(sizeof(*uses) * vreg_count * 2);
  int *defs = uses + vreg_count;
  for (int i = 0; i < vreg_count * 2; ++i)
    uses[i] = 0;
  count_vreg_uses(bbcon, uses, defs);
  VReg **repl = calloc_or_die(sizeof(*repl) * vreg_count);
  const int NG = VRF_CONST | VRF_PARAM | VRF_STACK_PARAM | VRF_FORCEMEMORY | VRF_VOLATILEREG |
                 VRF_FLONUM | VRF_NO_SPILL;
  bool any = false;
  Vector avail;
  vec_init(&avail);
  for (int ibb = 0; ibb < bbcon->len; ++ibb) {
    BB *bb = bbcon->data[ibb];
    Vector *irs = bb->irs;
    // A block whose only predecessor falls into it (it was the block
    // before) goes on with what that knew: an extended basic block
    if (ibb == 0 || bb->from_bbs->len != 1 || bb->from_bbs->data[0] != bbcon->data[ibb - 1])
      vec_clear(&avail);
    for (int j = 0; j < irs->len; ++j) {
      IR *ir = irs->data[j];
      if (any) {
        ir->opr1 = cse_subst(repl, ir->opr1);
        ir->opr2 = cse_subst(repl, ir->opr2);
      }
      VReg *dst = ir->dst;
      // A load: any destination but one in memory (the first one gives
      // its value by a MOV if it is not a temporary)
      bool load = ir->kind == IR_LOAD;
      if (dst != NULL && cse_kind(ir) && !(dst->flag & NG) && (defs[dst->virt] == 1 || load) &&
          !(load && (ir->opr1->flag & VRF_UNALIGNED)) &&
          !((ir->opr1 != NULL && (ir->opr1->flag & VRF_FLONUM)) ||
            (ir->opr2 != NULL && (ir->opr2->flag & VRF_FLONUM)))) {
        int k;
        for (k = 0; k < avail.len; ++k) {
          if (same_expr(avail.data[k], ir))
            break;
        }
        if (k < avail.len) {
          VReg *first = ((IR*)avail.data[k])->dst;
          if (defs[dst->virt] == 1 && defs[first->virt] == 1) {
            repl[dst->virt] = first;
            vec_remove_at(irs, j);
            --j;
            any = true;
            continue;
          }
          ir->kind = IR_MOV;
          ir->opr1 = first;
          ir->opr2 = NULL;
        } else if (dst != ir->opr1 && dst != ir->opr2) {  // (p = p->next: not known after)
          vec_push(&avail, ir);
          continue;
        }
      }
      if (clobbers_memory(ir)) {
        for (int k = 0; k < avail.len; ++k) {
          if (((IR*)avail.data[k])->kind == IR_LOAD) {
            vec_remove_at(&avail, k);
            --k;
          }
        }
      }
      if (dst != NULL && !(dst->flag & VRF_CONST)) {
        for (int k = 0; k < avail.len; ++k) {
          IR *a = avail.data[k];
          if (a->opr1 == dst || a->opr2 == dst || a->dst == dst) {
            vec_remove_at(&avail, k);
            --k;
          }
        }
      }
    }
  }
  if (any) {
    for (int ibb = 0; ibb < bbcon->len; ++ibb) {
      BB *bb = bbcon->data[ibb];
      if (bb->phis != NULL) {
        for (int j = 0; j < bb->phis->len; ++j) {
          Phi *phi = bb->phis->data[j];
          for (int k = 0; k < phi->params->len; ++k)
            phi->params->data[k] = cse_subst(repl, phi->params->data[k]);
        }
      }
      for (int j = 0; j < bb->irs->len; ++j) {
        IR *ir = bb->irs->data[j];
        ir->opr1 = cse_subst(repl, ir->opr1);
        ir->opr2 = cse_subst(repl, ir->opr2);
        Vector *additional = ir->additional_operands;
        if (additional != NULL) {
          for (int k = 0; k < additional->len; ++k)
            additional->data[k] = cse_subst(repl, additional->data[k]);
        }
        if (ir->kind == IR_CALL) {
          for (int k = 0; k < ir->call->total_arg_count; ++k)
            ir->call->args[k] = cse_subst(repl, ir->call->args[k]);
        }
      }
    }
  }
  free(repl);
  free(uses);
}

// FPC: loop-invariant code motion.  A loop is a backward jump's range of
// blocks (xcc lays loops out contiguously), entered only at its head, from
// the block before it.  A computation in it that can't trap, whose operands
// are not changed in the loop and whose result is a temporary assigned only
// there moves to that block (an address of a global, a constant, the row
// offset of an array access...).
static int bb_index(BBContainer *bbcon, BB *bb) {
  for (int i = 0; i < bbcon->len; ++i) {
    if (bbcon->data[i] == bb)
      return i;
  }
  return -1;
}

static bool bb_falls_through(BB *bb) {
  if (bb->irs->len == 0)
    return true;
  IR *ir = bb->irs->data[bb->irs->len - 1];
  return !((ir->kind == IR_JMP && ir->jmp.cond == COND_ANY) || ir->kind == IR_TJMP);
}

static bool hoistable_kind(IR *ir) {
  switch (ir->kind) {
  case IR_IOFS: case IR_BOFS:
  case IR_ADD: case IR_SUB: case IR_MUL: case IR_BITAND: case IR_BITOR: case IR_BITXOR:
  case IR_LSHIFT: case IR_RSHIFT: case IR_NEG: case IR_BITNOT: case IR_CAST:
    return true;
  default:
    return false;
  }
}

// FPC: a constant that costs an instruction or more each time it is used
// (a float's is loaded from memory; on ARM32 an integer that is not an
// immediate of its instruction is made in a register first)
#if XCC_TARGET_ARCH == XCC_ARCH_ARM32
static bool thumb_modimm(uint32_t v) {
  if (v < 256)
    return true;
  uint32_t b = v & 0xff;
  if (v == (b | (b << 16)) || v == ((b << 8) | (b << 24)) || v == (b * 0x01010101u))
    return true;
  for (int rot = 8; rot < 32; ++rot) {
    uint32_t x = (v << rot) | (v >> (32 - rot));  // rotate left: undo a rotate right
    if (x >= 0x80 && x <= 0xff)
      return true;
  }
  return false;
}
#endif

static bool costly_const(IR *ir, VReg *c, int which) {
  if (!(c->flag & VRF_CONST))
    return false;
  if (c->flag & VRF_FLONUM) {
    // (a float: s registers enough; a double would take a pair of core ones)
    return c->vsize == VRegSize4 && ir->kind != IR_PUSHARG && ir->kind != IR_RESULT &&
           ir->kind != IR_CAST;
  }
#if XCC_TARGET_ARCH == XCC_ARCH_ARM32
  // (the first operand: an operation that takes it second too)
  if (c->vsize > VRegSize4 || (ir->dst == NULL && ir->kind != IR_JMP) ||
      (which == 1 && ir->kind != IR_ADD && ir->kind != IR_MUL && ir->kind != IR_BITAND &&
       ir->kind != IR_BITOR && ir->kind != IR_BITXOR && ir->kind != IR_COND && ir->kind != IR_JMP))
    return false;
  uint32_t v = (uint32_t)c->fixnum;
  switch (ir->kind) {
  case IR_MUL: case IR_DIV: case IR_MOD:
    return true;  // (moved to a register at each use: tweak_irs)
  case IR_ADD: case IR_SUB:
    return !(v < 4096 || -v < 4096 || thumb_modimm(v) || thumb_modimm(-v));
  case IR_BITAND: case IR_BITOR:
    return !(thumb_modimm(v) || thumb_modimm(~v));
  case IR_BITXOR:
    return !thumb_modimm(v);
  case IR_COND: case IR_JMP:
    return !(thumb_modimm(v) || thumb_modimm(-v));
  default:
    return false;
  }
#else
  (void)ir;
  (void)which;
  return false;
#endif
}

static bool defined_in(Vector *irs, int from, int to, VReg *v);

// A load's address as base + offset: through an ADD/SUB of a constant just
// before it in its block (what tweak_irs folds into the load later)
static VReg *load_addr(Vector *irs, int pos, IR *ld, int64_t *pofs) {
  VReg *a = ld->opr1;
  *pofs = ld->load.offset;
  for (int k = pos - 1; k >= 0; --k) {
    IR *d = irs->data[k];
    if (d->dst != a)
      continue;
    if ((d->kind == IR_ADD || d->kind == IR_SUB) && (d->opr2->flag & VRF_CONST) &&
        !(d->opr2->flag & VRF_FLONUM) && !(d->opr1->flag & (VRF_CONST | VRF_FORCEMEMORY)) &&
        (d->opr1->flag & VRF_UNALIGNED) == (a->flag & VRF_UNALIGNED) &&
        !defined_in(irs, k + 1, pos, d->opr1)) {
      *pofs += d->kind == IR_ADD ? d->opr2->fixnum : -d->opr2->fixnum;
      return d->opr1;
    }
    break;
  }
  return a;
}

static void hoist_loop_invariants(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  int *uses = malloc_or_die(sizeof(*uses) * vreg_count * 2);
  int *defs = uses + vreg_count;
  for (int i = 0; i < vreg_count * 2; ++i)
    uses[i] = 0;
  count_vreg_uses(bbcon, uses, defs);
  unsigned char *inloop = malloc_or_die(vreg_count);
  unsigned char *hoisted = malloc_or_die(vreg_count);
  const int NG = VRF_CONST | VRF_PARAM | VRF_STACK_PARAM | VRF_FORCEMEMORY | VRF_VOLATILEREG |
                 VRF_FLONUM | VRF_NO_SPILL;

  for (int j = 0; j < bbcon->len; ++j) {
    BB *latch = bbcon->data[j];
    if (latch->irs->len == 0)
      continue;
    IR *jmp = latch->irs->data[latch->irs->len - 1];
    if (jmp->kind != IR_JMP)
      continue;
    int h = bb_index(bbcon, jmp->jmp.bb);
    if (h <= 0 || h > j)
      continue;
    BB *pre = bbcon->data[h - 1];
    if (!bb_falls_through(pre) || pre->next != bbcon->data[h])
      continue;
    // Innermost loops only (what moves out of a loop stays in a register
    // through all of it: the pressure on the registers in outer loops)
    bool inner = true;
    for (int k = h; k < j && inner; ++k) {
      BB *b = bbcon->data[k];
      if (b->irs->len > 0) {
        IR *last = b->irs->data[b->irs->len - 1];
        if (last->kind == IR_JMP) {
          int t = bb_index(bbcon, last->jmp.bb);
          if (t >= h && t <= k)
            inner = false;
        }
      }
    }
    // Entered only at the head, and only from `pre` and the loop itself
    bool ok = true;
    for (int k = h; k <= j && ok; ++k) {
      Vector *from = ((BB*)bbcon->data[k])->from_bbs;
      for (int m = 0; m < from->len; ++m) {
        BB *f = from->data[m];
        if (k == h && f == pre)
          continue;
        int fi = bb_index(bbcon, f);
        if (fi < h || fi > j) {
          ok = false;
          break;
        }
      }
    }
    if (!ok)
      continue;

    for (int i = 0; i < vreg_count; ++i)
      inloop[i] = false;
    for (int k = h; k <= j; ++k) {
      Vector *irs = ((BB*)bbcon->data[k])->irs;
      for (int m = 0; m < irs->len; ++m) {
        IR *ir = irs->data[m];
        if (ir->dst != NULL && !(ir->dst->flag & VRF_CONST))
          inloop[ir->dst->virt] = true;
      }
    }

    int at = pre->irs->len;
    if (at > 0 && ((IR*)pre->irs->data[at - 1])->kind == IR_JMP)
      --at;
    int nhoisted = 0;
    for (int i = 0; i < vreg_count; ++i)
      hoisted[i] = false;
    for (int k = h; k <= j; ++k) {
      Vector *irs = ((BB*)bbcon->data[k])->irs;
      for (int m = 0; m < irs->len; ++m) {
        IR *ir = irs->data[m];
        VReg *dst = ir->dst;
        if (dst == NULL || !hoistable_kind(ir) || (dst->flag & NG) || defs[dst->virt] != 1)
          continue;
        VReg *oprs[] = {ir->opr1, ir->opr2};
        bool inv = ir->additional_operands == NULL;
        for (int o = 0; o < 2 && inv; ++o) {
          VReg *opr = oprs[o];
          if (opr != NULL && (opr->flag & VRF_FLONUM || (!(opr->flag & VRF_CONST) &&
                                                         (inloop[opr->virt] || (opr->flag & VRF_FORCEMEMORY)))))
            inv = false;
        }
        if (!inv)
          continue;
        // Worth a register through the loop: an address of a global (two
        // instructions), a multiplication, or what is made from a value
        // moved out already; not a base + constant (an address the
        // load's offset takes), a few at most.  Out of a loop with loops
        // in it: an address of a global only.
        bool worth = ir->kind == IR_IOFS || (inner && ir->kind == IR_MUL);
        for (int o = 0; o < 2 && !worth; ++o) {
          VReg *opr = oprs[o];
          if (opr != NULL && !(opr->flag & VRF_CONST) && hoisted[opr->virt])
            worth = true;
        }
        if (!worth || nhoisted >= (inner ? 6 : 2))
          continue;
        vec_remove_at(irs, m);
        --m;
        vec_insert(pre->irs, at++, ir);
        inloop[dst->virt] = false;
        hoisted[dst->virt] = true;
        ++nhoisted;
      }
    }

    if (!inner)
      continue;

    // A load in a loop that writes no memory, from an address it doesn't
    // change, made already before it (its first test, in `pre`): its value
    // from there (no speculative load: it was made anyway)
    bool writes = false;
    for (int k = h; k <= j && !writes; ++k) {
      Vector *irs = ((BB*)bbcon->data[k])->irs;
      for (int m = 0; m < irs->len && !writes; ++m)
        writes = clobbers_memory(irs->data[m]);
    }
    for (int k = h; k <= j && !writes; ++k) {
      Vector *irs = ((BB*)bbcon->data[k])->irs;
      for (int m = 0; m < irs->len; ++m) {
        IR *ir = irs->data[m];
        if (ir->kind != IR_LOAD || ir->opr2 != NULL || ir->additional_operands != NULL ||
            (ir->dst->flag & (VRF_FORCEMEMORY | VRF_VOLATILEREG)) ||
            (ir->opr1->flag & (VRF_CONST | VRF_UNALIGNED)))
          continue;
        int64_t ofs;
        VReg *base = load_addr(irs, m, ir, &ofs);
        if ((base->flag & VRF_CONST) || inloop[base->virt])
          continue;
        Vector *pirs = pre->irs;
        for (int q = pirs->len; --q >= 0; ) {
          IR *pl = pirs->data[q];
          if (clobbers_memory(pl) || pl->dst == base)
            break;
          int64_t pofs;
          if (pl->kind == IR_LOAD && pl->opr2 == NULL && pl->additional_operands == NULL &&
              pl->dst->vsize == ir->dst->vsize &&
              (pl->dst->flag & VRF_MASK) == (ir->dst->flag & VRF_MASK) &&
              pl->load.unaligned == ir->load.unaligned &&
              !(pl->opr1->flag & (VRF_CONST | VRF_UNALIGNED)) &&
              load_addr(pirs, q, pl, &pofs) == base && pofs == ofs &&
              !inloop[pl->dst->virt] && !(pl->dst->flag & (VRF_FORCEMEMORY | VRF_VOLATILEREG)) &&
              !defined_in(pirs, q + 1, pirs->len, pl->dst)) {
            ir->kind = IR_MOV;
            ir->opr1 = pl->dst;
            break;
          }
        }
      }
    }

    // Constants that cost at each use (a float's load...): in registers
    // given them before the loop (innermost loops, a few)
    VReg *consts[6], *regs[6];
    int nconst = 0;
    for (int k = h; k <= j; ++k) {
      Vector *irs = ((BB*)bbcon->data[k])->irs;
      for (int m = 0; m < irs->len; ++m) {
        IR *ir = irs->data[m];
        if (ir->kind == IR_MOV && !(ir->dst->flag & VRF_FLONUM))
          continue;
        VReg **oprs[] = {&ir->opr1, &ir->opr2};
        for (int o = 0; o < 2; ++o) {
          VReg *cv = *oprs[o];
          if (cv == NULL || !costly_const(ir, cv, o + 1))
            continue;
          int n;
          for (n = 0; n < nconst; ++n) {
            VReg *x = consts[n];
            if (x->vsize == cv->vsize && (x->flag & VRF_MASK) == (cv->flag & VRF_MASK) &&
                ((x->flag & VRF_FLONUM) ? x->flonum.value == cv->flonum.value
                                        : x->fixnum == cv->fixnum))
              break;
          }
          if (n == nconst) {
            if (nconst >= 6)
              continue;
            VReg *r = reg_alloc_spawn(ra, cv->vsize, cv->flag & VRF_MASK);
            vec_insert(pre->irs, at++, new_ir_mov(r, cv));
            consts[nconst] = cv;
            regs[nconst++] = r;
          }
          *oprs[o] = regs[n];
        }
      }
    }
  }
  free(hoisted);
  free(inloop);
  free(uses);
}

// FPC: dead definitions: a computation whose result no later instruction
// reads (the initial value of a variable assigned again before any use...),
// with the registers' flow known (analyze_reg_flow).  Returns whether it
// removed any (the flow is to be analyzed again).
bool remove_dead_defs(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  unsigned char *live = malloc_or_die(vreg_count);
  bool changed = false;
  for (int ibb = 0; ibb < bbcon->len; ++ibb) {
    BB *bb = bbcon->data[ibb];
    for (int i = 0; i < vreg_count; ++i)
      live[i] = false;
    for (int i = 0; i < bb->out_regs->len; ++i) {
      VReg *vreg = bb->out_regs->data[i];
      live[vreg->virt] = true;
    }
    Vector *irs = bb->irs;
    for (int j = irs->len; --j >= 0; ) {
      IR *ir = irs->data[j];
      VReg *dst = ir->dst;
      if (dst != NULL && !(dst->flag & VRF_CONST)) {
        if (!live[dst->virt] && !(dst->flag & (VRF_FORCEMEMORY | VRF_VOLATILEREG))) {
          bool pure;
          switch (ir->kind) {
          case IR_MOV: case IR_ADD: case IR_SUB: case IR_MUL: case IR_BITAND: case IR_BITOR:
          case IR_BITXOR: case IR_LSHIFT: case IR_RSHIFT: case IR_NEG: case IR_BITNOT:
          case IR_CAST: case IR_COND: case IR_BOFS: case IR_IOFS:
            pure = true;
            break;
          default:
            pure = false;
            break;
          }
          if (pure) {
            vec_remove_at(irs, j);
            changed = true;
            continue;
          }
        }
        live[dst->virt] = false;
      }
      VReg *operands[] = {ir->opr1, ir->opr2};
      int n = 2 + (ir->additional_operands != NULL ? ir->additional_operands->len : 0);
      for (int k = 0; k < n; ++k) {
        VReg *vreg = k < 2 ? operands[k] : ir->additional_operands->data[k - 2];
        if (vreg != NULL && !(vreg->flag & VRF_CONST))
          live[vreg->virt] = true;
      }
    }
  }
  free(live);
  return changed;
}

//

// FPC: conditions as control flow.
//
//   (A) a diamond giving a variable 0 or 1 on a condition:
//         if (a cc b) v = 0; else v = 1;   ->  v = COND(a !cc b)
//       (v = (t != 0) is a MOV when t is 0 or 1 already)
//   (M) a block merged into its only predecessor when that falls into it
//   (B) a jump on a COND, or on an & or | of two CONDs (bitwise on 0/1
//       values: the C of `(c >= '0') & (c <= '9')`, or what (A) left):
//         t = COND(a cc b); JEQ t, 0 -> F  ->  J(a !cc b) -> F
//         t = c1 & c2; JEQ t, 0 -> F       ->  J(!c1) -> F; J(!c2) -> F
//   (C) two jumps out of a range, one compare:
//         JLT x, lo -> F; JGT x, hi -> F   ->  JGT (unsigned)(x - lo), hi - lo -> F

// Whether v is 0 or 1 (from its last definition before pos in the block)
static bool is_bool_vreg(Vector *irs, int pos, VReg *v, int depth) {
  if (v->flag & VRF_CONST)
    return !(v->flag & VRF_FLONUM) && (v->fixnum == 0 || v->fixnum == 1);
  if ((v->flag & (VRF_FLONUM | VRF_FORCEMEMORY | VRF_VOLATILEREG)) || depth <= 0)
    return false;
  for (int k = pos - 1; k >= 0; --k) {
    IR *ir = irs->data[k];
    if (ir->dst != v)
      continue;
    switch (ir->kind) {
    case IR_COND:
      return true;
    case IR_BITAND:
      return is_bool_vreg(irs, k, ir->opr1, depth - 1) || is_bool_vreg(irs, k, ir->opr2, depth - 1);
    case IR_BITOR: case IR_BITXOR:
      return is_bool_vreg(irs, k, ir->opr1, depth - 1) && is_bool_vreg(irs, k, ir->opr2, depth - 1);
    case IR_MOV:
      return is_bool_vreg(irs, k, ir->opr1, depth - 1);
    case IR_CAST:
      return !(ir->opr1->flag & VRF_FLONUM) && is_bool_vreg(irs, k, ir->opr1, depth - 1);
    default:
      return false;
    }
  }
  return false;
}

static bool plain_int_vreg(VReg *v) {
  return !(v->flag & (VRF_CONST | VRF_FLONUM | VRF_FORCEMEMORY | VRF_VOLATILEREG | VRF_VOLATILE));
}

// A block of `v = 0 or 1` (and a jump or falling into the next): its join
static bool const_assign_bb(BB *bb, VReg **pv, int64_t *pc, BB **pjoin) {
  Vector *irs = bb->irs;
  if (irs->len < 1 || irs->len > 2)
    return false;
  IR *mov = irs->data[0];
  if (mov->kind != IR_MOV || !(mov->opr1->flag & VRF_CONST) || (mov->opr1->flag & VRF_FLONUM) ||
      !plain_int_vreg(mov->dst) || (mov->opr1->fixnum != 0 && mov->opr1->fixnum != 1))
    return false;
  BB *join;
  if (irs->len == 2) {
    IR *j = irs->data[1];
    if (j->kind != IR_JMP || j->jmp.cond != COND_ANY)
      return false;
    join = j->jmp.bb;
  } else {
    join = bb->next;
  }
  if (join == NULL)
    return false;
  *pv = mov->dst;
  *pc = mov->opr1->fixnum;
  *pjoin = join;
  return true;
}

static bool cond_diamonds(BBContainer *bbcon) {
  bool changed = false;
  for (int i = 0; i < bbcon->len; ++i) {
    BB *x = bbcon->data[i];
    IR *jmp = is_last_jmp(x);
    if (jmp == NULL || jmp->jmp.cond == COND_ANY || jmp->jmp.cond == COND_NONE ||
        jmp->opr1 == NULL || jmp->opr2 == NULL || ((jmp->opr1->flag | jmp->opr2->flag) & VRF_FLONUM))
      continue;
    BB *t = x->next, *f = jmp->jmp.bb, *jt, *jf;
    VReg *vt, *vf;
    int64_t ct, cf;
    if (t == NULL || f == t || f == x || t == x || t->from_bbs->len != 1 ||
        f->from_bbs->len != 1 || !const_assign_bb(t, &vt, &ct, &jt) ||
        !const_assign_bb(f, &vf, &cf, &jf) || vt != vf || jt != jf || ct == cf || jt == t ||
        jt == f)
      continue;
    // t runs when !(a cc b), setting ct: v = (a want b)
    enum ConditionKind want = ct == 1 ? invert_cond(jmp->jmp.cond) : jmp->jmp.cond;
    VReg *a = jmp->opr1, *b = jmp->opr2;
    Vector *irs = x->irs;
    vec_pop(irs);
    IR *ir;
    if (want == COND_NE && (b->flag & VRF_CONST) && b->fixnum == 0 &&
        is_bool_vreg(irs, irs->len, a, 4)) {
      ir = new_ir_bop_raw(a->vsize == vt->vsize ? IR_MOV : IR_CAST, vt, a, NULL);
    } else {
      ir = new_ir_bop_raw(IR_COND, vt, a, b);
      ir->cond.kind = want;
    }
    vec_push(irs, ir);
    vec_push(irs, new_ir_jmp(jt));
    vec_clear(t->irs);
    vec_clear(f->irs);
    changed = true;
  }
  return changed;
}

static void replace_from_bb(BB *succ, BB *old, BB *new) {
  if (succ == NULL)
    return;
  for (int k = 0; k < succ->from_bbs->len; ++k) {
    if (succ->from_bbs->data[k] == old)
      succ->from_bbs->data[k] = new;
  }
}

static bool merge_blocks(BBContainer *bbcon) {
  bool changed = false;
  for (int i = 0; i + 1 < bbcon->len; ++i) {
    BB *p = bbcon->data[i], *b = bbcon->data[i + 1];
    if (p->next != b || b->from_bbs->len != 1 || b->from_bbs->data[0] != p)
      continue;
    Vector *irs = p->irs;
    IR *last = irs->len > 0 ? irs->data[irs->len - 1] : NULL;
    if (last != NULL && (last->kind == IR_TJMP ||
                         (last->kind == IR_JMP && (last->jmp.cond != COND_ANY || last->jmp.bb != b))))
      continue;
    if (last != NULL && last->kind == IR_JMP)
      vec_pop(irs);
    for (int j = 0; j < b->irs->len; ++j)
      vec_push(irs, b->irs->data[j]);
    p->next = b->next;
    vec_remove_at(bbcon, i + 1);
    // b's successors come from p now
    IR *jmp = is_last_jmp(p), *tjmp = is_last_jtable(p);
    if (jmp != NULL)
      replace_from_bb(jmp->jmp.bb, b, p);
    if (tjmp != NULL) {
      for (size_t k = 0; k < tjmp->tjmp.len; ++k)
        replace_from_bb(tjmp->tjmp.bbs[k], b, p);
    }
    if ((jmp == NULL || jmp->jmp.cond != COND_ANY) && tjmp == NULL)
      replace_from_bb(p->next, b, p);
    --i;
    changed = true;
  }
  return changed;
}

static bool defined_in(Vector *irs, int from, int to, VReg *v) {
  for (int k = from; k < to; ++k) {
    if (((IR*)irs->data[k])->dst == v)
      return true;
  }
  return false;
}

static int def_index(Vector *irs, int before, VReg *v) {
  for (int k = before - 1; k >= 0; --k) {
    if (((IR*)irs->data[k])->dst == v)
      return k;
  }
  return -1;
}

// A COND usable as a jump at the block's end (pos: the jump's index)
static bool movable_cond(Vector *irs, int k, int pos, int *uses, int *defs) {
  IR *ir = irs->data[k];
  if (ir->kind != IR_COND || uses[ir->dst->virt] != 1 || defs[ir->dst->virt] != 1)
    return false;
  VReg *a = ir->opr1, *b = ir->opr2;
  if (((a->flag | b->flag) & VRF_FLONUM) || (!(a->flag & VRF_CONST) && !plain_int_vreg(a)) ||
      (!(b->flag & VRF_CONST) && !plain_int_vreg(b)))
    return false;
  return (a->flag & VRF_CONST || !defined_in(irs, k + 1, pos, a)) &&
         (b->flag & VRF_CONST || !defined_in(irs, k + 1, pos, b));
}

static IR *cond_jump(IR *cond, bool invert, BB *to) {
  IR *j = new_ir_jmp(to);
  j->jmp.cond = invert ? invert_cond(cond->cond.kind) : cond->cond.kind;
  j->opr1 = cond->opr1;
  j->opr2 = cond->opr2;
  return j;
}

static int cmp_desc(const void *a, const void *b) {
  return *(const int*)b - *(const int*)a;
}

static bool split_cond_jumps(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  int *uses = calloc_or_die(sizeof(*uses) * vreg_count * 2);
  int *defs = uses + vreg_count;
  count_vreg_uses(bbcon, uses, defs);
  bool changed = false;
  for (int i = 0; i < bbcon->len; ++i) {
    BB *x = bbcon->data[i];
    Vector *irs = x->irs;
    IR *jmp = is_last_jmp(x);
    if (jmp == NULL || (jmp->jmp.cond != COND_EQ && jmp->jmp.cond != COND_NE) ||
        !(jmp->opr2->flag & VRF_CONST) || jmp->opr2->fixnum != 0 ||
        (jmp->opr1->flag & (VRF_CONST | VRF_FLONUM)))
      continue;
    int pos = irs->len - 1;
    int drop[8], ndrop = 0;
    VReg *t = jmp->opr1;
    int k;
    IR *src = NULL;
    for (;;) {
      if (!plain_int_vreg(t) || uses[t->virt] != 1 || defs[t->virt] != 1 ||
          (k = def_index(irs, pos, t)) < 0 || ndrop >= 5)
        break;
      IR *d = irs->data[k];
      if (d->kind == IR_MOV || (d->kind == IR_CAST && !(d->opr1->flag & VRF_FLONUM))) {
        drop[ndrop++] = k;
        t = d->opr1;
        continue;
      }
      if (d->kind == IR_COND || d->kind == IR_BITAND || d->kind == IR_BITOR)
        src = d;
      break;
    }
    if (src == NULL)
      continue;
    BB *f = jmp->jmp.bb, *next = x->next;
    bool jeq = jmp->jmp.cond == COND_EQ;
    if (src->kind == IR_COND) {
      if (!movable_cond(irs, k, irs->len - 1, uses, defs))
        continue;
      drop[ndrop++] = k;
      IR *j = cond_jump(src, jeq, f);
      qsort(drop, ndrop, sizeof(int), cmp_desc);
      vec_pop(irs);
      for (int m = 0; m < ndrop; ++m)
        vec_remove_at(irs, drop[m]);
      vec_push(irs, j);
      changed = true;
      continue;
    }
    // c1 & c2, c1 | c2 (0 or 1 each)
    if (next == NULL || src->opr1->flag & VRF_CONST || src->opr2->flag & VRF_CONST)
      continue;
    int k1 = def_index(irs, k, src->opr1), k2 = def_index(irs, k, src->opr2);
    if (k1 < 0 || k2 < 0 || k1 == k2 || !movable_cond(irs, k1, irs->len - 1, uses, defs) ||
        !movable_cond(irs, k2, irs->len - 1, uses, defs))
      continue;
    IR *c1 = irs->data[k1], *c2 = irs->data[k2];
    IR *j1, *j2;
    if (src->kind == IR_BITAND) {
      if (jeq) {  // to f if !c1 || !c2
        j1 = cond_jump(c1, true, f);
        j2 = cond_jump(c2, true, f);
      } else {    // to f if c1 && c2
        j1 = cond_jump(c1, true, next);
        j2 = cond_jump(c2, false, f);
      }
    } else {
      if (jeq) {  // to f if !c1 && !c2
        j1 = cond_jump(c1, false, next);
        j2 = cond_jump(c2, true, f);
      } else {    // to f if c1 || c2
        j1 = cond_jump(c1, false, f);
        j2 = cond_jump(c2, false, f);
      }
    }
    drop[ndrop++] = k;
    drop[ndrop++] = k1;
    drop[ndrop++] = k2;
    qsort(drop, ndrop, sizeof(int), cmp_desc);
    vec_pop(irs);
    for (int m = 0; m < ndrop; ++m)
      vec_remove_at(irs, drop[m]);
    vec_push(irs, j1);
    BB *x2 = new_bb();
    vec_push(x2->irs, j2);
    x2->next = next;
    x->next = x2;
    vec_insert(bbcon, i + 1, x2);
    changed = true;
  }
  free(uses);
  return changed;
}

// A jump's condition on x as a bound broken: x < lo (low) or x > hi (high)
static bool range_violation(enum ConditionKind cc, int64_t k, bool *low, int64_t *bound) {
  switch (cc) {
  case COND_LT: *low = true; *bound = k; return true;
  case COND_LE: *low = true; *bound = k + 1; return true;
  case COND_GT: *low = false; *bound = k; return true;
  case COND_GE: *low = false; *bound = k - 1; return true;
  default: return false;
  }
}

static bool fold_range_checks(RegAlloc *ra, BBContainer *bbcon) {
  bool changed = false;
  for (int i = 0; i + 1 < bbcon->len; ++i) {
    BB *x1 = bbcon->data[i], *x2 = x1->next;
    IR *j1 = is_last_jmp(x1);
    if (j1 == NULL || x2 == NULL || x2->irs->len != 1 || x2->from_bbs->len != 1 ||
        j1->jmp.cond == COND_ANY || j1->jmp.cond == COND_NONE)
      continue;
    IR *j2 = x2->irs->data[0];
    VReg *x = j1->opr1;
    if (j2->kind != IR_JMP || j2->jmp.cond == COND_ANY || j2->jmp.cond == COND_NONE ||
        j2->opr1 != x || !plain_int_vreg(x) || x->vsize > VRegSize4 ||
        !(j1->opr2->flag & VRF_CONST) || !(j2->opr2->flag & VRF_CONST) ||
        ((j1->opr2->flag | j2->opr2->flag) & VRF_FLONUM))
      continue;
    BB *f = j2->jmp.bb;
    bool same = j1->jmp.bb == f;  // out of the range: to f
    if (!same && j1->jmp.bb != x2->next)  // in the range: to f
      continue;
    bool low1, low2;
    int64_t b1, b2;
    if (!range_violation(j1->jmp.cond, j1->opr2->fixnum, &low1, &b1) ||
        !range_violation(same ? j2->jmp.cond : invert_cond(j2->jmp.cond), j2->opr2->fixnum,
                         &low2, &b2) ||
        low1 == low2)
      continue;
    int64_t lo = low1 ? b1 : b2, hi = low1 ? b2 : b1;
    bool is_unsigned = (x->flag & VRF_UNSIGNED) != 0;
    int64_t vmin = is_unsigned ? 0 : INT32_MIN, vmax = is_unsigned ? (int64_t)UINT32_MAX : INT32_MAX;
    if (lo < vmin || hi > vmax || lo > hi)
      continue;
    Vector *irs = x1->irs;
    vec_pop(irs);
    // (x as it is: a narrow value is kept extended to 32 bits in its
    // register, ARM32)
    VReg *x4 = x;
#if XCC_TARGET_ARCH != XCC_ARCH_ARM32
    if (x->vsize != VRegSize4) {
      x4 = reg_alloc_spawn(ra, VRegSize4, VRF_UNSIGNED);
      vec_push(irs, new_ir_bop_raw(IR_CAST, x4, x, NULL));
    }
#endif
    VReg *d = x4;
    if (lo != 0) {
      d = reg_alloc_spawn(ra, VRegSize4, VRF_UNSIGNED);
      VReg *c = reg_alloc_spawn_const(ra, lo, VRegSize4, 0);
      vec_push(irs, new_ir_bop_raw(IR_SUB, d, x4, c));
    } else if (!is_unsigned) {
      d = reg_alloc_spawn(ra, VRegSize4, VRF_UNSIGNED);
      vec_push(irs, new_ir_bop_raw(IR_CAST, d, x4, NULL));
    }
    IR *j = new_ir_jmp(f);
    j->jmp.cond = same ? COND_GT : COND_LE;
    j->opr1 = d;
    j->opr2 = reg_alloc_spawn_const(ra, hi - lo, VRegSize4, VRF_UNSIGNED);
    vec_push(irs, j);
    vec_clear(x2->irs);
    changed = true;
  }
  return changed;
}

// FPC: jump threading.  A block that ends knowing constants (`state = 4;`,
// `carry = 1;`) and goes on into blocks that test them: the path is copied
// after the block, up to the last test decided, so that the tests known go
// (a state machine's switch in a loop jumps straight to the next state's
// case; a flag tested after a join is not).  The path: blocks that don't
// call, at most THREAD_MAX_IRS instructions copied, a test not decided
// copied as it is (or inverted, to follow its target).

#define THREAD_MAX_IRS   16
#define THREAD_MAX_BBS   8
#define THREAD_FUNC_IRS  256

typedef struct {
  BB *bb;
  int how;  // 0: falls (or jumps) on, 1: decided, 2: a test copied, following its
            // fall, 3: a test copied inverted, following its target
} ThreadStep;

typedef struct {
  RegAlloc *ra;
  VReg **known;     // [virt]: a constant, or NULL
  int known_len;
  bool walking;     // On the path: constants assigned there are not known
                    // (deciding on them is the work of their own block)
  Vector undo;      // <(virt, old)> to put known back
  ThreadStep steps[THREAD_MAX_BBS];       // The path walked
  ThreadStep best_steps[THREAD_MAX_BBS];  // The best: to its last decision
  int best;         // its steps (0: none)
  BB *best_dest;    // and where it leads
  int budget;
} Thread;

static int ra_vreg_count(RegAlloc *ra) {
  return ra->vregs->len;
}

static void thread_set(Thread *th, VReg *v, VReg *c) {
  vec_push(&th->undo, INT2VOIDP(v->virt));
  vec_push(&th->undo, th->known[v->virt]);
  th->known[v->virt] = c;
}

static void thread_undo(Thread *th, int mark) {
  while (th->undo.len > mark) {
    VReg *old = vec_pop(&th->undo);
    int virt = VOIDP2INT(vec_pop(&th->undo));
    th->known[virt] = old;
  }
}

static bool thread_value(Thread *th, VReg *opr, int64_t *pv) {
  if (opr->flag & VRF_FLONUM)
    return false;
  VReg *c = opr->flag & VRF_CONST ? opr : th->known[opr->virt];
  if (c == NULL)
    return false;
  *pv = c->fixnum;
  return true;
}

// What an instruction makes known (or not)
static void thread_track(Thread *th, IR *ir) {
  VReg *dst = ir->dst;
  if (dst == NULL || (dst->flag & VRF_CONST))
    return;
  int64_t v;
  if ((ir->kind == IR_MOV || ir->kind == IR_CAST) && plain_int_vreg(dst) &&
      dst->vsize <= VRegSize4 && ir->opr1->vsize <= VRegSize4 &&
      !(th->walking && (ir->opr1->flag & VRF_CONST)) && thread_value(th, ir->opr1, &v)) {
    v = wrap_value(v, 1 << dst->vsize, (dst->flag & VRF_UNSIGNED) != 0);
    thread_set(th, dst, reg_alloc_spawn_const(th->ra, v, dst->vsize, dst->flag & VRF_UNSIGNED));
  } else if (th->known[dst->virt] != NULL) {
    thread_set(th, dst, NULL);
  }
}

static bool thread_copyable(IR *ir) {
  switch (ir->kind) {
  case IR_CALL: case IR_ASM: case IR_SUBSP: case IR_PUSHARG: case IR_KEEP: case IR_TJMP:
  case IR_RESULT:
    return false;
  default:
    return true;
  }
}

// The decided successor of a block's test, or NULL
static BB *thread_decide(Thread *th, BB *bb, IR *last) {
  int64_t a, b;
  if (last->kind == IR_TJMP) {
    if (!thread_value(th, last->opr1, &a) || a < 0 || (size_t)a >= last->tjmp.len)
      return NULL;
    return last->tjmp.bbs[a];
  }
  if (last->kind != IR_JMP || last->jmp.cond == COND_ANY || last->jmp.cond == COND_NONE ||
      !thread_value(th, last->opr1, &a) || !thread_value(th, last->opr2, &b))
    return NULL;
  VReg ca = *last->opr1, cb = *last->opr2;
  ca.flag = VRF_CONST | (last->opr1->flag & VRF_UNSIGNED);
  cb.flag = VRF_CONST | (last->opr1->flag & VRF_UNSIGNED);
  ca.fixnum = a;
  cb.fixnum = b;
  return calc_const_cond(last->jmp.cond, &ca, &cb) ? last->jmp.bb : bb->next;
}

static void thread_walk(Thread *th, BB *bb, int depth) {
  if (bb == NULL || depth >= THREAD_MAX_BBS)
    return;
  for (int i = 0; i < depth; ++i) {
    if (th->steps[i].bb == bb)
      return;
  }
  Vector *irs = bb->irs;
  IR *last = irs->len > 0 ? irs->data[irs->len - 1] : NULL;
  bool term = last != NULL && (last->kind == IR_JMP || last->kind == IR_TJMP);
  int n = irs->len - (term ? 1 : 0);
  if (th->budget < n)
    return;
  int mark = th->undo.len, budget = th->budget;
  for (int j = 0; j < n; ++j) {
    IR *ir = irs->data[j];
    if (!thread_copyable(ir)) {
      thread_undo(th, mark);
      return;
    }
    thread_track(th, ir);
  }
  th->budget -= n;
  th->steps[depth].bb = bb;
  BB *dest;
  if (!term || (last->kind == IR_JMP && last->jmp.cond == COND_ANY)) {
    th->steps[depth].how = 0;
    thread_walk(th, term ? last->jmp.bb : bb->next, depth + 1);
  } else if ((dest = thread_decide(th, bb, last)) != NULL) {
    th->steps[depth].how = 1;
    if (th->best <= depth) {
      th->best = depth + 1;
      th->best_dest = dest;
      memcpy(th->best_steps, th->steps, sizeof(*th->steps) * (depth + 1));
    }
    thread_walk(th, dest, depth + 1);
  } else if (last->kind == IR_JMP && th->budget >= 1) {
    // Not decided: copied; on to a decision along one way
    th->budget -= 1;
    int best = th->best;
    th->steps[depth].how = 2;
    thread_walk(th, bb->next, depth + 1);
    if (th->best == best && !((last->opr1->flag | last->opr2->flag) & VRF_FLONUM)) {
      th->steps[depth].how = 3;
      thread_walk(th, last->jmp.bb, depth + 1);
    }
  }
  th->budget = budget;
  thread_undo(th, mark);
}

static IR *copy_ir(IR *ir) {
  IR *c = malloc_or_die(sizeof(*c));
  *c = *ir;
  if (ir->additional_operands != NULL) {
    c->additional_operands = new_vector();
    vec_concat(c->additional_operands, ir->additional_operands);
  }
  return c;
}

// A temporary defined and used only in its block: its copy gets a register
// of its own (a temporary stays defined once: CSE, coalescing)
static void rename_local_temps(RegAlloc *ra, Vector *irs, int from, int *uses, int *defs,
                               int nvregs) {
  for (int j = from; j < irs->len; ++j) {
    IR *ir = irs->data[j];
    VReg *t = ir->dst;
    if (t == NULL || !plain_int_vreg(t) || (t->flag & (VRF_PARAM | VRF_STACK_PARAM | VRF_NO_SPILL)) ||
        t->virt >= nvregs || defs[t->virt] != 1)
      continue;
    int n = 0;
    for (int k = j + 1; k < irs->len; ++k) {
      IR *u = irs->data[k];
      n += (u->opr1 == t) + (u->opr2 == t);
      if (u->additional_operands != NULL) {
        for (int m = 0; m < u->additional_operands->len; ++m)
          n += u->additional_operands->data[m] == t;
      }
    }
    if (n != uses[t->virt])
      continue;
    VReg *nt = reg_alloc_spawn(ra, t->vsize, t->flag & (VRF_MASK | VRF_UNALIGNED));
    ir->dst = nt;
    for (int k = j + 1; k < irs->len; ++k) {
      IR *u = irs->data[k];
      if (u->opr1 == t)
        u->opr1 = nt;
      if (u->opr2 == t)
        u->opr2 = nt;
      if (u->additional_operands != NULL) {
        for (int m = 0; m < u->additional_operands->len; ++m) {
          if (u->additional_operands->data[m] == t)
            u->additional_operands->data[m] = nt;
        }
      }
    }
  }
}

static int bb_index_of(BBContainer *bbcon, BB *bb);

// Constants known at each block's entry: a forward dataflow over the
// function (the intersection over its predecessors of what they know at
// their ends, with what their jumps to it say).  Maps: <VReg*, VReg* const>.
typedef struct {
  Vector *in;   // NULL: not reached yet (everything)
} KnownIn;

static VReg *kmap_get(Vector *m, VReg *v) {
  for (int i = 0; i < m->len; i += 2) {
    if (m->data[i] == v)
      return m->data[i + 1];
  }
  return NULL;
}

static void kmap_set(Vector *m, VReg *v, VReg *c) {
  for (int i = 0; i < m->len; i += 2) {
    if (m->data[i] == v) {
      if (c == NULL) {
        vec_remove_at(m, i);
        vec_remove_at(m, i);
      } else {
        m->data[i + 1] = c;
      }
      return;
    }
  }
  if (c != NULL) {
    vec_push(m, v);
    vec_push(m, c);
  }
}

static bool kconst_equal(VReg *a, VReg *b) {
  return a->fixnum == b->fixnum;
}

// The map at a block's end, from its entry's
static void kmap_track(RegAlloc *ra, Vector *m, BB *bb) {
  for (int j = 0; j < bb->irs->len; ++j) {
    IR *ir = bb->irs->data[j];
    VReg *dst = ir->dst;
    if (dst == NULL || (dst->flag & VRF_CONST))
      continue;
    VReg *c = NULL;
    if ((ir->kind == IR_MOV || ir->kind == IR_CAST) && plain_int_vreg(dst) &&
        dst->vsize <= VRegSize4 && ir->opr1->vsize <= VRegSize4 && !(ir->opr1->flag & VRF_FLONUM)) {
      VReg *s = ir->opr1->flag & VRF_CONST ? ir->opr1 : kmap_get(m, ir->opr1);
      if (s != NULL) {
        int64_t v = wrap_value(s->fixnum, 1 << dst->vsize, (dst->flag & VRF_UNSIGNED) != 0);
        c = v == s->fixnum && s->vsize == dst->vsize &&
            (s->flag & VRF_UNSIGNED) == (dst->flag & VRF_UNSIGNED)
                ? s : reg_alloc_spawn_const(ra, v, dst->vsize, dst->flag & VRF_UNSIGNED);
      }
    }
    kmap_set(m, dst, c);
  }
}

// What the edge from q to bb adds: a table jump's index, x == c
static void kmap_edge(RegAlloc *ra, Vector *m, BB *q, BB *bb) {
  Vector *qirs = q->irs;
  if (qirs->len == 0)
    return;
  IR *last = qirs->data[qirs->len - 1];
  VReg *v;
  int64_t c;
  if (last->kind == IR_TJMP) {
    int idx = -1, n = 0;
    for (size_t k = 0; k < last->tjmp.len; ++k) {
      if (last->tjmp.bbs[k] == bb) {
        idx = (int)k;
        ++n;
      }
    }
    if (n != 1)
      return;
    v = last->opr1;
    c = idx;
  } else if (last->kind == IR_JMP && (last->jmp.cond == COND_EQ || last->jmp.cond == COND_NE) &&
             last->opr2 != NULL && (last->opr2->flag & VRF_CONST) &&
             !(last->opr2->flag & VRF_FLONUM) && !(last->opr1->flag & VRF_CONST)) {
    bool taken = last->jmp.bb == bb && q->next != bb;
    bool falls = q->next == bb && last->jmp.bb != bb;
    if (!(last->jmp.cond == COND_EQ ? taken : falls))
      return;
    v = last->opr1;
    c = last->opr2->fixnum;
  } else {
    return;
  }
  if (!plain_int_vreg(v) || v->vsize > VRegSize4)
    return;
  int64_t cv = wrap_value(c, 1 << v->vsize, (v->flag & VRF_UNSIGNED) != 0);
  VReg *k = reg_alloc_spawn_const(ra, cv, v->vsize, v->flag & VRF_UNSIGNED);
  kmap_set(m, v, k);
  // (and the register it was copied from before, in q or the blocks q is
  // the only way from: a switch's value copied before its range check)
  BB *chain[3] = {q, NULL, NULL};
  int nchain = 1;
  while (nchain < 3 && chain[nchain - 1]->from_bbs->len == 1)
    chain[nchain] = chain[nchain - 1]->from_bbs->data[0], ++nchain;
  for (int b = 0; b < nchain; ++b) {
    Vector *birs = chain[b]->irs;
    int j = birs->len - (b == 0 ? 2 : 1);
    for (; j >= 0; --j) {
      if (((IR*)birs->data[j])->dst == v)
        break;
    }
    if (j < 0)
      continue;
    IR *ir = birs->data[j];
    VReg *x = ir->opr1;
    if ((ir->kind == IR_MOV || ir->kind == IR_CAST) && plain_int_vreg(x) &&
        x->vsize == v->vsize && !defined_in(birs, j + 1, birs->len, x)) {
      bool kept = true;
      for (int d = 0; d < b && kept; ++d)
        kept = !defined_in(chain[d]->irs, 0, chain[d]->irs->len, x);
      if (kept) {  // (the same bits: the value as x's type has it)
        int64_t xv = wrap_value(cv, 1 << x->vsize, (x->flag & VRF_UNSIGNED) != 0);
        kmap_set(m, x, reg_alloc_spawn_const(ra, xv, x->vsize, x->flag & VRF_UNSIGNED));
      }
    }
    break;
  }
}

static Vector **known_in_maps(RegAlloc *ra, BBContainer *bbcon) {
  int n = bbcon->len;
  for (int i = 0; i < n; ++i)
    ((BB*)bbcon->data[i])->index = i;
  Vector **in = calloc_or_die(sizeof(*in) * (n + 1));
  Vector **out = calloc_or_die(sizeof(*out) * (n + 1));
  in[0] = new_vector();
  Vector tmp;
  vec_init(&tmp);
  for (int pass = 0; pass < 6; ++pass) {
    bool changed = false;
    for (int i = 0; i < n; ++i) {
      BB *bb = bbcon->data[i];
      if (i > 0) {
        // The meet over the predecessors reached
        Vector *meet = NULL;
        for (int f = 0; f < bb->from_bbs->len; ++f) {
          BB *q = bb->from_bbs->data[f];
          int qi = q->index;
          if (qi < 0 || qi >= n || bbcon->data[qi] != q || out[qi] == NULL)
            continue;
          vec_clear(&tmp);
          vec_concat(&tmp, out[qi]);
          kmap_edge(ra, &tmp, q, bb);
          if (meet == NULL) {
            meet = new_vector();
            vec_concat(meet, &tmp);
          } else {
            for (int k = 0; k < meet->len; k += 2) {
              VReg *o = kmap_get(&tmp, meet->data[k]);
              if (o == NULL || !kconst_equal(o, meet->data[k + 1])) {
                vec_remove_at(meet, k);
                vec_remove_at(meet, k);
                k -= 2;
              }
            }
          }
        }
        if (meet == NULL)
          continue;
        if (in[i] != NULL && in[i]->len == meet->len) {
          bool same = true;
          for (int k = 0; k < meet->len && same; k += 2) {
            VReg *o = kmap_get(in[i], meet->data[k]);
            same = o != NULL && kconst_equal(o, meet->data[k + 1]);
          }
          if (same) {
            free_vector(meet);
            goto track;
          }
        }
        if (in[i] != NULL)
          free_vector(in[i]);
        in[i] = meet;
        changed = true;
      }
    track:
      if (in[i] == NULL)
        continue;
      if (out[i] == NULL)
        out[i] = new_vector();
      vec_clear(out[i]);
      vec_concat(out[i], in[i]);
      kmap_track(ra, out[i], bb);
    }
    if (!changed && pass > 0)
      break;
  }
  vec_release(&tmp);
  for (int i = 0; i < n; ++i) {
    if (out[i] != NULL)
      free_vector(out[i]);
  }
  free(out);
  return in;
}

static void free_known_in_maps(Vector **in, int n) {
  for (int i = 0; i < n; ++i) {
    if (in[i] != NULL)
      free_vector(in[i]);
  }
  free(in);
}

// The constants known at the end of p: along its chain of only
// predecessors (a few), what the jumps between them say and what they do
static bool thread_known_at_end(Thread *th, BB *p, Vector *in) {
  if (in == NULL)
    return false;
  for (int k = 0; k < in->len; k += 2) {
    VReg *v = in->data[k];
    if (v->virt < th->known_len)
      thread_set(th, v, in->data[k + 1]);
  }
  for (int j = 0; j < p->irs->len; ++j)
    thread_track(th, p->irs->data[j]);
  for (int k = 0; k < th->known_len; ++k) {
    if (th->known[k] != NULL)
      return true;
  }
  return false;
}

// What the edge of p's conditional jump to `to` says: x == c
static void thread_edge_known(Thread *th, BB *p, IR *jmp, BB *to) {
  VReg *x = jmp->opr1, *c = jmp->opr2;
  if (c == NULL || !(c->flag & VRF_CONST) || (c->flag & VRF_FLONUM) || !plain_int_vreg(x) ||
      x->vsize > VRegSize4 || x->virt >= th->known_len)
    return;
  bool taken = to == jmp->jmp.bb;
  if ((jmp->jmp.cond == COND_EQ && taken) || (jmp->jmp.cond == COND_NE && !taken)) {
    int64_t v = wrap_value(c->fixnum, 1 << x->vsize, (x->flag & VRF_UNSIGNED) != 0);
    thread_set(th, x, reg_alloc_spawn_const(th->ra, v, x->vsize, x->flag & VRF_UNSIGNED));
  }
  (void)p;
}

static int bb_index_of(BBContainer *bbcon, BB *bb) {
  for (int k = 0; k < bbcon->len; ++k) {
    if (bbcon->data[k] == bb)
      return k;
  }
  return -1;
}

static bool thread_jumps(RegAlloc *ra, BBContainer *bbcon) {
  Thread th = {.ra = ra};
  vec_init(&th.undo);
  bool changed = false;
  int nvregs = ra->vregs->len;
  int *uses = calloc_or_die(sizeof(*uses) * nvregs * 2);
  int *defs = uses + nvregs;
  count_vreg_uses(bbcon, uses, defs);
  detect_from_bbs(bbcon);
  int total = 0;
  int count = bbcon->len;
  BB **orig = malloc_or_die(sizeof(*orig) * (count + 1));
  for (int i = 0; i < count; ++i)
    orig[i] = bbcon->data[i];
  int kin_len = bbcon->len;
  Vector **kin = known_in_maps(ra, bbcon);
  for (int i = 0; i < count && total < THREAD_FUNC_IRS; ++i) {
    BB *p = orig[i];
    for (int edge = 0; edge < 2 && total < THREAD_FUNC_IRS; ++edge) {
      Vector *irs = p->irs;
      IR *last = irs->len > 0 ? irs->data[irs->len - 1] : NULL;
      if (last != NULL && last->kind == IR_TJMP)
        break;
      // mode 0: p falls or jumps on (the copies go at its end); 1: the
      // jump's target edge, 2: its fall edge (the copies in a block of
      // their own on the edge)
      int mode;
      BB *succ;
      if (last == NULL || last->kind != IR_JMP || last->jmp.cond == COND_ANY) {
        if (edge > 0)
          break;
        mode = 0;
        succ = last != NULL && last->kind == IR_JMP ? last->jmp.bb : p->next;
      } else {
        if (last->jmp.cond == COND_NONE || last->opr1 == NULL ||
            ((last->opr1->flag | (last->opr2 != NULL ? last->opr2->flag : 0)) & VRF_FLONUM) ||
            last->jmp.bb == p->next)
          break;
        mode = edge == 0 ? 1 : 2;
        succ = mode == 1 ? last->jmp.bb : p->next;
      }
      if (succ == NULL || succ == p)
        continue;
      th.known = calloc_or_die(sizeof(*th.known) * (ra->vregs->len + 1));
      th.known_len = ra->vregs->len;
      int pi = bb_index_of(bbcon, p);
      bool any = pi >= 0 && thread_known_at_end(&th, p, kin[pi]);
      if (mode != 0) {
        thread_edge_known(&th, p, last, succ);
        for (int k = 0; k < th.known_len && !any; ++k)
          any = th.known[k] != NULL;
      }
      vec_clear(&th.undo);
      if (!any) {
        free(th.known);
        continue;
      }
      th.best = 0;
      th.best_dest = NULL;
      th.budget = THREAD_MAX_IRS;
      th.walking = true;
      thread_walk(&th, succ, 0);
      th.walking = false;
      if (th.best == 0 || th.best_dest == NULL) {
        free(th.known);
        continue;
      }
      // Where the copies go
      BB *cur, *after;
      int at;
      if (mode == 0) {
        if (last != NULL && last->kind == IR_JMP)
          vec_pop(irs);
        cur = p;
        after = p->next;
        at = bb_index_of(bbcon, p);
      } else {
        cur = new_bb();
        if (mode == 2) {  // on the fall edge: right after p
          at = bb_index_of(bbcon, p) + 1;
          after = p->next;
          cur->next = after;
          p->next = cur;
          vec_insert(bbcon, at, cur);
        } else {  // the target's: after the next block that doesn't fall through
          int k = bb_index_of(bbcon, p) + 1;
          for (; k < bbcon->len; ++k) {
            BB *b = bbcon->data[k - 1];
            IR *bl = b->irs->len > 0 ? b->irs->data[b->irs->len - 1] : NULL;
            if (bl != NULL && (bl->kind == IR_TJMP ||
                               (bl->kind == IR_JMP && bl->jmp.cond == COND_ANY)))
              break;
          }
          if (k >= bbcon->len) {
            free(th.known);
            continue;
          }
          at = k;
          BB *prev = bbcon->data[k - 1];
          after = prev->next;
          cur->next = after;
          prev->next = cur;
          vec_insert(bbcon, at, cur);
          last->jmp.bb = cur;
        }
      }
      for (int s = 0; s < th.best; ++s) {
        BB *bb = th.best_steps[s].bb;
        Vector *birs = bb->irs;
        IR *blast = birs->len > 0 ? birs->data[birs->len - 1] : NULL;
        bool term = blast != NULL && (blast->kind == IR_JMP || blast->kind == IR_TJMP);
        int n = birs->len - (term ? 1 : 0);
        int from = cur->irs->len;
        for (int j = 0; j < n; ++j) {
          vec_push(cur->irs, copy_ir(birs->data[j]));
          ++total;
        }
        int how = th.best_steps[s].how;
        IR *jcopy = NULL;
        if (how == 2 || how == 3) {  // (its test reads its temporaries too)
          jcopy = copy_ir(blast);
          vec_push(cur->irs, jcopy);
        }
        rename_local_temps(ra, cur->irs, from, uses, defs, nvregs);
        if (jcopy != NULL) {
          vec_pop(cur->irs);
          if (how == 3) {
            jcopy->jmp.cond = invert_cond(jcopy->jmp.cond);
            jcopy->jmp.bb = bb->next;
          }
          vec_push(cur->irs, jcopy);
          BB *nb = new_bb();
          nb->next = cur->next;
          cur->next = nb;
          vec_insert(bbcon, ++at, nb);
          cur = nb;
        }
      }
      vec_push(cur->irs, new_ir_jmp(th.best_dest));
      cur->next = after;
      free(th.known);
      changed = true;
      // The facts again, with the new blocks
      detect_from_bbs(bbcon);
      free_known_in_maps(kin, kin_len);
      kin_len = bbcon->len;
      kin = known_in_maps(ra, bbcon);
    }
  }
  free_known_in_maps(kin, kin_len);
  free(orig);
  free(uses);
  vec_release(&th.undo);
  return changed;
}

// A block jumped to only by an unconditional jump moves to after that jump
// (which goes; merge_blocks then joins them), if it can leave its place:
// it ends with a jump, or one to where it fell through is added.
static bool layout_jump_targets(BBContainer *bbcon) {
  bool changed = false;
  for (int i = 0; i < bbcon->len; ++i) {
    BB *p = bbcon->data[i];
    IR *jmp = is_last_jmp(p);
    if (jmp == NULL || jmp->jmp.cond != COND_ANY)
      continue;
    BB *d = jmp->jmp.bb;
    if (d == p || d == p->next || d->from_bbs->len != 1 || d->from_bbs->data[0] != p ||
        d == bbcon->data[0])
      continue;
    int k = -1;
    for (int m = 1; m < bbcon->len; ++m) {
      if (bbcon->data[m] == d) {
        k = m;
        break;
      }
    }
    if (k < 0)
      continue;
    Vector *dirs = d->irs;
    IR *dlast = dirs->len > 0 ? dirs->data[dirs->len - 1] : NULL;
    if (dlast != NULL && dlast->kind == IR_JMP && dlast->jmp.cond != COND_ANY)
      continue;  // Falls through after a test: stays
    if (dlast == NULL || (dlast->kind != IR_JMP && dlast->kind != IR_TJMP)) {
      if (d->next == NULL)
        continue;
      vec_push(dirs, new_ir_jmp(d->next));
    }
    BB *q = bbcon->data[k - 1];
    q->next = d->next;
    vec_remove_at(bbcon, k);
    if (k < i)
      --i;
    vec_insert(bbcon, i + 1, d);
    d->next = p->next;
    p->next = d;
    vec_pop(p->irs);
    changed = true;
  }
  return changed;
}

// (narrow)x & c is x & c when c fits the narrow type (registers keep values
// extended to 32 bits: ARM32)
static void drop_masked_casts(RegAlloc *ra, BBContainer *bbcon) {
#if XCC_TARGET_ARCH == XCC_ARCH_ARM32
  int vreg_count = ra->vregs->len;
  int *uses = calloc_or_die(sizeof(*uses) * vreg_count * 2);
  int *defs = uses + vreg_count;
  count_vreg_uses(bbcon, uses, defs);
  for (int i = 0; i < bbcon->len; ++i) {
    Vector *irs = ((BB*)bbcon->data[i])->irs;
    for (int j = 0; j < irs->len; ++j) {
      IR *ir = irs->data[j];
      if (ir->kind != IR_BITAND || ir->dst->vsize > VRegSize4 || (ir->dst->flag & VRF_FLONUM))
        continue;
      for (int side = 0; side < 2; ++side) {
        VReg *t = side == 0 ? ir->opr1 : ir->opr2, *c = side == 0 ? ir->opr2 : ir->opr1;
        if ((t->flag & (VRF_CONST | VRF_FLONUM)) || !(c->flag & VRF_CONST) || (c->flag & VRF_FLONUM) ||
            !plain_int_vreg(t) || uses[t->virt] != 1 || defs[t->virt] != 1)
          continue;
        int k = def_index(irs, j, t);
        if (k < 0)
          continue;
        IR *cast = irs->data[k];
        VReg *x = cast->opr1;
        if (cast->kind != IR_CAST || (x->flag & VRF_FLONUM) || x->vsize > VRegSize4 ||
            t->vsize > VRegSize4 || c->fixnum < 0 || c->fixnum >= ((int64_t)1 << (8 << t->vsize)) ||
            (!(x->flag & VRF_CONST) && (!plain_int_vreg(x) || defined_in(irs, k + 1, j, x))))
          continue;
        if (side == 0)
          ir->opr1 = x;
        else
          ir->opr2 = x;
        vec_remove_at(irs, k);
        --j;
        break;
      }
    }
  }
  free(uses);
#else
  (void)ra;
  (void)bbcon;
#endif
}

static void cond_flow(RegAlloc *ra, BBContainer *bbcon) {
  for (int pass = 0; pass < 3; ++pass) {
    bool changed = cond_diamonds(bbcon);
    if (changed) {
      detect_from_bbs(bbcon);
      clear_unused_bbs(bbcon);
      remove_unnecessary_bb(bbcon);
      detect_from_bbs(bbcon);
    }
    if (layout_jump_targets(bbcon)) {
      changed = true;
      detect_from_bbs(bbcon);
    }
    changed |= merge_blocks(bbcon);
    if (split_cond_jumps(ra, bbcon)) {
      changed = true;
      detect_from_bbs(bbcon);
    }
    if (fold_range_checks(ra, bbcon)) {
      changed = true;
      detect_from_bbs(bbcon);
      remove_unnecessary_bb(bbcon);
      detect_from_bbs(bbcon);
    }
    if (!changed)
      break;
  }
}

void optimize(RegAlloc *ra, BBContainer *bbcon) {
  clear_unused_bbs(bbcon);

  // Peephole
  for (int i = 0; i < bbcon->len; ++i) {
    BB *bb = bbcon->data[i];
    peephole(ra, bb);
  }

  if (apply_ssa) {
    make_ssa(ra, bbcon);
    copy_propagation(ra, bbcon);
    remove_unused_vregs(ra, bbcon);
    if (!keep_phi) {
      resolve_phis(ra, bbcon);

      detect_from_bbs(bbcon);
      clear_unused_bbs(bbcon);

      remove_unnecessary_bb(bbcon);
      coalesce_moves(ra, bbcon);  // FPC
    }
  } else {
    coalesce_moves(ra, bbcon);  // FPC
    drop_masked_casts(ra, bbcon);  // FPC
    cond_flow(ra, bbcon);  // FPC
    if (thread_jumps(ra, bbcon)) {  // FPC
      detect_from_bbs(bbcon);
      clear_unused_bbs(bbcon);
      remove_unnecessary_bb(bbcon);
      detect_from_bbs(bbcon);
      cond_flow(ra, bbcon);
    }
    if (fold_known_jumps(ra, bbcon)) {  // FPC
      detect_from_bbs(bbcon);
      clear_unused_bbs(bbcon);
    }
    local_cse(ra, bbcon);  // FPC
    coalesce_moves(ra, bbcon);  // FPC: what the passes above left
    remove_unused_vregs(ra, bbcon);
    remove_unnecessary_bb(bbcon);
    detect_from_bbs(bbcon);
    hoist_loop_invariants(ra, bbcon);  // FPC
    coalesce_moves(ra, bbcon);  // FPC: the loads made MOVs
  }
  detect_from_bbs(bbcon);
}
