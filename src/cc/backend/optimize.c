#include "../../config.h"
#include "optimize.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>  // free

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
  const int NG = VRF_CONST | VRF_PARAM | VRF_STACK_PARAM | VRF_FORCEMEMORY | VRF_VOLATILEREG |
                 VRF_NO_SPILL;
  return t != v && !(t->flag & NG) && !(v->flag & NG) && t->vsize == v->vsize &&
         (t->flag & (VRF_MASK | VRF_UNALIGNED)) == (v->flag & (VRF_MASK | VRF_UNALIGNED));
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
        if (t == NULL || next->kind != IR_MOV || next->opr1 != t)
          continue;
        switch (ir->kind) {
        case IR_ADD: case IR_SUB: case IR_MUL: case IR_DIV: case IR_MOD:
        case IR_BITAND: case IR_BITOR: case IR_BITXOR: case IR_LSHIFT: case IR_RSHIFT:
        case IR_COND: case IR_NEG: case IR_BITNOT: case IR_CAST: case IR_MOV:
        case IR_LOAD: case IR_BOFS: case IR_IOFS:
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
            !(ir->kind == IR_CAST && t->vsize == v->vsize &&
              !((t->flag | v->flag) & VRF_FLONUM)))
          continue;
        // Only the signedness differs (a cast): fine where it doesn't matter
        bool signonly = false;
        if (!coalescable(t, v)) {
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
              case IR_LOAD: case IR_STORE:
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
  default:
    return true;
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
    vec_clear(&avail);
    for (int j = 0; j < irs->len; ++j) {
      IR *ir = irs->data[j];
      if (any) {
        ir->opr1 = cse_subst(repl, ir->opr1);
        ir->opr2 = cse_subst(repl, ir->opr2);
      }
      VReg *dst = ir->dst;
      if (dst != NULL && cse_kind(ir) && !(dst->flag & NG) && defs[dst->virt] == 1 &&
          !((ir->opr1 != NULL && (ir->opr1->flag & VRF_FLONUM)) ||
            (ir->opr2 != NULL && (ir->opr2->flag & VRF_FLONUM)))) {
        int k;
        for (k = 0; k < avail.len; ++k) {
          if (same_expr(avail.data[k], ir))
            break;
        }
        if (k < avail.len) {
          repl[dst->virt] = ((IR*)avail.data[k])->dst;
          vec_remove_at(irs, j);
          --j;
          any = true;
          continue;
        }
        vec_push(&avail, ir);
        continue;
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

static void hoist_loop_invariants(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  int *uses = malloc_or_die(sizeof(*uses) * vreg_count * 2);
  int *defs = uses + vreg_count;
  for (int i = 0; i < vreg_count * 2; ++i)
    uses[i] = 0;
  count_vreg_uses(bbcon, uses, defs);
  unsigned char *inloop = malloc_or_die(vreg_count);
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
    if (!inner)
      continue;
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
        vec_remove_at(irs, m);
        --m;
        vec_insert(pre->irs, at++, ir);
        inloop[dst->virt] = false;
      }
    }
  }
  free(inloop);
  free(uses);
}

//

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
    if (fold_known_jumps(ra, bbcon)) {  // FPC
      detect_from_bbs(bbcon);
      clear_unused_bbs(bbcon);
    }
    local_cse(ra, bbcon);  // FPC
    remove_unused_vregs(ra, bbcon);
    remove_unnecessary_bb(bbcon);
    detect_from_bbs(bbcon);
    hoist_loop_invariants(ra, bbcon);  // FPC
  }
  detect_from_bbs(bbcon);
}
