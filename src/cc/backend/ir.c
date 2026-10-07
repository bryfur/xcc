#include "../../config.h"
#include "ir.h"

#include <assert.h>
#include <stdlib.h>  // malloc
#include <string.h>  // memcpy

#include "regalloc.h"
#include "table.h"
#include "util.h"

static const enum VRegSize vtVoidPtr = TARGET_POINTER_SIZE == 4 ? VRegSize4 : VRegSize8;  // FPC: was always 8
static const enum VRegSize vtBool = VRegSize1;

Phi *new_phi(VReg *dst, Vector *params) {
  Phi *phi = calloc_or_die(sizeof(*phi));
  phi->dst = dst;
  phi->params = params;
  return phi;
}

enum ConditionKind swap_cond(enum ConditionKind cond) {
  if (cond >= COND_LT)
    cond = (COND_GT + COND_LT) - cond;
  return cond;
}

enum ConditionKind invert_cond(enum ConditionKind cond) {
  int c = cond;
  assert(COND_EQ <= c && c <= COND_GT);
  int ic = c <= COND_NE ? (COND_NE + COND_EQ) - c
                        : (assert((COND_LT & 3) == 0), c ^ 2);  // COND_LT + ((c - COND_LT) ^ 2)
  return ic;
}

// Virtual register

void spill_vreg(VReg *vreg) {
  vreg->phys = -1;  // SPILLED_REG_NO(ra);
  assert(!(vreg->flag & VRF_NO_SPILL));
  vreg->flag |= VRF_SPILLED;
}

//
RegAlloc *curra;

// Intermediate Representation

static IR *new_ir(enum IrKind kind) {
  IR *ir = calloc_or_die(sizeof(*ir));
  ir->kind = kind;
  ir->dst = ir->opr1 = ir->opr2 = NULL;
  ir->additional_operands = NULL;
  if (curbb != NULL)
    vec_push(curbb->irs, ir);
  return ir;
}

VReg *new_const_vreg(int64_t value, enum VRegSize vsize, int vflag) {
  assert(curra != NULL);
  assert(!(vflag & VRF_FLONUM));
  return reg_alloc_spawn_const(curra, value, vsize, vflag);
}

#ifndef __NO_FLONUM
VReg *new_const_vfreg(double value, enum VRegSize vsize) {
  assert(curra != NULL);
  return reg_alloc_spawn_fconst(curra, value, vsize);
}
#endif

IR *new_ir_bop_raw(enum IrKind kind, VReg *dst, VReg *opr1, VReg *opr2) {
  IR *ir = new_ir(kind);
  ir->dst = dst;
  ir->opr1 = opr1;
  ir->opr2 = opr2;
  return ir;
}

VReg *new_ir_bop(enum IrKind kind, VReg *opr1, VReg *opr2, enum VRegSize vsize) {
  do {
    if ((opr2->flag & (VRF_FLONUM | VRF_CONST)) == VRF_CONST && opr2->fixnum == 0 &&
        (kind == IR_DIV || kind == IR_MOD)) {
      // error("Divide by 0");
      break;  // Skip constant folding (causing runtime error executable).
    }

    if ((opr1->flag & (VRF_FLONUM | VRF_CONST)) == VRF_CONST) {
      int64_t lval = opr1->fixnum;
      if ((opr2->flag & (VRF_FLONUM | VRF_CONST)) == VRF_CONST) {
        int64_t rval = opr2->fixnum;
        int64_t value = 0;
        switch (kind) {
        case IR_ADD:     value = lval + rval; break;
        case IR_SUB:     value = lval - rval; break;
        case IR_MUL:     value = lval * rval; break;
        case IR_DIV:
          if (opr1->flag & VRF_UNSIGNED)
            value = (uint64_t)lval / (uint64_t)rval;
          else
            value = lval / rval;
          break;
        case IR_MOD:
          if (opr1->flag & VRF_UNSIGNED)
            value = (uint64_t)lval % (uint64_t)rval;
          else
            value = lval % rval;
          break;
        case IR_BITAND:  value = lval & rval; break;
        case IR_BITOR:   value = lval | rval; break;
        case IR_BITXOR:  value = lval ^ rval; break;
        case IR_LSHIFT:  value = lval << rval; break;
        case IR_RSHIFT:
          // assert(opr1->type->kind == TY_FIXNUM);
          if (opr1->flag & VRF_UNSIGNED)
            value = (uint64_t)lval >> rval;
          else
            value = lval >> rval;
          break;
        default: assert(false); break;
        }
        return new_const_vreg(wrap_value(value, 1 << vsize, (opr1->flag & VRF_UNSIGNED) != 0),
                              vsize, opr1->flag & VRF_MASK);
      } else {
        switch (kind) {
        case IR_ADD:
          if (lval == 0)
            return opr2;  // no effect.
          break;
        case IR_SUB:
          if (lval == 0)
            return new_ir_unary(IR_NEG, opr2, opr2->vsize);
          break;
        case IR_MUL:
          switch (lval) {
          case 1:   return opr2;  // no effect.
          case 0:   return opr1;  // 0
          case -1:
            if (!(opr2->flag & VRF_UNSIGNED))
              return new_ir_unary(IR_NEG, opr2, opr2->vsize);  // -opr2
            break;
          default: break;
          }
          break;
        case IR_DIV:
        case IR_MOD:
          if (lval == 0)
            return opr1;  // TODO: whether opr2 is zero.
          break;
        case IR_BITAND:
          if (lval == 0)
            return opr1;  // 0
          break;
        case IR_BITOR:
        case IR_BITXOR:
          if (lval == 0)
            return opr2;  // no effect.
          break;
        case IR_LSHIFT:
        case IR_RSHIFT:
          if (lval == 0)
            return opr1;  // 0
          break;
        default:
          break;
        }
      }
    } else {
      if ((opr2->flag & (VRF_FLONUM | VRF_CONST)) == VRF_CONST) {
        int64_t rval = opr2->fixnum;
        switch (kind) {
        case IR_ADD:
        case IR_SUB:
          if (rval == 0)
            return opr1;  // no effect.
          break;
        case IR_DIV:
          if (rval == 0)
            return opr1;  // Detect zero division.
          // Fallthrough
        case IR_MUL:
          switch (rval) {
          case 1:   return opr1;  // no effect.
          case 0:   return opr2;  // 0
          case -1:
            if (!(opr1->flag & VRF_UNSIGNED))
              return new_ir_unary(IR_NEG, opr1, opr1->vsize);  // -opr1
            break;
          default: break;
          }
          break;
        case IR_BITAND:
          if (rval == 0)
            return opr2;  // 0
          break;
        case IR_BITOR:
        case IR_BITXOR:
          if (rval == 0)
            return opr1;  // no effect.
          break;
        case IR_LSHIFT:
        case IR_RSHIFT:
          if (opr2->fixnum == 0)
            return opr1;  // no effect.
          break;
        default:
          break;
        }
      }
    }
  } while (0);

  VReg *dst = reg_alloc_spawn(curra, vsize, opr1->flag & VRF_MASK);
  new_ir_bop_raw(kind, dst, opr1, opr2);
  return dst;
}

VReg *new_ir_unary(enum IrKind kind, VReg *opr, enum VRegSize vsize) {
  assert(kind != IR_LOAD);
  if (opr->flag & VRF_CONST) {
    int64_t value = 0;
    switch (kind) {
    case IR_NEG:     value = -opr->fixnum; break;
    case IR_BITNOT:  value = ~opr->fixnum; break;
    default: assert(false); break;
    }
    return new_const_vreg(wrap_value(value, 1 << vsize, (opr->flag & VRF_UNSIGNED) != 0), vsize,
                          opr->flag & VRF_MASK);
  }

  IR *ir = new_ir(kind);
  ir->opr1 = opr;
  return ir->dst = reg_alloc_spawn(curra, vsize, opr->flag & VRF_MASK);
}

IR *new_ir_load(VReg *opr, int64_t offset, enum VRegSize vsize, int vflag) {
  IR *ir = new_ir(IR_LOAD);
  ir->opr1 = opr;
  ir->load.offset = offset;
  ir->load.unaligned = (opr->flag & VRF_UNALIGNED) != 0;
  ir->dst = reg_alloc_spawn(curra, vsize, vflag);
  return ir;
}

IR *new_ir_bofs(FrameInfo *fi) {
  assert(fi != NULL);
  IR *ir = new_ir(IR_BOFS);
  ir->bofs.frameinfo = fi;
  ir->bofs.offset = 0;
  ir->dst = reg_alloc_spawn(curra, vtVoidPtr, VRF_UNSIGNED);
  return ir;
}

IR *new_ir_iofs(const Name *label, bool global) {
  IR *ir = new_ir(IR_IOFS);
  ir->iofs.label = label;
  ir->iofs.global = global;
  ir->iofs.offset = 0;
  ir->dst = reg_alloc_spawn(curra, vtVoidPtr, VRF_UNSIGNED);
  return ir;
}

IR *new_ir_sofs(VReg *offset) {
  IR *ir = new_ir(IR_SOFS);
  ir->opr1 = offset;
  ir->dst = reg_alloc_spawn(curra, vtVoidPtr, VRF_UNSIGNED);
  return ir;
}

IR *new_ir_store(VReg *dst, int64_t offset, VReg *src) {
  IR *ir = new_ir(IR_STORE);
  ir->opr1 = src;
  ir->opr2 = dst;  // `dst` is used by indirect, so it is not actually `dst`.
  ir->store.offset = offset;
  ir->store.unaligned = (dst->flag & VRF_UNALIGNED) != 0;
  return ir;
}

IR *new_ir_cond(VReg *opr1, VReg *opr2, enum ConditionKind cond) {
  IR *ir = new_ir(IR_COND);
  ir->opr1 = opr1;
  ir->opr2 = opr2;
  ir->cond.kind = cond;
  ir->dst = reg_alloc_spawn(curra, vtBool, 0);
  return ir;
}

IR *new_ir_jmp(BB *bb) {
  IR *ir = new_ir(IR_JMP);
  ir->jmp.bb = bb;
  ir->jmp.cond = COND_ANY;
  return ir;
}

void new_ir_cjmp(VReg *opr1, VReg *opr2, enum ConditionKind cond, BB *bb) {
  if (cond == COND_NONE)
    return;
  IR *ir = new_ir(IR_JMP);
  ir->opr1 = opr1;
  ir->opr2 = opr2;
  ir->jmp.bb = bb;
  ir->jmp.cond = cond;
}

void new_ir_tjmp(VReg *val, BB **bbs, size_t len) {
  assert(len >= 1);
  IR *ir = new_ir(IR_TJMP);
  ir->opr1 = val;
  ir->tjmp.bbs = bbs;
  ir->tjmp.len = len;
}

IR *new_ir_pusharg(VReg *vreg, int index) {
  assert(index >= 0);
  IR *ir = new_ir(IR_PUSHARG);
  ir->opr1 = vreg;
  ir->pusharg.index = index;
#if VAARG_FP_AS_GP || PUSHARG_FP_AS_GP
  ir->pusharg.fp_as_gp = false;
#endif
  return ir;
}

IR *new_ir_call(IrCallInfo *callinfo, VReg *dst, VReg *freg) {
  IR *ir = new_ir(IR_CALL);
  ir->call = callinfo;
  ir->dst = dst;
  ir->opr1 = freg;
  return ir;
}

void new_ir_result(VReg *vreg, int index) {
  IR *ir = new_ir(IR_RESULT);
  ir->opr1 = vreg;
  ir->result.index = index;
}

void new_ir_subsp(VReg *value, VReg *dst) {
  IR *ir = new_ir(IR_SUBSP);
  ir->opr1 = value;
  ir->dst = dst;
}

IR *new_ir_cast(VReg *vreg, enum VRegSize dstsize, int vflag) {
  IR *ir = new_ir(IR_CAST);
  ir->opr1 = vreg;
  ir->dst = reg_alloc_spawn(curra, dstsize, vflag);
  return ir;
}

IR *new_ir_mov(VReg *dst, VReg *src) {
  IR *ir = new_ir(IR_MOV);
  ir->dst = dst;
  ir->opr1 = src;
  return ir;
}

IR *new_ir_keep(VReg *dst, VReg *opr1, VReg *opr2) {
  IR *ir = new_ir(IR_KEEP);
  ir->dst = dst;
  ir->opr1 = opr1;
  ir->opr2 = opr2;
  return ir;
}

void new_ir_asm(Vector *templates, VReg *dst, Vector *registers) {
  IR *ir = new_ir(IR_ASM);
  ir->asm_.templates = templates;
  ir->additional_operands = registers;
  ir->dst = dst;
}

IR *new_ir_load_spilled(VReg *vreg, VReg *src) {
  IR *ir = new_ir(IR_LOAD_S);
  ir->dst = vreg;
  ir->opr1 = src;
  return ir;
}

IR *new_ir_store_spilled(VReg *dst, VReg *vreg) {
  IR *ir = new_ir(IR_STORE_S);
  ir->opr1 = vreg;
  ir->opr2 = dst;  // `dst` is used by indirect, so it is not actually `dst`.
  return ir;
}

// Basic Block

BB *curbb;

BB *new_bb(void) {
  BB *bb = calloc_or_die(sizeof(*bb));
  bb->next = NULL;
  bb->from_bbs = new_vector();
#if XCC_TARGET_ARCH == XCC_ARCH_WASM
  bb->label = alloc_label_uninterned();  // FPC: wcc frees a function's IR whole
#else
  bb->label = alloc_label();
#endif
  bb->irs = new_vector();
  bb->in_regs = new_vector();
  bb->out_regs = new_vector();
  bb->assigned_regs = new_vector();
  bb->phis = NULL;
  return bb;
}

BBContainer *new_func_blocks(void) {
  return new_vector();
}

//

void detect_from_bbs(BBContainer *bbcon) {
  int count = bbcon->len;
  if (count <= 0)
    return;

  // Clear all from_bbs
  for (int i = 0; i < count; ++i) {
    BB *bb = bbcon->data[i];
    vec_clear(bb->from_bbs);
    bb->index = i;  // FPC: the blocks seen marked by their places
  }

  char *checked = calloc_or_die(count + 1);
  Vector unchecked;
  vec_init(&unchecked);
  vec_push(&unchecked, bbcon->data[0]);

  do {
    BB *bb = vec_pop(&unchecked);
    int bi = bb->index;
    if (bi < 0 || bi >= count || bbcon->data[bi] != bb)
      continue;  // (not one of the function's blocks any more)
    if (checked[bi])
      continue;
    checked[bi] = 1;

    Vector *irs = bb->irs;
    if (irs->len > 0) {
      IR *ir = irs->data[irs->len - 1];  // JMP must be the last IR.
      switch (ir->kind) {
      case IR_JMP:
        {
          BB *dst = ir->jmp.bb;
          vec_push(dst->from_bbs, bb);
          vec_push(&unchecked, dst);
          if (ir->jmp.cond == COND_ANY)
            continue;  // Next BB is not reachable.
        }
        break;
      case IR_TJMP:
        for (size_t j = 0; j < ir->tjmp.len; ++j) {
          BB *nbb = ir->tjmp.bbs[j];
          vec_push(nbb->from_bbs, bb);
          vec_push(&unchecked, nbb);
        }
        continue;
      default: break;
      }
    }
    BB *next = bb->next;
    if (next != NULL) {
      vec_push(next->from_bbs, bb);
      vec_push(&unchecked, next);
    }
  } while (unchecked.len > 0);
  vec_release(&unchecked);
  free(checked);
}


// FPC: liveness with bit sets (the sorted vectors and their searches took
// a good part of cc1's time): in_regs live at the entry, out_regs at the
// end, assigned_regs defined in the block, each sorted by number
void analyze_reg_flow(BBContainer *bbcon) {
  int nb = bbcon->len;
  int nv = 0;
  for (int i = 0; i < nb; ++i) {
    BB *bb = bbcon->data[i];
    bb->index = i;
    if (bb->phis != NULL) {
      for (int j = 0; j < bb->phis->len; ++j) {
        Phi *phi = bb->phis->data[j];
        if (phi->dst->virt >= nv)
          nv = phi->dst->virt + 1;
        for (int k = 0; k < phi->params->len; ++k) {
          VReg *v = phi->params->data[k];
          if (!(v->flag & VRF_CONST) && v->virt >= nv)
            nv = v->virt + 1;
        }
      }
    }
    for (int j = 0; j < bb->irs->len; ++j) {
      IR *ir = bb->irs->data[j];
      VReg *vs[] = {ir->dst, ir->opr1, ir->opr2};
      for (int k = 0; k < 3; ++k) {
        if (vs[k] != NULL && !(vs[k]->flag & VRF_CONST) && vs[k]->virt >= nv)
          nv = vs[k]->virt + 1;
      }
      if (ir->additional_operands != NULL) {
        for (int k = 0; k < ir->additional_operands->len; ++k) {
          VReg *v = ir->additional_operands->data[k];
          if (v != NULL && !(v->flag & VRF_CONST) && v->virt >= nv)
            nv = v->virt + 1;
        }
      }
    }
  }
  int words = (nv + 31) / 32;
  if (words == 0)
    words = 1;
  VReg **byvirt = calloc_or_die(sizeof(*byvirt) * (nv + 1));
  uint32_t *use = calloc_or_die(sizeof(uint32_t) * words * nb * 4 + 4);
  uint32_t *def = use + words * nb, *in = def + words * nb, *out = in + words * nb;
#define BIT_SET(s, b, v)  ((s)[(b) * words + ((v) >> 5)] |= 1u << ((v) & 31))
#define BIT_TEST(s, b, v)  (((s)[(b) * words + ((v) >> 5)] >> ((v) & 31)) & 1)

  for (int i = 0; i < nb; ++i) {
    BB *bb = bbcon->data[i];
    if (bb->phis != NULL) {
      for (int j = 0; j < bb->phis->len; ++j) {
        Phi *phi = bb->phis->data[j];
        for (int k = 0; k < phi->params->len; ++k) {
          VReg *v = phi->params->data[k];
          if (v->flag & VRF_CONST)
            continue;
          byvirt[v->virt] = v;
          BIT_SET(use, i, v->virt);
        }
        byvirt[phi->dst->virt] = phi->dst;
        BIT_SET(def, i, phi->dst->virt);
      }
    }
    for (int j = 0; j < bb->irs->len; ++j) {
      IR *ir = bb->irs->data[j];
      VReg *vs[] = {ir->opr1, ir->opr2};
      int n = 2 + (ir->additional_operands != NULL ? ir->additional_operands->len : 0);
      for (int k = 0; k < n; ++k) {
        VReg *v = k < 2 ? vs[k] : ir->additional_operands->data[k - 2];
        if (v == NULL || (v->flag & VRF_CONST))
          continue;
        byvirt[v->virt] = v;
        if (!BIT_TEST(def, i, v->virt))
          BIT_SET(use, i, v->virt);
      }
      if (ir->dst != NULL && !(ir->dst->flag & VRF_CONST)) {
        byvirt[ir->dst->virt] = ir->dst;
        BIT_SET(def, i, ir->dst->virt);
      }
    }
  }

  // live_in = use | (live_out & ~def); live_out = the successors' live_in
  // (a block's predecessors get its live_in)
  memcpy(in, use, sizeof(uint32_t) * words * nb);
  for (bool changed = true; changed; ) {
    changed = false;
    for (int i = nb; --i >= 0; ) {
      BB *bb = bbcon->data[i];
      uint32_t *bin = in + i * words;
      for (int f = 0; f < bb->from_bbs->len; ++f) {
        BB *p = bb->from_bbs->data[f];
        int pi = p->index;
        if (pi < 0 || pi >= nb || bbcon->data[pi] != p)
          continue;
        uint32_t *pout = out + pi * words, *pin = in + pi * words, *pdef = def + pi * words;
        for (int w = 0; w < words; ++w) {
          uint32_t add = bin[w] & ~pout[w];
          if (add != 0) {
            pout[w] |= add;
            uint32_t nin = add & ~pdef[w] & ~pin[w];
            if (nin != 0) {
              pin[w] |= nin;
              changed = true;
            }
          }
        }
      }
    }
  }

  for (int i = 0; i < nb; ++i) {
    BB *bb = bbcon->data[i];
    vec_clear(bb->in_regs);
    vec_clear(bb->out_regs);
    vec_clear(bb->assigned_regs);
    for (int v = 0; v < nv; ++v) {
      if (byvirt[v] == NULL)
        continue;
      if (BIT_TEST(in, i, v))
        vec_push(bb->in_regs, byvirt[v]);
      if (BIT_TEST(out, i, v))
        vec_push(bb->out_regs, byvirt[v]);
      if (BIT_TEST(def, i, v))
        vec_push(bb->assigned_regs, byvirt[v]);
    }
  }
#undef BIT_SET
#undef BIT_TEST
  free(use);
  free(byvirt);
}
