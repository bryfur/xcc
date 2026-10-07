// FPC: wcc's functions through xcc's IR and its optimiser.
//
// wcc's own code generator writes WebAssembly straight from the syntax
// tree.  A function gen_wasm.c finds plain enough (scalar parameters,
// arguments and results, no builtins, no inline assembly, no alloca) goes
// instead through xcc's IR code generator and optimiser (src/cc/backend:
// jump threading, CSE, loop-invariant motion, strength reduction, ...),
// the native backend's, and then here into WebAssembly:
//   - the blocks get WebAssembly's structured control flow from their
//     dominator tree (Norman Ramsey, "Beyond Relooper", 2022): a loop
//     around each loop header, a block ending before each join; branches
//     to them are br/br_if/br_table, two-way branches to blocks only one
//     edge reaches are if/else; an irreducible graph is left to wcc;
//   - each virtual register is a local, each instruction's operands
//     computed in place where they were made just before and used once
//     (trees on the value stack); a constant part of an address is the
//     access's offset;
//   - address-taken scalars and aggregates live in a frame on the shadow
//     stack (__stack_pointer), as wcc's own code has them;
//   - wasm_opt.c then gives non-overlapping locals the same index.
// Narrow integers are kept extended to 32 bits in their locals, as the
// ARM32 backend keeps them in registers (the optimiser's assumption).

#include "../config.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../cc/backend/be_aux.h"
#include "../cc/backend/codegen.h"
#include "../cc/backend/ir.h"
#include "../cc/backend/optimize.h"
#include "../cc/backend/regalloc.h"
#include "ast.h"
#include "fe_misc.h"
#include "initializer.h"
#include "table.h"
#include "type.h"
#include "util.h"
#include "var.h"
#include "wasm.h"
#include "wcc_ir.h"

// The backend's target settings: arguments and parameters are all wasm
// values (no registers, no stack); the IR is not register-allocated.
const ArchSetting kArchSetting = {
  .max_reg_args = {1 << 16, 1 << 16},
};

const RegAllocSettings kArchRegAllocSettings = {
  .regset = {
    {.phys_max = 16, .phys_temporary_count = 16},
    {.phys_max = 16, .phys_temporary_count = 16},
  },
};

void tweak_irs(FuncBackend *fnbe) {
  UNUSED(fnbe);
}

int calculate_func_param_bottom(Function *func) {
  UNUSED(func);
  return 0;
}

//

enum { CX_LOOP, CX_BLOCK, CX_IF };

typedef struct {
  int kind;
  int node;  // block index (CX_LOOP: its header, CX_BLOCK: the block after its end)
} Ctx;

typedef struct {
  Function *func;
  FuncBackend *fnbe;
  DataStorage *code;
  Vector *relocs;    // the relocations of code
  BBContainer *bbcon;
  int nbb;
  int *rpo;        // [nbb] reverse postorder number (-1: unreachable)
  int *byrpo;      // [nreach] block by number
  int nreach;
  int *idom;       // [nbb]
  char *header;    // [nbb] a loop's header
  char *merge;     // [nbb] reached by two or more forward edges (or a br_table)
  int *kids;       // dominator tree: children of each block, by descending number
  int *kid_start;  // [nbb + 1]
  // locals
  int nvirt;
  int *local;      // [nvirt] its local (-1: none yet)
  int *uses, *defs;  // [nvirt]
  int *foldpos;      // [nvirt] a folded temporary's place in its block
  unsigned char *ltype;
  int nlocals, cap_locals;
  int nparams;
  int fp, rv;      // locals: the frame, the result (-1: none)
  unsigned char ret_wt;  // 0: void
  uint32_t frame_size;
  Vector *frames;  // <FrameInfo*> laid out in the frame
  // control
  Ctx *ctx;
  int nctx, cap_ctx;
  // the block being emitted
  char *folded;
  int cap_folded;
  unsigned char last_op;
  bool has_subsp;  // alloca: the stack pointer back from the frame pointer at the end
  bool tail_returned;  // the function's end: the result pushed last
  bool fail;
} WE;

static int fail_line;  // (where the last attempt gave up: WCC_IRLIST)
#define we_fail(we)  ((we)->fail = true, fail_line = fail_line == 0 ? __LINE__ : fail_line)

// Code

static void op(WE *we, unsigned char c) {
  data_push(we->code, c);
  we->last_op = c;
}

static void uleb(WE *we, uint64_t v) {
  data_uleb128(we->code, -1, v);
}

static void sleb(WE *we, int64_t v) {
  data_leb128(we->code, -1, v);
}

static void op_i32(WE *we, int32_t v) {
  op(we, OP_I32_CONST);
  sleb(we, v);
}

static void op_local(WE *we, unsigned char o, int l) {
  op(we, o);
  uleb(we, l);
}

// Types

static unsigned char vreg_wt(const VReg *v) {
  if (v->flag & VRF_FLONUM)
    return v->vsize == VRegSize4 ? WT_F32 : WT_F64;
  return v->vsize == VRegSize8 ? WT_I64 : WT_I32;
}

static unsigned char type_wt(const Type *t) {
  switch (t->kind) {
  case TY_FIXNUM: return type_size(t) <= 4 ? WT_I32 : WT_I64;
  case TY_FLONUM: return t->flonum.kind < FL_DOUBLE ? WT_F32 : WT_F64;
  case TY_PTR: case TY_ARRAY: case TY_FUNC: return WT_I32;
  default: return 0;
  }
}

static int new_local(WE *we, unsigned char wt) {
  if (we->nlocals >= we->cap_locals) {
    we->cap_locals = we->cap_locals * 2 + 16;
    we->ltype = realloc_or_die(we->ltype, we->cap_locals);
  }
  we->ltype[we->nlocals] = wt;
  return we->nlocals++;
}

static inline bool is_mem(const VReg *v) {
  return (v->flag & VRF_FORCEMEMORY) != 0;
}

static int vreg_local(WE *we, VReg *v) {
  assert(!(v->flag & VRF_CONST) && !is_mem(v));
  if (v->virt < 0 || v->virt >= we->nvirt) {
    we_fail(we);
    return 0;
  }
  int l = we->local[v->virt];
  if (l < 0)
    l = we->local[v->virt] = new_local(we, vreg_wt(v));
  return l;
}

// Value conversions between the wasm types of two vregs (i32 <-> i64)
static void convert(WE *we, unsigned char from, unsigned char to, bool from_unsigned) {
  if (from == to)
    return;
  if (from == WT_I32 && to == WT_I64)
    op(we, from_unsigned ? OP_I64_EXTEND_I32_U : OP_I64_EXTEND_I32_S);
  else if (from == WT_I64 && to == WT_I32)
    op(we, OP_I32_WRAP_I64);
  else
    we_fail(we);
}

static const unsigned char kLoadOps[2][4] = {
  // signed, unsigned: by size 1, 2, 4, 8 (integers)
  {OP_I32_LOAD8_S, OP_I32_LOAD16_S, OP_I32_LOAD, OP_I64_LOAD},
  {OP_I32_LOAD8_U, OP_I32_LOAD16_U, OP_I32_LOAD, OP_I64_LOAD},
};

static void memarg(WE *we, int size_pow, bool unaligned, int64_t offset) {
  uleb(we, unaligned ? 0 : size_pow);
  uleb(we, offset);
}

// A load of a value of v's kind from the address on the stack + offset.
static void load_of(WE *we, const VReg *v, int64_t offset, bool unaligned) {
  int s = v->vsize;
  if (v->flag & VRF_FLONUM)
    op(we, s == VRegSize4 ? OP_F32_LOAD : OP_F64_LOAD);
  else
    op(we, kLoadOps[(v->flag & VRF_UNSIGNED) != 0][s]);
  memarg(we, s, unaligned, offset);
}

static void store_of(WE *we, int vsize, bool flonum, int64_t offset, bool unaligned) {
  static const unsigned char kStoreOps[4] = {OP_I32_STORE8, OP_I32_STORE16, OP_I32_STORE,
                                             OP_I64_STORE};
  if (flonum)
    op(we, vsize == VRegSize4 ? OP_F32_STORE : OP_F64_STORE);
  else
    op(we, kStoreOps[vsize]);
  memarg(we, vsize, unaligned, offset);
}

static bool frame_known(WE *we, const FrameInfo *fi) {
  for (int i = 0; i < we->frames->len; ++i)
    if (we->frames->data[i] == fi)
      return true;
  return false;
}

static int64_t frame_offset(WE *we, const FrameInfo *fi) {
  if (!frame_known(we, fi) || we->fp < 0) {
    we_fail(we);
    return 0;
  }
  return fi->offset;
}

// A vreg's value pushed (as wasm type want: 0 its own)
static void push_vreg(WE *we, VReg *v, unsigned char want) {
  unsigned char wt = vreg_wt(v);
  if (want == 0)
    want = wt;
  if (v->flag & VRF_CONST) {
    if (v->flag & VRF_FLONUM) {
      if (want == WT_F32) {
        float f = v->flonum.value;
        op(we, OP_F32_CONST);
        data_append(we->code, &f, sizeof(f));
      } else if (want == WT_F64) {
        double d = v->flonum.value;
        op(we, OP_F64_CONST);
        data_append(we->code, &d, sizeof(d));
      } else {
        we_fail(we);
      }
    } else {
      int64_t x = v->fixnum;
      if (want == WT_I32) {
        op_i32(we, (int32_t)x);
      } else if (want == WT_I64) {
        // (a narrow constant: its value as its type has it)
        if (v->vsize < VRegSize8) {
          int bits = 8 << v->vsize;
          x = (v->flag & VRF_UNSIGNED) ? (int64_t)((uint64_t)x & (((uint64_t)1 << bits) - 1))
                                        : (int64_t)((uint64_t)x << (64 - bits)) >> (64 - bits);
        }
        op(we, OP_I64_CONST);
        sleb(we, x);
      } else {
        we_fail(we);
      }
    }
    return;
  }
  if (is_mem(v)) {
    op_local(we, OP_LOCAL_GET, we->fp >= 0 ? we->fp : 0);
    load_of(we, v, frame_offset(we, &v->frame), false);
  } else {
    op_local(we, OP_LOCAL_GET, vreg_local(we, v));
  }
  convert(we, wt, want, (v->flag & VRF_UNSIGNED) != 0);
}

// The operands of an IR in the order they are pushed (NULL: none)
static int ir_operands(IR *ir, VReg **ops, int max) {
  int n = 0;
#define ADD(v)  do { if (n < max) ops[n++] = (v); } while (0)
  switch (ir->kind) {
  case IR_LOAD: ADD(ir->opr1); break;
  case IR_STORE: ADD(ir->opr2); ADD(ir->opr1); break;
  case IR_ADD: case IR_SUB: case IR_MUL: case IR_DIV: case IR_MOD: case IR_BITAND:
  case IR_BITOR: case IR_BITXOR: case IR_LSHIFT: case IR_RSHIFT: case IR_COND:
    ADD(ir->opr1); ADD(ir->opr2); break;
  case IR_JMP:
    if (ir->jmp.cond != COND_ANY && ir->jmp.cond != COND_NONE) {
      ADD(ir->opr1); ADD(ir->opr2);
    }
    break;
  case IR_NEG: case IR_BITNOT: case IR_CAST: case IR_MOV: case IR_TJMP: case IR_SUBSP:
    ADD(ir->opr1); break;
  case IR_RESULT:
    if (ir->opr1 != NULL)
      ADD(ir->opr1);
    break;
  case IR_CALL:
    if (ir->additional_operands != NULL) {
      for (int i = 0; i < ir->additional_operands->len; ++i)
        ADD(ir->additional_operands->data[i]);
    }
    if (ir->opr1 != NULL)
      ADD(ir->opr1);
    break;
  default: break;
  }
#undef ADD
  return n;
}

#define MAX_OPS  64

// A temporary made once and used once (in the block it is made in): its
// value stays on the stack.
static bool foldable_temp(WE *we, const VReg *v) {
  return v != NULL && !(v->flag & (VRF_CONST | VRF_FORCEMEMORY | VRF_PARAM | VRF_VOLATILEREG)) &&
         v->virt >= 0 && v->virt < we->nvirt && we->uses[v->virt] == 1 &&
         we->defs[v->virt] == 1;
}

static bool foldable_kind(const IR *ir) {
  switch (ir->kind) {
  case IR_BOFS: case IR_IOFS: case IR_LOAD:
  case IR_ADD: case IR_SUB: case IR_MUL: case IR_DIV: case IR_MOD: case IR_BITAND:
  case IR_BITOR: case IR_BITXOR: case IR_LSHIFT: case IR_RSHIFT: case IR_COND:
  case IR_NEG: case IR_BITNOT: case IR_CAST: case IR_MOV: case IR_CALL:
    return ir->dst != NULL;
  default:
    return false;
  }
}

// Fold the operands of irs[j] made just before it (from cur back): their
// values stay on the stack.  Returns the first instruction before them.
static int fold_ops(WE *we, Vector *irs, int j, int cur) {
  VReg *ops[MAX_OPS];
  int n = ir_operands(irs->data[j], ops, MAX_OPS);
  for (int k = n; --k >= 0; ) {
    VReg *v = ops[k];
    if (v == NULL)
      continue;
    if (is_mem(v))
      break;  // (its load stays where it is: after what comes before)
    if (cur >= 0 && foldable_temp(we, v)) {
      IR *d = irs->data[cur];
      if (d->dst == v && foldable_kind(d)) {
        we->folded[cur] = 1;
        we->foldpos[v->virt] = cur;
        cur = fold_ops(we, irs, cur, cur - 1);
      }
    }
  }
  return cur;
}

static void emit_ir(WE *we, Vector *irs, int j);

// An operand of irs[j]: its folded tree or its value (as want)
// The place of v's folded definition before irs[j] in this block, or -1
static int folded_def(WE *we, Vector *irs, int j, VReg *v) {
  if (v == NULL || !foldable_temp(we, v))
    return -1;
  int k = we->foldpos[v->virt];
  return k >= 0 && k < j && we->folded[k] && ((IR*)irs->data[k])->dst == v ? k : -1;
}

static void emit_operand(WE *we, Vector *irs, int j, VReg *v, unsigned char want) {
  int k = folded_def(we, irs, j, v);
  if (k >= 0) {
    emit_ir(we, irs, k);
    convert(we, vreg_wt(v), want == 0 ? vreg_wt(v) : want, (v->flag & VRF_UNSIGNED) != 0);
    return;
  }
  push_vreg(we, v, want);
}

// An address operand: what is constant in it goes to the offset.
static int64_t emit_address(WE *we, Vector *irs, int j, VReg *v) {
  int64_t ofs = 0;
  int k = folded_def(we, irs, j, v);
  if (k >= 0) {
    {
      IR *d = irs->data[k];
      if (d->kind == IR_BOFS) {
        int64_t o = frame_offset(we, d->bofs.frameinfo) + d->bofs.offset;
        if (o >= 0) {
          op_local(we, OP_LOCAL_GET, we->fp >= 0 ? we->fp : 0);
          return o;
        }
      } else if (d->kind == IR_ADD && (d->opr2->flag & VRF_CONST) &&
                 !(d->opr2->flag & VRF_FLONUM) && d->dst->vsize == VRegSize4 &&
                 d->opr2->fixnum >= 0 && d->opr2->fixnum < 0x40000000 &&
                 !(d->opr1->flag & VRF_CONST)) {
        ofs = d->opr2->fixnum;
        ofs += emit_address(we, irs, k, d->opr1);
        return ofs;
      }
      emit_ir(we, irs, k);
      return 0;
    }
  }
  push_vreg(we, v, WT_I32);
  return 0;
}

static void emit_access_offset(WE *we, int64_t *pofs) {
  // (a negative offset: added to the address)
  if (*pofs < 0 || *pofs >= 0x80000000LL) {
    op_i32(we, (int32_t)*pofs);
    op(we, OP_I32_ADD);
    *pofs = 0;
  }
}

static const unsigned char kCmpOps[4][2][6] = {
  // EQ NE LT LE GE GT; [type][unsigned]
  {{OP_I32_EQ, OP_I32_NE, OP_I32_LT_S, OP_I32_LE_S, OP_I32_GE_S, OP_I32_GT_S},
   {OP_I32_EQ, OP_I32_NE, OP_I32_LT_U, OP_I32_LE_U, OP_I32_GE_U, OP_I32_GT_U}},
  {{OP_I64_EQ, OP_I64_NE, OP_I64_LT_S, OP_I64_LE_S, OP_I64_GE_S, OP_I64_GT_S},
   {OP_I64_EQ, OP_I64_NE, OP_I64_LT_U, OP_I64_LE_U, OP_I64_GE_U, OP_I64_GT_U}},
  {{OP_F32_EQ, OP_F32_NE, OP_F32_LT, OP_F32_LE, OP_F32_GE, OP_F32_GT},
   {OP_F32_EQ, OP_F32_NE, OP_F32_LT, OP_F32_LE, OP_F32_GE, OP_F32_GT}},
  {{OP_F64_EQ, OP_F64_NE, OP_F64_LT, OP_F64_LE, OP_F64_GE, OP_F64_GT},
   {OP_F64_EQ, OP_F64_NE, OP_F64_LT, OP_F64_LE, OP_F64_GE, OP_F64_GT}},
};

static int wt_index(unsigned char wt) {
  switch (wt) {
  case WT_I32: return 0;
  case WT_I64: return 1;
  case WT_F32: return 2;
  case WT_F64: return 3;
  default: return -1;
  }
}

// A comparison of opr1 and opr2 (cond) to 0/1 on the stack; invert: its
// opposite (a float's through eqz: NaN)
static void emit_compare(WE *we, Vector *irs, int j, VReg *opr1, VReg *opr2, int cond,
                         bool invert) {
  const VReg *t = (opr1->flag & VRF_CONST) ? opr2 : opr1;
  unsigned char wt = vreg_wt(t);
  if (!(opr1->flag & VRF_CONST) && !(opr2->flag & VRF_CONST) && vreg_wt(opr2) != wt) {
    // (mixed widths: the wider)
    if (vreg_wt(opr2) == WT_I64)
      wt = WT_I64;
  }
  bool flo = wt == WT_F32 || wt == WT_F64;
  if (invert && !flo) {
    cond = invert_cond(cond);
    invert = false;
  }
  bool uns = ((opr1->flag | opr2->flag) & VRF_UNSIGNED) != 0;
  if (!flo && (opr1->flag & VRF_CONST) == 0 && (opr2->flag & VRF_CONST) &&
      opr2->fixnum == 0 && (cond == COND_EQ || cond == COND_NE)) {
    // x == 0: eqz; x != 0: x itself where a condition is wanted
    emit_operand(we, irs, j, opr1, wt);
    if (cond == COND_EQ)
      op(we, wt == WT_I64 ? OP_I64_EQZ : OP_I32_EQZ);
    else if (wt == WT_I64)
      op(we, OP_I64_EQZ), op(we, OP_I32_EQZ);
    else
      return;  // (the caller: br_if/if on any nonzero; IR_COND: see there)
  } else {
    emit_operand(we, irs, j, opr1, wt);
    emit_operand(we, irs, j, opr2, wt);
    if (cond < COND_EQ || cond > COND_GT) {
      we_fail(we);
      return;
    }
    op(we, kCmpOps[wt_index(wt)][uns][cond - COND_EQ]);
  }
  if (invert)
    op(we, OP_I32_EQZ);
}

// The cast of src's value (on the stack) to dst's kind
static void emit_cast(WE *we, IR *ir) {
  VReg *dst = ir->dst, *src = ir->opr1;
  bool df = (dst->flag & VRF_FLONUM) != 0, sf = (src->flag & VRF_FLONUM) != 0;
  bool du = (dst->flag & VRF_UNSIGNED) != 0, su = (src->flag & VRF_UNSIGNED) != 0;
  int powd = dst->vsize, pows = src->vsize;
  if (!df && !sf) {
    if (ir->cast.fits && powd < VRegSize4 && pows <= VRegSize4)
      return;  // (the value fits already, or a store of its size takes it)
    if (powd == VRegSize8) {
      if (pows != VRegSize8)
        op(we, su ? OP_I64_EXTEND_I32_U : OP_I64_EXTEND_I32_S);
      return;
    }
    if (pows == VRegSize8)
      op(we, OP_I32_WRAP_I64);
    if (powd < VRegSize4 &&
        (powd < pows || (powd == pows && du != su) || (powd > pows && du && !su))) {
      if (du) {
        op_i32(we, powd == VRegSize1 ? 0xff : 0xffff);
        op(we, OP_I32_AND);
      } else {
        op(we, powd == VRegSize1 ? OP_I32_EXTEND8_S : OP_I32_EXTEND16_S);
      }
    }
    return;
  }
  if (df && sf) {
    if (powd != pows)
      op(we, powd == VRegSize8 ? OP_F64_PROMOTE_F32 : OP_F32_DEMOTE_F64);
    return;
  }
  if (df) {  // int -> floating point
    bool u = su && pows >= VRegSize4;
    if (pows == VRegSize8)
      op(we, powd == VRegSize4 ? (u ? OP_F32_CONVERT_I64_U : OP_F32_CONVERT_I64_S)
                               : (u ? OP_F64_CONVERT_I64_U : OP_F64_CONVERT_I64_S));
    else
      op(we, powd == VRegSize4 ? (u ? OP_F32_CONVERT_I32_U : OP_F32_CONVERT_I32_S)
                               : (u ? OP_F64_CONVERT_I32_U : OP_F64_CONVERT_I32_S));
    return;
  }
  // floating point -> int: saturating (out of range is undefined in C; the
  // trapping conversions cost the JIT range checks)
  int sub;
  if (powd == VRegSize8)
    sub = 4 + (pows == VRegSize4 ? 0 : 2) + du;
  else
    sub = (pows == VRegSize4 ? 0 : 2) + (du && powd == VRegSize4);
  op(we, OP_0xFC);
  uleb(we, sub);
}

// Arithmetic under 4 bytes is done in 32 bits: its result extended to its
// size (as the ARM32 backend's tweak_irs casts it; a counter that does not
// wrap, IR_ADD's bfx.nowrap, not)
static void narrow_result(WE *we, IR *ir) {
  VReg *dst = ir->dst;
  if (dst == NULL || (dst->flag & VRF_FLONUM) || dst->vsize >= VRegSize4)
    return;
  switch (ir->kind) {
  case IR_ADD:
    if (ir->bfx.nowrap)
      return;
    break;
  case IR_SUB: case IR_MUL: case IR_LSHIFT: case IR_NEG: case IR_BITNOT:
    break;
  default:
    return;
  }
  if (dst->flag & VRF_UNSIGNED) {
    op_i32(we, dst->vsize == VRegSize1 ? 0xff : 0xffff);
    op(we, OP_I32_AND);
  } else {
    op(we, dst->vsize == VRegSize1 ? OP_I32_EXTEND8_S : OP_I32_EXTEND16_S);
  }
}

static void emit_bop(WE *we, Vector *irs, int j) {
  IR *ir = irs->data[j];
  unsigned char wt = vreg_wt(ir->dst);
  int ti = wt_index(wt);
  // (as the ARM32 backend: division by the first operand's sign, a right
  // shift by the result's)
  bool u = ((ir->kind == IR_RSHIFT ? ir->dst : ir->opr1)->flag & VRF_UNSIGNED) != 0;
  static const unsigned char kOps[4][2][10] = {
    {{OP_I32_ADD, OP_I32_SUB, OP_I32_MUL, OP_I32_DIV_S, OP_I32_REM_S, OP_I32_AND, OP_I32_OR,
      OP_I32_XOR, OP_I32_SHL, OP_I32_SHR_S},
     {OP_I32_ADD, OP_I32_SUB, OP_I32_MUL, OP_I32_DIV_U, OP_I32_REM_U, OP_I32_AND, OP_I32_OR,
      OP_I32_XOR, OP_I32_SHL, OP_I32_SHR_U}},
    {{OP_I64_ADD, OP_I64_SUB, OP_I64_MUL, OP_I64_DIV_S, OP_I64_REM_S, OP_I64_AND, OP_I64_OR,
      OP_I64_XOR, OP_I64_SHL, OP_I64_SHR_S},
     {OP_I64_ADD, OP_I64_SUB, OP_I64_MUL, OP_I64_DIV_U, OP_I64_REM_U, OP_I64_AND, OP_I64_OR,
      OP_I64_XOR, OP_I64_SHL, OP_I64_SHR_U}},
    {{OP_F32_ADD, OP_F32_SUB, OP_F32_MUL, OP_F32_DIV, 0, 0, 0, 0, 0, 0},
     {OP_F32_ADD, OP_F32_SUB, OP_F32_MUL, OP_F32_DIV, 0, 0, 0, 0, 0, 0}},
    {{OP_F64_ADD, OP_F64_SUB, OP_F64_MUL, OP_F64_DIV, 0, 0, 0, 0, 0, 0},
     {OP_F64_ADD, OP_F64_SUB, OP_F64_MUL, OP_F64_DIV, 0, 0, 0, 0, 0, 0}},
  };
  unsigned char o = kOps[ti][u][ir->kind - IR_ADD];
  if (o == 0 || ir->bfx.sh_kind != 0 || (ir->kind == IR_BITAND && ir->bfx.shift != 0)) {
    we_fail(we);
    return;
  }
  emit_operand(we, irs, j, ir->opr1, wt);
  emit_operand(we, irs, j, ir->opr2, wt);
  op(we, o);
  narrow_result(we, ir);
}

static void emit_epilogue(WE *we) {
  if (we->fp >= 0) {
    op_local(we, OP_LOCAL_GET, we->fp);
    if (we->frame_size > 0) {
      op_i32(we, we->frame_size);
      op(we, OP_I32_ADD);
    }
    if (!wir_sp(we->relocs, we->code, true))
      we_fail(we);
  }
}

// irs[j] (and the trees folded into it)
static void emit_ir(WE *we, Vector *irs, int j) {
  IR *ir = irs->data[j];
  bool top = !we->folded[j];
  VReg *dst = ir->dst;
  bool mem_dst = top && dst != NULL && is_mem(dst);
  if (mem_dst)
    op_local(we, OP_LOCAL_GET, we->fp >= 0 ? we->fp : 0);

  switch (ir->kind) {
  case IR_BOFS:
    {
      int64_t o = frame_offset(we, ir->bofs.frameinfo) + ir->bofs.offset;
      op_local(we, OP_LOCAL_GET, we->fp >= 0 ? we->fp : 0);
      if (o != 0) {
        op_i32(we, (int32_t)o);
        op(we, OP_I32_ADD);
      }
    }
    break;
  case IR_IOFS:
    if (!wir_global_addr(we->relocs, we->code, ir->iofs.label)) {
      we_fail(we);
      break;
    }
    we->last_op = OP_I32_CONST;
    if (ir->iofs.offset != 0) {
      op_i32(we, (int32_t)ir->iofs.offset);
      op(we, OP_I32_ADD);
    }
    break;
  case IR_LOAD:
    {
      int64_t ofs = emit_address(we, irs, j, ir->opr1) + ir->load.offset;
      if (ir->load.index_shift != 0 && ir->opr2 != NULL) {
        we_fail(we);
        break;
      }
      emit_access_offset(we, &ofs);
      load_of(we, dst, ofs, ir->load.unaligned);
    }
    break;
  case IR_STORE:
    {
      int64_t ofs = emit_address(we, irs, j, ir->opr2) + ir->store.offset;
      if (ir->additional_operands != NULL) {
        we_fail(we);
        break;
      }
      emit_access_offset(we, &ofs);
      VReg *src = ir->opr1;
      unsigned char wt = (src->flag & VRF_FLONUM) ? vreg_wt(src)
                         : src->vsize == VRegSize8 ? WT_I64 : WT_I32;
      emit_operand(we, irs, j, src, wt);
      store_of(we, src->vsize, (src->flag & VRF_FLONUM) != 0, ofs, ir->store.unaligned);
    }
    break;
  case IR_ADD: case IR_SUB: case IR_MUL: case IR_DIV: case IR_MOD: case IR_BITAND:
  case IR_BITOR: case IR_BITXOR: case IR_LSHIFT: case IR_RSHIFT:
    if (ir->additional_operands != NULL) {
      we_fail(we);
      break;
    }
    emit_bop(we, irs, j);
    break;
  case IR_COND:
    {
      int cond = ir->cond.kind;
      if (cond == COND_ANY || cond == COND_NONE) {
        op_i32(we, cond == COND_ANY);
        break;
      }
      VReg *a = ir->opr1, *b = ir->opr2;
      bool ne0 = !(a->flag & (VRF_CONST | VRF_FLONUM)) && (b->flag & VRF_CONST) &&
                 b->fixnum == 0 && cond == COND_NE;
      emit_compare(we, irs, j, a, b, cond, false);
      if (ne0 && vreg_wt(a) == WT_I32) {  // (x != 0 as 0/1)
        op_i32(we, 0);
        op(we, OP_I32_NE);
      }
    }
    break;
  case IR_NEG:
    {
      unsigned char wt = vreg_wt(dst);
      if (ir->bfx.sbfx) {
        we_fail(we);
        break;
      }
      if (wt == WT_F32 || wt == WT_F64) {
        emit_operand(we, irs, j, ir->opr1, wt);
        op(we, wt == WT_F32 ? OP_F32_NEG : OP_F64_NEG);
      } else {
        if (wt == WT_I64) {
          op(we, OP_I64_CONST);
          sleb(we, 0);
        } else {
          op_i32(we, 0);
        }
        emit_operand(we, irs, j, ir->opr1, wt);
        op(we, wt == WT_I64 ? OP_I64_SUB : OP_I32_SUB);
        narrow_result(we, ir);
      }
    }
    break;
  case IR_BITNOT:
    {
      unsigned char wt = vreg_wt(dst);
      emit_operand(we, irs, j, ir->opr1, wt);
      if (wt == WT_I64) {
        op(we, OP_I64_CONST);
        sleb(we, -1);
        op(we, OP_I64_XOR);
      } else {
        op_i32(we, -1);
        op(we, OP_I32_XOR);
        narrow_result(we, ir);
      }
    }
    break;
  case IR_CAST:
    if (ir->cast.trunc != 0) {
      we_fail(we);
      break;
    }
    emit_operand(we, irs, j, ir->opr1, 0);
    emit_cast(we, ir);
    break;
  case IR_MOV:
    emit_operand(we, irs, j, ir->opr1, vreg_wt(dst));
    break;
  case IR_RESULT:
    if (ir->opr1 != NULL) {
      if (we->rv < 0 || ir->result.index != 0) {
        we_fail(we);
        break;
      }
      emit_operand(we, irs, j, ir->opr1, we->ret_wt);
      op_local(we, OP_LOCAL_SET, we->rv);
    }
    break;
  case IR_CALL:
    {
      IrCallInfo *call = ir->call;
      const Type *ft = call->functype;
      Vector *args = ir->additional_operands;
      int nargs = args != NULL ? args->len : 0;
      if (ft == NULL || ft->kind != TY_FUNC || ft->func.vaargs || ft->func.params == NULL ||
          ft->func.params->len != nargs || nargs != call->total_arg_count) {
        we_fail(we);
        break;
      }
      for (int i = 0; i < nargs; ++i) {
        const Type *pt = ft->func.params->data[i];
        unsigned char pw = type_wt(pt);
        if (pw == 0 || args->data[i] == NULL) {
          we_fail(we);
          break;
        }
        emit_operand(we, irs, j, args->data[i], pw);
      }
      if (we->fail)
        break;
      if (call->label != NULL) {
        if (!wir_call(we->relocs, we->code, call->label))
          we_fail(we);
      } else {
        emit_operand(we, irs, j, ir->opr1, WT_I32);
        if (!wir_call_indirect(we->relocs, we->code, ft))
          we_fail(we);
      }
      we->last_op = OP_CALL;
      const Type *rt = ft->func.ret;
      unsigned char rw = rt->kind == TY_VOID ? 0 : type_wt(rt);
      if (dst != NULL) {
        if (rw == 0) {
          we_fail(we);
          break;
        }
        convert(we, rw, vreg_wt(dst), is_unsigned(rt));
      } else if (rw != 0) {
        op(we, OP_DROP);
      }
    }
    break;
  case IR_KEEP:
    break;
  case IR_SUBSP:  // sp -= opr1; dst = sp
    if (we->fp < 0 || !wir_sp(we->relocs, we->code, false)) {
      we_fail(we);
      break;
    }
    emit_operand(we, irs, j, ir->opr1, WT_I32);
    op(we, OP_I32_SUB);
    if (dst != NULL) {
      op_local(we, OP_LOCAL_TEE, vreg_local(we, dst));
      dst = NULL;  // (set here)
    }
    if (!wir_sp(we->relocs, we->code, true))
      we_fail(we);
    break;
  default:
    we_fail(we);
    break;
  }

  if (top && dst != NULL && !we->fail && ir->kind != IR_RESULT) {
    if (mem_dst) {
      store_of(we, dst->vsize, (dst->flag & VRF_FLONUM) != 0, frame_offset(we, &dst->frame),
               false);
    } else {
      op_local(we, OP_LOCAL_SET, vreg_local(we, dst));
    }
  }
}

// Control flow

static BB *bb_at(WE *we, int i) {
  return we->bbcon->data[i];
}

static int bb_succs(WE *we, int i, int *out) {
  BB *bb = bb_at(we, i);
  int n = 0;
  IR *last = bb->irs->len > 0 ? bb->irs->data[bb->irs->len - 1] : NULL;
  if (last != NULL && last->kind == IR_TJMP) {
    for (size_t k = 0; k < last->tjmp.len; ++k)
      out[n++] = last->tjmp.bbs[k]->index;
    return n;
  }
  if (last != NULL && last->kind == IR_JMP && last->jmp.cond != COND_NONE) {
    out[n++] = last->jmp.bb->index;
    if (last->jmp.cond == COND_ANY)
      return n;
  }
  if (bb->next != NULL)
    out[n++] = bb->next->index;
  return n;
}

static int max_succs(WE *we) {
  int m = 2;
  for (int i = 0; i < we->nbb; ++i) {
    BB *bb = bb_at(we, i);
    IR *last = bb->irs->len > 0 ? bb->irs->data[bb->irs->len - 1] : NULL;
    if (last != NULL && last->kind == IR_TJMP && (int)last->tjmp.len > m)
      m = last->tjmp.len;
  }
  return m;
}

static int intersect(WE *we, int a, int b) {
  while (a != b) {
    while (we->rpo[a] > we->rpo[b])
      a = we->idom[a];
    while (we->rpo[b] > we->rpo[a])
      b = we->idom[b];
  }
  return a;
}

// Reverse postorder, dominators, loop headers, joins.  false: irreducible.
static bool analyze_cfg(WE *we) {
  int nbb = we->nbb;
  int ms = max_succs(we);
  int *succ = malloc_or_die(sizeof(int) * (ms + 1));
  we->rpo = malloc_or_die(sizeof(int) * nbb);
  we->byrpo = malloc_or_die(sizeof(int) * nbb);
  we->idom = malloc_or_die(sizeof(int) * nbb);
  we->header = calloc_or_die(nbb);
  we->merge = calloc_or_die(nbb);
  for (int i = 0; i < nbb; ++i) {
    we->rpo[i] = -1;
    we->idom[i] = -1;
  }

  // Postorder by an explicit stack: (block, next successor to visit)
  int *stk = malloc_or_die(sizeof(int) * nbb * 2);
  char *seen = calloc_or_die(nbb);
  int *post = malloc_or_die(sizeof(int) * nbb);
  int npost = 0, sp = 0;
  stk[sp++] = 0;
  stk[sp++] = 0;
  seen[0] = 1;
  while (sp > 0) {
    int b = stk[sp - 2], k = stk[sp - 1];
    int ns = bb_succs(we, b, succ);
    if (k < ns) {
      stk[sp - 1] = k + 1;
      int s = succ[k];
      if (!seen[s]) {
        seen[s] = 1;
        stk[sp++] = s;
        stk[sp++] = 0;
      }
    } else {
      post[npost++] = b;
      sp -= 2;
    }
  }
  we->nreach = npost;
  for (int i = 0; i < npost; ++i) {
    int b = post[npost - 1 - i];
    we->rpo[b] = i;
    we->byrpo[i] = b;
  }
  free(post);
  free(seen);
  free(stk);

  // Predecessors
  int *npred = calloc_or_die(sizeof(int) * (nbb + 1));
  for (int b = 0; b < nbb; ++b) {
    if (we->rpo[b] < 0)
      continue;
    int ns = bb_succs(we, b, succ);
    for (int k = 0; k < ns; ++k)
      npred[succ[k]]++;
  }
  int *pstart = malloc_or_die(sizeof(int) * (nbb + 1));
  pstart[0] = 0;
  for (int b = 0; b < nbb; ++b)
    pstart[b + 1] = pstart[b] + npred[b];
  int *preds = malloc_or_die(sizeof(int) * (pstart[nbb] + 1));
  memset(npred, 0, sizeof(int) * nbb);
  for (int b = 0; b < nbb; ++b) {
    if (we->rpo[b] < 0)
      continue;
    int ns = bb_succs(we, b, succ);
    for (int k = 0; k < ns; ++k) {
      int s = succ[k];
      preds[pstart[s] + npred[s]++] = b;
    }
  }

  // Dominators (Cooper, Harvey, Kennedy)
  we->idom[0] = 0;
  for (bool changed = true; changed; ) {
    changed = false;
    for (int i = 1; i < npost; ++i) {
      int b = we->byrpo[i];
      int nd = -1;
      for (int k = pstart[b]; k < pstart[b + 1]; ++k) {
        int p = preds[k];
        if (we->idom[p] < 0)
          continue;
        nd = nd < 0 ? p : intersect(we, p, nd);
      }
      if (nd >= 0 && we->idom[b] != nd) {
        we->idom[b] = nd;
        changed = true;
      }
    }
  }

  bool ok = true;
  for (int b = 0; b < nbb && ok; ++b) {
    if (we->rpo[b] < 0)
      continue;
    int nfwd = 0;
    for (int k = pstart[b]; k < pstart[b + 1]; ++k) {
      int p = preds[k];
      if (we->rpo[p] >= we->rpo[b]) {  // back edge: b must dominate p
        int d = p;
        while (d != b && d != 0)
          d = we->idom[d];
        if (d != b) {
          if (getenv("WCC_IRLIST") != NULL)
            fprintf(stderr, "irreducible: bb%d -> bb%d\n", p, b);
          ok = false;
          break;
        }
        we->header[b] = 1;
      } else {
        ++nfwd;
      }
    }
    if (nfwd >= 2)
      we->merge[b] = 1;
  }
  // A br_table's targets are blocks' ends.
  for (int b = 0; b < nbb && ok; ++b) {
    if (we->rpo[b] < 0)
      continue;
    BB *bb = bb_at(we, b);
    IR *last = bb->irs->len > 0 ? bb->irs->data[bb->irs->len - 1] : NULL;
    if (last != NULL && last->kind == IR_TJMP) {
      for (size_t k = 0; k < last->tjmp.len; ++k) {
        int t = last->tjmp.bbs[k]->index;
        if (we->rpo[t] > we->rpo[b])
          we->merge[t] = 1;
      }
    }
  }

  // Dominator tree children, by descending number
  if (ok) {
    int *nk = calloc_or_die(sizeof(int) * (nbb + 1));
    for (int i = 1; i < npost; ++i)
      nk[we->idom[we->byrpo[i]]]++;
    we->kid_start = malloc_or_die(sizeof(int) * (nbb + 1));
    we->kid_start[0] = 0;
    for (int b = 0; b < nbb; ++b)
      we->kid_start[b + 1] = we->kid_start[b] + nk[b];
    we->kids = malloc_or_die(sizeof(int) * (we->kid_start[nbb] + 1));
    memset(nk, 0, sizeof(int) * nbb);
    for (int i = npost; --i >= 1; ) {
      int b = we->byrpo[i];
      int d = we->idom[b];
      we->kids[we->kid_start[d] + nk[d]++] = b;
    }
    free(nk);
  }

  free(preds);
  free(pstart);
  free(npred);
  free(succ);
  return ok;
}

static void push_ctx(WE *we, int kind, int node) {
  if (we->nctx >= we->cap_ctx) {
    we->cap_ctx = we->cap_ctx * 2 + 16;
    we->ctx = realloc_or_die(we->ctx, sizeof(*we->ctx) * we->cap_ctx);
  }
  we->ctx[we->nctx].kind = kind;
  we->ctx[we->nctx].node = node;
  ++we->nctx;
}

static int label_depth(WE *we, int kind, int node) {
  for (int i = we->nctx; --i >= 0; )
    if (we->ctx[i].kind == kind && we->ctx[i].node == node)
      return we->nctx - 1 - i;
  we_fail(we);
  return 0;
}

// A block that only returns: nothing in it (the function's end)
static bool is_exit(WE *we, int b) {
  BB *bb = bb_at(we, b);
  int succ[2];
  if (bb->irs->len != 0)
    return false;
  return bb->next == NULL && bb_succs(we, b, succ) == 0;
}

static void emit_return(WE *we, bool tail) {
  emit_epilogue(we);
  if (we->rv >= 0)
    op_local(we, OP_LOCAL_GET, we->rv);
  if (!tail)
    op(we, OP_RETURN);
  else
    we->tail_returned = true;
}

static void do_tree(WE *we, int x, int follow);

static bool branchable(WE *we, int x, int t) {
  return we->rpo[t] <= we->rpo[x] || we->merge[t];
}

static void do_branch(WE *we, int x, int t, int follow) {
  if (we->fail)
    return;
  if (is_exit(we, t)) {
    emit_return(we, false);
    return;
  }
  if (we->rpo[t] <= we->rpo[x]) {  // back to a loop's header
    op(we, OP_BR);
    uleb(we, label_depth(we, CX_LOOP, t));
  } else if (we->merge[t]) {
    if (t == follow)
      return;  // (it comes next)
    op(we, OP_BR);
    uleb(we, label_depth(we, CX_BLOCK, t));
  } else {
    do_tree(we, t, follow);
  }
}

static void br_if_to(WE *we, int x, int t) {
  op(we, OP_BR_IF);
  uleb(we, we->rpo[t] <= we->rpo[x] ? label_depth(we, CX_LOOP, t) : label_depth(we, CX_BLOCK, t));
}

// Block x's instructions, then where it goes
static void emit_node(WE *we, int x, int follow) {
  BB *bb = bb_at(we, x);
  Vector *irs = bb->irs;
  int n = irs->len;
  if (n > we->cap_folded) {
    we->cap_folded = n * 2;
    we->folded = realloc_or_die(we->folded, we->cap_folded);
  }
  char *folded = we->folded;
  if (n > 0)
    memset(folded, 0, n);
  for (int j = n; --j >= 0; ) {
    if (!folded[j])
      fold_ops(we, irs, j, j - 1);
  }
  IR *last = n > 0 ? irs->data[n - 1] : NULL;
  int nbody = n;
  if (last != NULL && (last->kind == IR_JMP || last->kind == IR_TJMP))
    --nbody;
  // (the folded instructions are emitted in their consumers: keep the
  // array of this block's while doing so)
  for (int j = 0; j < nbody && !we->fail; ++j) {
    if (!we->folded[j])
      emit_ir(we, irs, j);
  }
  if (we->fail)
    return;

  int succ[2];
  if (last != NULL && last->kind == IR_TJMP) {
    emit_operand(we, irs, n - 1, last->opr1, WT_I32);
    size_t len = last->tjmp.len;
    op(we, OP_BR_TABLE);
    uleb(we, len - 1);
    for (size_t k = 0; k < len; ++k) {
      int t = last->tjmp.bbs[k]->index;
      uleb(we, we->rpo[t] <= we->rpo[x] ? label_depth(we, CX_LOOP, t)
                                        : label_depth(we, CX_BLOCK, t));
    }
    return;
  }
  if (last != NULL && last->kind == IR_JMP && last->jmp.cond != COND_ANY &&
      last->jmp.cond != COND_NONE) {
    int t = last->jmp.bb->index;
    int f = bb->next != NULL ? bb->next->index : -1;
    if (f < 0) {
      we_fail(we);
      return;
    }
    if (t == f) {
      do_branch(we, x, t, follow);
      return;
    }
    int cond = last->jmp.cond;
    if (branchable(we, x, t) && !is_exit(we, t)) {
      emit_compare(we, irs, n - 1, last->opr1, last->opr2, cond, false);
      br_if_to(we, x, t);
      do_branch(we, x, f, follow);
    } else if (branchable(we, x, f) && !is_exit(we, f)) {
      emit_compare(we, irs, n - 1, last->opr1, last->opr2, cond, true);
      br_if_to(we, x, f);
      do_branch(we, x, t, follow);
    } else {
      emit_compare(we, irs, n - 1, last->opr1, last->opr2, cond, false);
      op(we, OP_IF);
      data_push(we->code, WT_VOID);
      push_ctx(we, CX_IF, -1);
      do_branch(we, x, t, follow);
      op(we, OP_ELSE);
      do_branch(we, x, f, follow);
      --we->nctx;
      op(we, OP_END);
    }
    return;
  }
  int ns = bb_succs(we, x, succ);
  if (ns == 0) {
    emit_return(we, follow < 0 && we->nctx == 0);
    return;
  }
  do_branch(we, x, succ[0], follow);
}

static void do_tree(WE *we, int x, int follow) {
  for (;;) {
    if (we->fail)
      return;
    // x's children that are joins: blocks around x's code, ending each
    // before its join's code (the last join outermost)
    int ks = we->kid_start[x], ke = we->kid_start[x + 1];
    int ybuf[16];
    int *ys = ke - ks <= 16 ? ybuf : malloc_or_die(sizeof(int) * (ke - ks));
    int n = 0;
    for (int k = ks; k < ke; ++k) {
      int y = we->kids[k];
      if (we->merge[y])
        ys[n++] = y;  // (descending number: the outermost first)
    }
    bool loop = we->header[x] != 0;
    if (loop) {
      op(we, OP_LOOP);
      data_push(we->code, WT_VOID);
      push_ctx(we, CX_LOOP, x);
    }
    int base = we->nctx;
    for (int i = 0; i < n; ++i) {
      op(we, OP_BLOCK);
      data_push(we->code, WT_VOID);
      push_ctx(we, CX_BLOCK, ys[i]);
    }
    emit_node(we, x, n > 0 ? ys[n - 1] : follow);
    for (int i = n; --i >= 1 && !we->fail; ) {
      --we->nctx;
      op(we, OP_END);
      do_tree(we, ys[i], ys[i - 1]);
    }
    int y0 = n > 0 ? ys[0] : -1;
    if (ys != ybuf)
      free(ys);
    if (we->fail)
      return;
    if (n > 0) {
      --we->nctx;
      op(we, OP_END);
    }
    assert(we->nctx == base);
    UNUSED(base);
    if (loop) {
      if (n > 0)
        do_tree(we, y0, follow);  // (the first join's code inside the loop, then its end)
      if (we->fail)
        return;
      --we->nctx;
      op(we, OP_END);
      return;
    }
    if (n == 0)
      return;
    x = y0;  // (tail)
  }
}

// The frame: address-taken scalars and aggregates, from the frame pointer up
static void lay_out_frame(WE *we) {
  Function *func = we->func;
  uint32_t size = 0;
  for (int i = 0; i < func->scopes->len; ++i) {
    Scope *scope = func->scopes->data[i];
    for (int j = 0; j < scope->vars->len; ++j) {
      VarInfo *varinfo = scope->vars->data[j];
      if (!is_local_storage(varinfo))
        continue;
      FrameInfo *fi;
      size_t sz, align;
      if (is_prim_type(varinfo->type)) {
        VReg *v = varinfo->local.vreg;
        if (v == NULL || !is_mem(v))
          continue;
        fi = &v->frame;
        sz = align = (size_t)1 << v->vsize;
      } else {
        if (varinfo->storage & VS_PARAM) {
          we_fail(we);
          return;
        }
        fi = varinfo->local.frameinfo;
        if (fi == NULL)
          continue;
        const Type *type = varinfo->type;
        sz = type_size(type);
        if (type->kind == TY_STRUCT && (type->struct_.info->flag & SIF_FLEXIBLE)) {
          Initializer *init = varinfo->local.init;
          if (init != NULL) {
            int m = type->struct_.info->member_count;
            Initializer *e = init->multi->data[m - 1];
            if (e != NULL) {
              MemberInfo *member = &type->struct_.info->members[m - 1];
              sz += type_size(member->type->pa.ptrof) * e->multi->len;
            }
          }
        }
        if (sz < 1)
          sz = 1;
        align = align_size(type);
      }
      if (frame_known(we, fi))
        continue;
      size = ALIGN(size, align);
      fi->offset = size;
      size += sz;
      vec_push(we->frames, fi);
    }
  }
  // Other vregs in memory (a compound literal's)
  RegAlloc *ra = we->fnbe->ra;
  for (int i = 0; i < ra->vregs->len; ++i) {
    VReg *v = ra->vregs->data[i];
    if (v == NULL || !is_mem(v) || frame_known(we, &v->frame))
      continue;
    size_t sz = (size_t)1 << v->vsize;
    size = ALIGN(size, sz);
    v->frame.offset = size;
    size += sz;
    vec_push(we->frames, &v->frame);
  }
  we->frame_size = ALIGN(size, 16);
}

static void count_vregs(WE *we) {
  RegAlloc *ra = we->fnbe->ra;
  we->nvirt = ra->vregs->len;
  we->local = malloc_or_die(sizeof(int) * (we->nvirt + 1));
  we->uses = calloc_or_die(sizeof(int) * (we->nvirt + 1));
  we->foldpos = malloc_or_die(sizeof(int) * (we->nvirt + 1));
  for (int i = 0; i <= we->nvirt; ++i)
    we->foldpos[i] = -1;
  we->defs = calloc_or_die(sizeof(int) * (we->nvirt + 1));
  for (int i = 0; i < we->nvirt; ++i)
    we->local[i] = -1;
  VReg *ops[MAX_OPS];
  for (int b = 0; b < we->nbb; ++b) {
    if (we->rpo[b] < 0)
      continue;
    Vector *irs = bb_at(we, b)->irs;
    for (int j = 0; j < irs->len; ++j) {
      IR *ir = irs->data[j];
      int n = ir_operands(ir, ops, MAX_OPS);
      if (ir->kind == IR_CALL && n >= MAX_OPS)
        we_fail(we);
      if (ir->kind == IR_SUBSP)
        we->has_subsp = true;
      for (int k = 0; k < n; ++k) {
        VReg *v = ops[k];
        if (v != NULL && !(v->flag & VRF_CONST) && v->virt >= 0 && v->virt < we->nvirt)
          we->uses[v->virt]++;
      }
      if (ir->dst != NULL && !(ir->dst->flag & VRF_CONST) && ir->dst->virt >= 0 &&
          ir->dst->virt < we->nvirt)
        we->defs[ir->dst->virt]++;
      // (operands this emitter does not know: not folded)
      VReg *extra[] = {ir->opr1, ir->opr2};
      for (int k = 0; k < 2; ++k) {
        VReg *v = extra[k];
        if (v == NULL || (v->flag & VRF_CONST) || v->virt < 0 || v->virt >= we->nvirt)
          continue;
        bool listed = false;
        for (int m = 0; m < n; ++m)
          listed |= ops[m] == v;
        if (!listed)
          we->uses[v->virt] += 2;
      }
    }
  }
}

// Irreducible control flow (a loop entered at more than one block: jump
// threading's paths into a state machine's loop from before it) made
// reducible: the entries from outside go through a new block that jumps
// by a label local to the one wanted (br_table), the loop's only entry.

typedef struct {
  int n;
  int *sstart, *succ;  // successors of each block
  int *pstart, *pred;  // predecessors
  char *reach;
} Graph;

static void graph_free(Graph *g) {
  free(g->sstart); free(g->succ); free(g->pstart); free(g->pred); free(g->reach);
}

static void graph_build(WE *we, Graph *g) {
  int n = g->n = we->bbcon->len;
  for (int i = 0; i < n; ++i)
    bb_at(we, i)->index = i;
  we->nbb = n;
  int ms = max_succs(we);
  int *tmp = malloc_or_die(sizeof(int) * (ms + 1));
  g->reach = calloc_or_die(n + 1);
  int *stk = malloc_or_die(sizeof(int) * (n + 1));
  int sp = 0;
  stk[sp++] = 0;
  g->reach[0] = 1;
  while (sp > 0) {
    int b = stk[--sp];
    int ns = bb_succs(we, b, tmp);
    for (int k = 0; k < ns; ++k) {
      if (!g->reach[tmp[k]]) {
        g->reach[tmp[k]] = 1;
        stk[sp++] = tmp[k];
      }
    }
  }
  free(stk);
  int *cnt = calloc_or_die(sizeof(int) * (n + 1));
  g->sstart = malloc_or_die(sizeof(int) * (n + 1));
  g->sstart[0] = 0;
  for (int b = 0; b < n; ++b)
    g->sstart[b + 1] = g->sstart[b] + (g->reach[b] ? bb_succs(we, b, tmp) : 0);
  g->succ = malloc_or_die(sizeof(int) * (g->sstart[n] + 1));
  for (int b = 0; b < n; ++b) {
    if (!g->reach[b])
      continue;
    int ns = bb_succs(we, b, tmp);
    for (int k = 0; k < ns; ++k) {
      g->succ[g->sstart[b] + k] = tmp[k];
      cnt[tmp[k]]++;
    }
  }
  g->pstart = malloc_or_die(sizeof(int) * (n + 1));
  g->pstart[0] = 0;
  for (int b = 0; b < n; ++b)
    g->pstart[b + 1] = g->pstart[b] + cnt[b];
  g->pred = malloc_or_die(sizeof(int) * (g->pstart[n] + 1));
  memset(cnt, 0, sizeof(int) * n);
  for (int b = 0; b < n; ++b)
    for (int k = g->sstart[b]; k < g->sstart[b + 1]; ++k) {
      int t = g->succ[k];
      g->pred[g->pstart[t] + cnt[t]++] = b;
    }
  free(cnt);
  free(tmp);
}

// Strongly connected components of the blocks with in[b] set (edges
// between them only): comp[b] (-1: not in), returns the count.  Tarjan's,
// with an explicit stack.
static int sccs(const Graph *g, const char *in, int *comp) {
  int n = g->n;
  int *index = malloc_or_die(sizeof(int) * n * 5 + 4);
  int *low = index + n, *stk = low + n, *cs = stk + n, *ck = cs + n;
  char *on = calloc_or_die(n + 1);
  for (int i = 0; i < n; ++i) {
    index[i] = -1;
    comp[i] = -1;
  }
  int idx = 0, sp = 0, ncomp = 0;
  for (int r = 0; r < n; ++r) {
    if (!in[r] || index[r] >= 0)
      continue;
    int csp = 0;
    cs[csp] = r;
    ck[csp++] = g->sstart[r];
    index[r] = low[r] = idx++;
    stk[sp++] = r;
    on[r] = 1;
    while (csp > 0) {
      int v = cs[csp - 1];
      int k = ck[csp - 1];
      if (k < g->sstart[v + 1]) {
        ck[csp - 1] = k + 1;
        int w = g->succ[k];
        if (!in[w])
          continue;
        if (index[w] < 0) {
          index[w] = low[w] = idx++;
          stk[sp++] = w;
          on[w] = 1;
          cs[csp] = w;
          ck[csp++] = g->sstart[w];
        } else if (on[w] && index[w] < low[v]) {
          low[v] = index[w];
        }
      } else {
        --csp;
        if (csp > 0 && low[v] < low[cs[csp - 1]])
          low[cs[csp - 1]] = low[v];
        if (low[v] == index[v]) {
          int w;
          do {
            w = stk[--sp];
            on[w] = 0;
            comp[w] = ncomp;
          } while (w != v);
          ++ncomp;
        }
      }
    }
  }
  free(on);
  free(index);
  return ncomp;
}

// A loop with more than one entry: its blocks (members[b]) and entries,
// or false.
static bool find_multi_entry(const Graph *g, char *members, int *entries, int *nentries) {
  int n = g->n;
  // Sets still to look into, as lists of blocks (one set: all reachable)
  int *comp = malloc_or_die(sizeof(int) * (n + 1));
  char *in = calloc_or_die(n + 1);
  int *work = malloc_or_die(sizeof(int) * (n * 2 + 2));  // stack of (set id, ...)
  int *setof = malloc_or_die(sizeof(int) * (n + 1));     // block -> set id being processed
  bool found = false;
  // Each work item: a set of blocks marked with a set number in setof
  int nsets = 0;
  for (int b = 0; b < n; ++b)
    setof[b] = g->reach[b] ? 0 : -1;
  int wsp = 0;
  work[wsp++] = nsets++;
  while (wsp > 0 && !found) {
    int set = work[--wsp];
    for (int b = 0; b < n; ++b)
      in[b] = setof[b] == set;
    int nc = sccs(g, in, comp);
    for (int c = 0; c < nc && !found; ++c) {
      // its blocks; a single block only with an edge to itself
      int cnt = 0, any = -1;
      for (int b = 0; b < n; ++b)
        if (comp[b] == c)
          ++cnt, any = b;
      if (cnt == 1) {
        bool self = false;
        for (int k = g->sstart[any]; k < g->sstart[any + 1]; ++k)
          self |= g->succ[k] == any;
        if (!self)
          continue;
      }
      int ne = 0;
      for (int b = 0; b < n; ++b) {
        if (comp[b] != c)
          continue;
        bool entry = b == 0;
        for (int k = g->pstart[b]; k < g->pstart[b + 1] && !entry; ++k)
          entry = comp[g->pred[k]] != c;
        if (entry)
          entries[ne++] = b;
      }
      if (ne > 1) {
        for (int b = 0; b < n; ++b)
          members[b] = comp[b] == c;
        *nentries = ne;
        found = true;
      } else if (ne == 1 && cnt > 1) {
        // a proper loop: inside it, without its header
        int id = nsets++;
        for (int b = 0; b < n; ++b)
          if (comp[b] == c && b != entries[0])
            setof[b] = id;
        work[wsp++] = id;
      }
    }
  }
  free(setof);
  free(work);
  free(in);
  free(comp);
  return found;
}

static void redirect(WE *we, int u, int e, BB *to) {
  BB *bu = bb_at(we, u), *be = bb_at(we, e);
  IR *last = bu->irs->len > 0 ? bu->irs->data[bu->irs->len - 1] : NULL;
  if (last != NULL && last->kind == IR_TJMP) {
    for (size_t k = 0; k < last->tjmp.len; ++k)
      if (last->tjmp.bbs[k] == be)
        last->tjmp.bbs[k] = to;
    return;
  }
  bool jumps = last != NULL && last->kind == IR_JMP && last->jmp.cond != COND_NONE;
  if (jumps && last->jmp.bb == be)
    last->jmp.bb = to;
  if ((!jumps || last->jmp.cond != COND_ANY) && bu->next == be)
    bu->next = to;
}

static bool fix_irreducible(WE *we) {
  for (int iter = 0; iter < 32; ++iter) {
    Graph g;
    graph_build(we, &g);
    int n = g.n;
    char *members = calloc_or_die(n + 1);
    int *entries = malloc_or_die(sizeof(int) * (n + 1));
    int ne = 0;
    bool found = find_multi_entry(&g, members, entries, &ne);
    if (!found) {
      free(entries);
      free(members);
      graph_free(&g);
      return true;
    }
    if (getenv("WCC_IRLIST") != NULL) {
      fprintf(stderr, "multi-entry loop:");
      for (int b = 0; b < n; ++b)
        if (members[b])
          fprintf(stderr, " %d", b);
      fprintf(stderr, "; entries");
      for (int i = 0; i < ne; ++i) {
        fprintf(stderr, " %d(", entries[i]);
        for (int k = g.pstart[entries[i]]; k < g.pstart[entries[i] + 1]; ++k)
          fprintf(stderr, "%s%d", members[g.pred[k]] ? "" : "out:", g.pred[k]);
        fprintf(stderr, ")");
      }
      fprintf(stderr, "\n");
    }
    RegAlloc *ra = we->fnbe->ra;
    curra = ra;
    VReg *label = reg_alloc_spawn(ra, VRegSize4, VRF_UNSIGNED);
    BB *disp = new_bb();
    BB **targets = malloc_or_die(sizeof(BB*) * ne);
    for (int i = 0; i < ne; ++i)
      targets[i] = bb_at(we, entries[i]);
    curbb = disp;
    new_ir_tjmp(label, targets, ne);
    // Each entry's edges from outside: label = i, to the dispatch
    BB *first_entry = NULL;
    Vector *added = new_vector();
    for (int i = 0; i < ne; ++i) {
      int e = entries[i];
      BB *set = new_bb();
      curbb = set;
      new_ir_mov(label, new_const_vreg(i, VRegSize4, VRF_UNSIGNED));
      new_ir_jmp(disp);
      vec_push(added, set);
      for (int k = g.pstart[e]; k < g.pstart[e + 1]; ++k) {
        int u = g.pred[k];
        if (!members[u])
          redirect(we, u, e, set);
      }
      if (e == 0)
        first_entry = set;
    }
    curbb = NULL;
    curra = NULL;
    // (the new blocks after the others; the function's start the dispatch's
    // way in if it was an entry)
    vec_push(we->bbcon, disp);
    for (int i = 0; i < added->len; ++i)
      if (added->data[i] != first_entry)
        vec_push(we->bbcon, added->data[i]);
    if (first_entry != NULL)
      vec_insert(we->bbcon, 0, first_entry);
    free_vector(added);
    free(entries);
    free(members);
    graph_free(&g);
  }
  return false;
}

// A switch's range check just before its table jump: the table jump's
// default instead (br_table checks anyway)
static void fold_table_checks(WE *we) {
  BBContainer *bbcon = we->bbcon;
  for (int i = 0; i < bbcon->len; ++i) {
    BB *a = bbcon->data[i];
    IR *jmp = a->irs->len > 0 ? a->irs->data[a->irs->len - 1] : NULL;
    BB *b = a->next;
    if (jmp == NULL || jmp->kind != IR_JMP || jmp->jmp.cond != COND_GT || b == NULL ||
        b->irs->len != 1 || jmp->opr1 == NULL || !(jmp->opr1->flag & VRF_UNSIGNED) ||
        (jmp->opr1->flag & VRF_CONST) || jmp->opr2 == NULL || !(jmp->opr2->flag & VRF_CONST))
      continue;
    IR *tj = b->irs->data[0];
    if (tj->kind != IR_TJMP || tj->opr1 != jmp->opr1 ||
        (int64_t)tj->tjmp.len != jmp->opr2->fixnum + 1)
      continue;
    // b reached only from a (by its fall)
    bool only = true;
    for (int k = 0; k < bbcon->len && only; ++k) {
      BB *x = bbcon->data[k];
      if (x == a)
        continue;
      IR *l = x->irs->len > 0 ? x->irs->data[x->irs->len - 1] : NULL;
      if (x->next == b && !(l != NULL && ((l->kind == IR_JMP && l->jmp.cond == COND_ANY) ||
                                          l->kind == IR_TJMP)))
        only = false;
      if (l != NULL && l->kind == IR_JMP && l->jmp.bb == b)
        only = false;
      if (l != NULL && l->kind == IR_TJMP)
        for (size_t m = 0; m < l->tjmp.len; ++m)
          if (l->tjmp.bbs[m] == b)
            only = false;
    }
    if (!only)
      continue;
    size_t len = tj->tjmp.len;
    BB **bbs = malloc_or_die(sizeof(BB*) * (len + 1));
    memcpy(bbs, tj->tjmp.bbs, sizeof(BB*) * len);
    bbs[len] = jmp->jmp.bb;  // (the last: br_table's default)
    jmp->kind = IR_TJMP;
    jmp->opr2 = NULL;
    jmp->tjmp.bbs = bbs;
    jmp->tjmp.len = len + 1;
    vec_clear(b->irs);  // (b: not reached any more)
  }
}

static bool emit_function(WE *we) {
  Function *func = we->func;
  const Type *functype = func->type;
  const Vector *params = func->params;
  we->bbcon = we->fnbe->bbcon;
  we->nbb = we->bbcon->len;
  if (we->nbb == 0)
    return false;
  fold_table_checks(we);
  if (!fix_irreducible(we)) {
    if (getenv("WCC_IRLIST") != NULL)
      fprintf(stderr, "irreducible: not fixed\n");
    return false;
  }
  for (int i = 0; i < we->nbb; ++i)
    bb_at(we, i)->index = i;
  if (!analyze_cfg(we))
    return false;

  // Parameters: locals 0..
  we->nparams = params != NULL ? params->len : 0;
  for (int i = 0; i < we->nparams; ++i) {
    VarInfo *vi = params->data[i];
    unsigned char wt = type_wt(vi->type);
    if (wt == 0)
      return false;
    new_local(we, wt);
  }
  count_vregs(we);
  if (we->fail)
    return false;
  for (int i = 0; i < we->nparams; ++i) {
    VarInfo *vi = params->data[i];
    VReg *v = vi->local.vreg;
    if (v == NULL)
      return false;
    if (!is_mem(v)) {
      if (v->virt < 0 || v->virt >= we->nvirt || vreg_wt(v) != we->ltype[i])
        return false;
      we->local[v->virt] = i;
    }
  }
  const Type *rettype = functype->func.ret;
  we->ret_wt = rettype->kind == TY_VOID ? 0 : type_wt(rettype);
  if (rettype->kind != TY_VOID && we->ret_wt == 0)
    return false;
  if (we->ret_wt != 0)
    we->rv = new_local(we, we->ret_wt);

  lay_out_frame(we);
  if (we->fail)
    return false;
  if (we->frame_size > 0 || we->has_subsp) {
    we->fp = new_local(we, WT_I32);
    // fp = sp - size; sp = fp (alloca: sp back to fp + size at the end)
    if (!wir_sp(we->relocs, we->code, false))
      return false;
    if (we->frame_size > 0) {
      op_i32(we, we->frame_size);
      op(we, OP_I32_SUB);
      op_local(we, OP_LOCAL_TEE, we->fp);
      if (!wir_sp(we->relocs, we->code, true))
        return false;
    } else {
      op_local(we, OP_LOCAL_SET, we->fp);
    }
    // Address-taken parameters to their places in the frame
    for (int i = 0; i < we->nparams; ++i) {
      VarInfo *vi = params->data[i];
      VReg *v = vi->local.vreg;
      if (!is_mem(v))
        continue;
      op_local(we, OP_LOCAL_GET, we->fp);
      op_local(we, OP_LOCAL_GET, i);
      store_of(we, v->vsize, (v->flag & VRF_FLONUM) != 0, frame_offset(we, &v->frame), false);
    }
  } else {
    for (int i = 0; i < we->nparams; ++i) {
      VarInfo *vi = params->data[i];
      if (is_mem(vi->local.vreg))
        return false;
    }
  }

  we->last_op = OP_NOP;
  do_tree(we, 0, -1);
  if (we->fail || we->nctx != 0)
    return false;
  if (we->ret_wt != 0 && !(we->tail_returned && we->last_op == OP_LOCAL_GET) &&
      we->last_op != OP_RETURN && we->last_op != OP_BR && we->last_op != OP_BR_TABLE &&
      we->last_op != OP_UNREACHABLE)
    op(we, OP_UNREACHABLE);  // (the end not reached: validation still wants a value)
  op(we, OP_END);
  return true;
}

// WCC_IRDUMP=1: each function's IR on stderr (after the optimiser)
static void dump_vreg(const VReg *v) {
  if (v == NULL) {
    fprintf(stderr, "-");
  } else if (v->flag & VRF_CONST) {
    if (v->flag & VRF_FLONUM)
      fprintf(stderr, "(%g)", v->flonum.value);
    else
      fprintf(stderr, "(%lld)", (long long)v->fixnum);
  } else {
    fprintf(stderr, "v%d%s%s%s", v->virt, (const char*[]){".b", ".w", "", ".q"}[v->vsize],
            (v->flag & VRF_UNSIGNED) ? "u" : "", is_mem(v) ? "@" : "");
  }
}

static void dump_function(Function *func, BBContainer *bbcon) {
  static const char *kNames[] = {
    "BOFS", "IOFS", "SOFS", "LOAD", "LOAD_S", "STORE", "STORE_S", "ADD", "SUB", "MUL", "DIV",
    "MOD", "AND", "OR", "XOR", "SHL", "SHR", "COND", "NEG", "NOT", "CAST", "MOV", "RESULT",
    "JMP", "TJMP", "PUSHARG", "CALL", "SUBSP", "KEEP", "ASM",
  };
  fprintf(stderr, "%.*s:\n", NAMES(func->ident->ident));
  for (int i = 0; i < bbcon->len; ++i) {
    BB *bb = bbcon->data[i];
    fprintf(stderr, " bb%d (next %d):\n", i, bb->next != NULL ? bb->next->index : -1);
    for (int j = 0; j < bb->irs->len; ++j) {
      IR *ir = bb->irs->data[j];
      fprintf(stderr, "  %s ", kNames[ir->kind]);
      dump_vreg(ir->dst);
      fprintf(stderr, " = ");
      dump_vreg(ir->opr1);
      fprintf(stderr, ", ");
      dump_vreg(ir->opr2);
      if (ir->kind == IR_JMP)
        fprintf(stderr, " cond %d -> bb%d", ir->jmp.cond, ir->jmp.bb->index);
      if (ir->kind == IR_LOAD)
        fprintf(stderr, " +%lld", (long long)ir->load.offset);
      if (ir->kind == IR_STORE)
        fprintf(stderr, " +%lld", (long long)ir->store.offset);
      if (ir->kind == IR_COND)
        fprintf(stderr, " cond %d", ir->cond.kind);
      if (ir->kind == IR_CALL) {
        if (ir->call->label != NULL)
          fprintf(stderr, " %.*s(", NAMES(ir->call->label));
        else
          fprintf(stderr, " *(");
        for (int k = 0; ir->additional_operands != NULL && k < ir->additional_operands->len; ++k) {
          dump_vreg(ir->additional_operands->data[k]);
          fprintf(stderr, " ");
        }
        fprintf(stderr, ")");
      }
      fprintf(stderr, "\n");
    }
  }
}

#if defined(__NuttX__)  // (xcc_nuttx.c's arena)
#define ARENA_USED()  xcc_arena_used()
#define ARENA_MARK(m)  struct xcc_arena_mark_s m; xcc_arena_mark(&m)
#define ARENA_RELEASE(m)  xcc_arena_release(&m)
#else
#define ARENA_USED()  ((size_t)0)
#define ARENA_MARK(m)  do {} while (0)
#define ARENA_RELEASE(m)  do {} while (0)
#endif

// WCC_IRTIME=1: the IR path's phases' times, at the end
static double ir_time[5];
static double ir_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void print_ir_time(void) {
  fprintf(stderr, "wcc IR: gen %.1f ms, optimize %.1f ms, emit %.1f ms, wasm_opt %.1f ms\n",
          ir_time[0] * 1e3, ir_time[1] * 1e3, ir_time[2] * 1e3, ir_time[3] * 1e3);
}

static bool compile_ir(Function *func, DataStorage *code, Vector *relocs) {
  static int timing = -1;
  if (timing < 0) {
    timing = getenv("WCC_IRTIME") != NULL;
    if (timing)
      atexit(print_ir_time);
  }
  double t0 = timing ? ir_now() : 0, t1 = 0, t2 = 0, t3 = 0;
  void *wcc_extra = func->extra;
  bool ok = false;
  static int arena_stats = -1;
  if (arena_stats < 0)
    arena_stats = getenv("WCC_ARENA") != NULL;
  size_t a0 = arena_stats ? ARENA_USED() : 0, a1 = 0, a2 = 0;
  // The function's IR in a window of the arena: given back whole at the
  // end (what is kept: in a large block, copied out after)
  ARENA_MARK(mark);
  if (!gen_defun(func)) {
    func->extra = wcc_extra;
    ARENA_RELEASE(mark);
    return false;
  }
  FuncBackend *fnbe = func->extra;
  curfunc = func;
  curra = fnbe->ra;
  if (arena_stats)
    a1 = ARENA_USED();
  if (timing)
    t1 = ir_now();
  if (getenv("WCC_NOOPTIM") == NULL)
    optimize(fnbe->ra, fnbe->bbcon);
  else
    detect_from_bbs(fnbe->bbcon);
  analyze_reg_flow(fnbe->bbcon);
  for (int i = 0; i < 3 && remove_dead_defs(fnbe->ra, fnbe->bbcon); ++i)
    analyze_reg_flow(fnbe->bbcon);
  curra = NULL;
  curfunc = NULL;
  if (arena_stats)
    a2 = ARENA_USED();
  if (timing)
    t2 = ir_now();
  {
    static int dump = -1;
    if (dump < 0)
      dump = getenv("WCC_IRDUMP") != NULL;
    if (dump) {
      for (int i = 0; i < fnbe->bbcon->len; ++i)
        ((BB*)fnbe->bbcon->data[i])->index = i;
      dump_function(func, fnbe->bbcon);
    }
  }

  // Calls' backend information off the tree again (wcc's own use of it)
  if (fnbe->funcalls != NULL) {
    for (int i = 0; i < fnbe->funcalls->len; ++i) {
      Expr *e = fnbe->funcalls->data[i];
      e->funcall.fcinfo = NULL;
    }
  }

  func->extra = wcc_extra;
  WE we;
  memset(&we, 0, sizeof(we));
  we.func = func;
  we.fnbe = fnbe;
  we.fp = we.rv = -1;
  we.frames = new_vector();
  we.relocs = new_vector();

  // Body first (its locals known at the end), the declarations then
  DataStorage body;
  data_init(&body);
  we.code = &body;
  // (WCC_IRFAIL=1: every function through the IR code generator and the
  // optimiser, then wcc's own code generator after all: a test that the
  // first leaves the syntax tree as wcc's wants it)
  static int irfail = -1;
  if (irfail < 0) {
    const char *e = getenv("WCC_IRFAIL");
    irfail = e != NULL && *e == '1';
  }
  ok = !irfail && emit_function(&we);
  char *out = NULL;
  size_t ncode = 0, nrel = 0;
  uint32_t nparams = we.nparams, nlocals = we.nlocals;
  if (ok) {
    DataStorage head;
    data_init(&head);
    // locals beyond the parameters, in runs of one type
    uint32_t nruns = 0;
    for (int i = we.nparams; i < we.nlocals; ++i)
      if (i == we.nparams || we.ltype[i] != we.ltype[i - 1])
        ++nruns;
    data_uleb128(&head, -1, nruns);
    for (int i = we.nparams; i < we.nlocals; ) {
      int k = i;
      while (k < we.nlocals && we.ltype[k] == we.ltype[i])
        ++k;
      data_uleb128(&head, -1, k - i);
      data_push(&head, we.ltype[i]);
      i = k;
    }
    // [declarations][body][relocations (offsets past the declarations)][types]
    ncode = head.len + body.len;
    nrel = wir_relocs_size(we.relocs);
    size_t total = ncode + nrel + nlocals;
    out = malloc_or_die(total > 1024 ? total : 1024);  // (large: outlives the window)
    memcpy(out, head.buf, head.len);
    memcpy(out + head.len, body.buf, body.len);
    wir_relocs_save(we.relocs, out + ncode, head.len);
    memcpy(out + ncode + nrel, we.ltype, nlocals);
    data_release(&head);
  }
  data_release(&body);
  for (int i = 0; i < we.relocs->len; ++i)
    free(we.relocs->data[i]);
  free_vector(we.relocs);
  free(we.rpo); free(we.byrpo); free(we.idom); free(we.header); free(we.merge);
  free(we.kids); free(we.kid_start); free(we.local); free(we.uses); free(we.defs);
  free(we.foldpos); free(we.ltype); free(we.ctx); free(we.folded);
  free_vector(we.frames);
  if (arena_stats)
    fprintf(stderr, "arena %.*s: gen %zu, optimize %zu, emit %zu\n", NAMES(func->ident->ident),
            a1 - a0, a2 - a1, ARENA_USED() - a2);
  ARENA_RELEASE(mark);

  if (timing)
    t3 = ir_now();
  if (ok) {
    data_append(code, out, ncode);
    wir_relocs_restore(out + ncode, nrel, relocs);
    wasm_opt_func(code, relocs, (unsigned char*)out + ncode + nrel, nparams);
    free(out);
  }
  if (timing) {
    double t4 = ir_now();
    ir_time[0] += t1 - t0;
    ir_time[1] += t2 - t1;
    ir_time[2] += t3 - t2;
    ir_time[3] += t4 - t3;
  }
  return ok;
}

bool wir_gen_defun(Function *func, DataStorage *code, Vector *relocs) {
  static int disabled = -1;
  if (disabled < 0) {
    const char *e = getenv("WCC_NOIR");
    disabled = e != NULL && *e != '\0' && *e != '0';
  }
  if (disabled)
    return false;
  {  // (WCC_IRMAX=n: only the first n functions, WCC_IRSKIP=n: not the nth: to find a miscompile)
    static int count, max = -2, skip = -2;
    if (max == -2) {
      const char *e = getenv("WCC_IRMAX");
      max = e != NULL ? atoi(e) : -1;
      e = getenv("WCC_IRSKIP");
      skip = e != NULL ? atoi(e) : -1;
    }
    int n = count++;
    if ((max >= 0 && n >= max) || n == skip)
      return false;
  }
  fail_line = 0;
  bool ok = compile_ir(func, code, relocs);
  if (getenv("WCC_IRLIST") != NULL)
    fprintf(stderr, "ir %.*s: %s (%d)\n", NAMES(func->ident->ident), ok ? "ok" : "no", fail_line);
  return ok;
}
