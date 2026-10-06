// FPC: xcc's ARM32 target: IR to Thumb-2 assembly (unified syntax).
//
// Registers (arm32.h): r0-r3, r4-r6, r8-r11, r7 allocated; ip (r12) the
// scratch of every sequence here.  A 64-bit value (long long; a double in
// core registers) takes a pair of adjacent allocation indices (lo, hi); a
// double in the VFP takes s2n:s2n+1 (dn).  The FPU is single precision:
// double arithmetic, comparisons and conversions are calls to the run-time
// ABI's helpers (__aeabi_dadd...), as are 64-bit division and variable
// shifts; such an IR breaks r0-r3, ip, lr and s0-s15, which the register
// allocator keeps clear of anything live across it (detect_extra_occupied)
// and of its own operands.

#include "../../../config.h"
#include "ir.h"

#include <assert.h>
#include <inttypes.h>  // PRId64
#include <stdlib.h>
#include <string.h>

#include "arm32.h"
#include "ast.h"
#include "be_aux.h"
#include "regalloc.h"
#include "table.h"
#include "util.h"

const char *kReg32s[PHYSICAL_REG_MAX] = {
  "r0", "r1", "r2", "r3",                              // Temporary
  "r4", "r5", "r6", "r8", "r9", "r10", "r11", "r7",    // Callee save
};

// For ei_asm (emit_code.c): every size names the 32-bit register.
const char **kRegSizeTable[] = {kReg32s, kReg32s, kReg32s, kReg32s};

const char *kFReg32s[PHYSICAL_FREG_MAX] = {
  "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
  "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
  "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
  "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",
};

static const char *kDRegs[16] = {
  "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
  "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
};

const int ArchRegParamMapping[] = {0, 1, 2, 3};
#define ArchRegReturnMapping  ArchRegParamMapping

#define SZ_FLOAT   VRegSize4
#define SZ_DOUBLE  VRegSize8

const char *dreg(int phys) {
  assert(phys >= 0 && (phys & 1) == 0 && phys < 32);
  return kDRegs[phys >> 1];
}

// The low and high registers of a core register pair.
#define RLO(v)  kReg32s[(v)->phys]
#define RHI(v)  kReg32s[(v)->phys + 1]
#define R(v)    kReg32s[(v)->phys]
#define S(v)    kFReg32s[(v)->phys]
#define D(v)    dreg((v)->phys)

static inline bool is_i64(const VReg *v) {
  return !(v->flag & VRF_FLONUM) && v->vsize == VRegSize8;
}
static inline bool is_double(const VReg *v) {
  return (v->flag & VRF_FLONUM) && v->vsize == VRegSize8;
}
static inline bool is_float(const VReg *v) {
  return (v->flag & VRF_FLONUM) && v->vsize == VRegSize4;
}

// Does this IR call a run-time helper (breaking r0-r3, ip, lr, s0-s15)?
static bool ir_calls_helper(const IR *ir) {
  switch (ir->kind) {
  case IR_ADD: case IR_SUB: case IR_MUL:
    return is_double(ir->dst);
  case IR_DIV:
    return is_double(ir->dst) || is_i64(ir->dst);
  case IR_MOD:
    return is_i64(ir->dst);
  case IR_LSHIFT: case IR_RSHIFT:
    return is_i64(ir->dst) && !(ir->opr2->flag & VRF_CONST);
  case IR_COND:
    return is_double(ir->opr1);
  case IR_JMP:
    return ir->jmp.cond != COND_ANY && ir->opr1 != NULL && is_double(ir->opr1);
  case IR_CAST:
    {
      const VReg *d = ir->dst, *s = ir->opr1;
      bool df = (d->flag & VRF_FLONUM) != 0, sf = (s->flag & VRF_FLONUM) != 0;
      if (!df && !sf)
        return false;
      if (df && sf)
        return d->vsize != s->vsize;
      return d->vsize == VRegSize8 || s->vsize == VRegSize8;
    }
  default:
    return false;
  }
}

bool arm32_ir_calls_helper(const IR *ir) {
  return ir_calls_helper(ir);
}

static RegBits detect_extra_occupied(RegAlloc *ra, IR *ir) {
  RegBits ioccupy = 0;
  if (ir_calls_helper(ir))
    ioccupy |= REGBIT(0) | REGBIT(1) | REGBIT(2) | REGBIT(3);
  if (ra->flag & RAF_STACK_FRAME)
    ioccupy |= REGBIT(GET_FPREG_INDEX());
  return ioccupy;
}

static RegBits detect_extra_occupied_fp(RegAlloc *ra, IR *ir) {
  UNUSED(ra);
  return ir_calls_helper(ir) ? REGBIT(PHYSICAL_FREG_TEMPORARY) - 1 : 0;
}

const RegAllocSettings kArchRegAllocSettings = {
  .detect_extra_occupied = detect_extra_occupied,
  .detect_extra_occupied_fp = detect_extra_occupied_fp,
  .reg_param_mapping = ArchRegParamMapping,
  .reg_return_mapping = ArchRegReturnMapping,
  {
    {
      .phys_max = PHYSICAL_REG_MAX,
      .phys_temporary_count = PHYSICAL_REG_TEMPORARY,
      .pair_align = 1,
      .reserved = 0,
    },
#ifndef __NO_FLONUM
    {
      .phys_max = PHYSICAL_FREG_MAX,
      .phys_temporary_count = PHYSICAL_FREG_TEMPORARY,
      .pair_align = 2,
      .reserved = REGBIT(FREG_SCRATCH),
    },
#endif
  },
};

//

char *im(int64_t x) {
  return fmt("#%" PRId64, x);
}

// A Thumb-2 modified immediate (ThumbExpandImm)?
bool is_thumb_modimm(uint32_t x) {
  if (x <= 0xff)
    return true;
  uint32_t b = x & 0xff, b2 = (x >> 8) & 0xff;
  if (x == (b | (b << 16)))
    return true;
  if (x == ((b2 << 8) | (b2 << 24)))
    return true;
  if (x == (b | (b << 8) | (b << 16) | (b << 24)))
    return true;
  // An 8-bit value with its top bit set, rotated.
  for (int rot = 8; rot < 32; ++rot) {
    uint32_t v = (x << rot) | (x >> (32 - rot));
    if (v >= 0x80 && v <= 0xff)
      return true;
  }
  return false;
}

void mov_immediate(const char *dst, int64_t value) {
  uint32_t u = (uint32_t)value;
  if (is_thumb_modimm(u)) {
    EMIT_ASM("mov", dst, im(u));
  } else if (is_thumb_modimm(~u)) {
    EMIT_ASM("mvn", dst, im(~u));
  } else if (u <= 0xffff) {
    EMIT_ASM("movw", dst, im(u));
  } else {
    EMIT_ASM("movw", dst, im(u & 0xffff));
    EMIT_ASM("movt", dst, im(u >> 16));
  }
}

// A register holding the 32-bit operand: its own, or ip with the constant.
static const char *reg_or_ip(const VReg *v) {
  if (v->flag & VRF_CONST) {
    mov_immediate(IP, v->fixnum);
    return IP;
  }
  return R(v);
}

// Register pair move: (dlo, dhi) <- (slo, shi), the two moves in a safe order.
static void move_pair(const char *dlo, const char *dhi, const char *slo, const char *shi) {
  if (strcmp(dlo, shi) == 0) {
    if (strcmp(dhi, slo) == 0) {  // Swapped.
      EMIT_ASM("mov", IP, slo);
      EMIT_ASM("mov", dlo, shi);
      EMIT_ASM("mov", dhi, IP);
      return;
    }
    if (strcmp(dhi, shi) != 0)
      EMIT_ASM("mov", dhi, shi);
    EMIT_ASM("mov", dlo, slo);
    return;
  }
  if (strcmp(dlo, slo) != 0)
    EMIT_ASM("mov", dlo, slo);
  if (strcmp(dhi, shi) != 0)
    EMIT_ASM("mov", dhi, shi);
}

// A 64-bit constant into a register pair.
static void mov_immediate64(const char *lo, const char *hi, int64_t value) {
  mov_immediate(lo, (uint32_t)value);
  mov_immediate(hi, (uint32_t)((uint64_t)value >> 32));
}

// s-register moves of a double (no double precision VMOV on this FPU).
static void move_double(int dphys, int sphys) {
  if (dphys == sphys)
    return;
  EMIT_ASM("vmov.f32", kFReg32s[dphys], kFReg32s[sphys]);
  EMIT_ASM("vmov.f32", kFReg32s[dphys + 1], kFReg32s[sphys + 1]);
}

// Load/store addressing: [base, #offset] if it fits the instruction, else
// [base, ip] with ip = offset.
enum MemKind {
  MK_WORD,    // ldr/str/ldrb/strb/ldrh/strh: imm12 up, imm8 down
  MK_SIGNED,  // ldrsb/ldrsh: same
  MK_DUAL,    // ldrd/strd: +-1020, multiple of 4
  MK_VFP,     // vldr/vstr: +-1020, multiple of 4
};

static const char *mem_operand(const char *base, int64_t offset, enum MemKind kind) {
  bool fits;
  switch (kind) {
  case MK_WORD: case MK_SIGNED:
    fits = offset >= -255 && offset <= 4095;
    break;
  default:
    fits = offset >= -1020 && offset <= 1020 && (offset & 3) == 0;
    break;
  }
  if (fits)
    return offset != 0 ? fmt("[%s, #%" PRId64 "]", base, offset) : fmt("[%s]", base);
  mov_immediate(IP, offset);
  if (kind == MK_DUAL || kind == MK_VFP) {
    EMIT_ASM("add", IP, IP, base);
    return fmt("[%s]", IP);
  }
  return fmt("[%s, %s]", base, IP);
}

static const char *mem_dst(const VReg *opr, int64_t offset, bool spilled, enum MemKind kind) {
  if (!spilled) {
    assert(!(opr->flag & VRF_SPILLED));
    return mem_operand(R(opr), offset, kind);
  }
  assert(opr->flag & VRF_SPILLED);
  return mem_operand(FP, offset + opr->frame.offset, kind);
}

// Calls a run-time helper (veneered by the module linker when far away).
static void call_helper(const char *name) {
  EMIT_ASM("bl", name);
}

//

static void ei_bofs(IR *ir) {
  const char *dst = R(ir->dst);
  int64_t offset = ir->bofs.frameinfo->offset + ir->bofs.offset;
  if (offset >= 0 && offset <= 4095) {
    EMIT_ASM("add", dst, FP, im(offset));
  } else if (offset < 0 && offset >= -4095) {
    EMIT_ASM("sub", dst, FP, im(-offset));
  } else {
    mov_immediate(IP, offset);
    EMIT_ASM("add", dst, FP, IP);
  }
}

static void ei_iofs(IR *ir) {
  char *label = fmt_name(ir->iofs.label);
  if (ir->iofs.global)
    label = MANGLE(label);
  label = quote_label(label);
  const char *dst = R(ir->dst);
  if (ir->iofs.offset != 0) {
    EMIT_ASM("movw", dst, fmt("#:lower16:%s+%" PRId64, label, ir->iofs.offset));
    EMIT_ASM("movt", dst, fmt("#:upper16:%s+%" PRId64, label, ir->iofs.offset));
  } else {
    EMIT_ASM("movw", dst, fmt("#:lower16:%s", label));
    EMIT_ASM("movt", dst, fmt("#:upper16:%s", label));
  }
}

static void ei_sofs(IR *ir) {
  assert(ir->opr1->flag & VRF_CONST);
  const char *dst = R(ir->dst);
  int64_t offset = ir->opr1->fixnum;
  if (offset == 0) {
    EMIT_ASM("mov", dst, SP);
  } else if (offset > 0 && offset <= 4095) {
    EMIT_ASM("add", dst, SP, im(offset));
  } else {
    mov_immediate(IP, offset);
    EMIT_ASM("add", dst, SP, IP);
  }
}

// A load or store that may be unaligned (a packed struct): word accesses
// (unaligned LDR/STR are allowed), never LDRD/STRD/VLDR/VSTR.
static void unaligned_load(IR *ir) {
  VReg *dst = ir->dst;
  const char *base = R(ir->opr1);
  int64_t offset = ir->load.offset;
  if (offset < -255 || offset + 4 > 4095) {
    mov_immediate(IP, offset);
    EMIT_ASM("add", IP, IP, base);
    base = IP;
    offset = 0;
  }
  const char *m0 = fmt("[%s, #%" PRId64 "]", base, offset);
  const char *m1 = fmt("[%s, #%" PRId64 "]", base, offset + 4);
  if (dst->flag & VRF_FLONUM) {
    EMIT_ASM("ldr", IP, m0);
    EMIT_ASM("vmov", kFReg32s[dst->phys], IP);
    if (dst->vsize == SZ_DOUBLE) {
      EMIT_ASM("ldr", IP, m1);
      EMIT_ASM("vmov", kFReg32s[dst->phys + 1], IP);
    }
    return;
  }
  // A core register pair: the half that is the base last.
  if (strcmp(RLO(dst), base) == 0) {
    EMIT_ASM("ldr", RHI(dst), m1);
    EMIT_ASM("ldr", RLO(dst), m0);
  } else {
    EMIT_ASM("ldr", RLO(dst), m0);
    EMIT_ASM("ldr", RHI(dst), m1);
  }
}

static void unaligned_store(IR *ir) {
  VReg *src = ir->opr1;
  const char *base = R(ir->opr2);
  int64_t offset = ir->store.offset;
  bool flo = (src->flag & VRF_FLONUM) != 0;
  if (offset < -255 || offset + 4 > 4095) {
    EMIT_ASM("push", "{r0}");
    mov_immediate("r0", offset);
    EMIT_ASM("add", "r0", "r0", base);
    base = "r0";
    offset = 0;
    if (!flo) {
      // The value may be in r0: keep it simple, through the stack copy.
      assert(strcmp(RLO(src), "r0") != 0 && strcmp(RHI(src), "r0") != 0);
    }
  }
  const char *m0 = fmt("[%s, #%" PRId64 "]", base, offset);
  const char *m1 = fmt("[%s, #%" PRId64 "]", base, offset + 4);
  if (flo) {
    EMIT_ASM("vmov", IP, kFReg32s[src->phys]);
    EMIT_ASM("str", IP, m0);
    if (src->vsize == SZ_DOUBLE) {
      EMIT_ASM("vmov", IP, kFReg32s[src->phys + 1]);
      EMIT_ASM("str", IP, m1);
    }
  } else {
    EMIT_ASM("str", RLO(src), m0);
    EMIT_ASM("str", RHI(src), m1);
  }
  if (strcmp(base, "r0") == 0)
    EMIT_ASM("pop", "{r0}");
}

#define ei_load_s  ei_load
static const char *indexed(VReg *base, VReg *index, int shift) {
  assert(!(base->flag & VRF_CONST) && !(index->flag & VRF_CONST));
  return shift != 0 ? fmt("[%s, %s, lsl #%d]", R(base), R(index), shift)
                    : fmt("[%s, %s]", R(base), R(index));
}

// FPC: the next IR steps the address by a constant ARM32 post-indexing
// takes (-255..255): the step (that IR then emitted with the access), or 0
static IR *s_skip_ir;

// FPC: an address stepped by -255..255 right before a load or store of it
// (offset 0): the access pre-indexed with writeback (ldrb r, [p, #1]!), the
// step not emitted.  Whether the access can (as ei_load/ei_store do it).
static IR *s_pre_ir;
static int s_pre_step;

static bool pre_index_ok(IR *add, IR *nx) {
  VReg *p = add->dst;
  if (nx == NULL || add->opr1 != p || add->opr2 == NULL || !(add->opr2->flag & VRF_CONST) ||
      (add->opr2->flag & VRF_FLONUM) || add->additional_operands != NULL ||
      add->bfx.sh_kind != 0 || (p->flag & (VRF_SPILLED | VRF_FLONUM)) || p->vsize != VRegSize4)
    return false;
  int64_t c = add->kind == IR_ADD ? add->opr2->fixnum : -add->opr2->fixnum;
  if (c == 0 || c < -255 || c > 255)
    return false;
  if (nx->kind == IR_LOAD) {
    VReg *d = nx->dst;
    return nx->opr1 == p && nx->opr2 == NULL && nx->load.offset == 0 && !nx->load.unaligned &&
           !(d->flag & VRF_FLONUM) && d->vsize <= VRegSize4 && d->phys != p->phys;
  }
  if (nx->kind == IR_STORE) {
    VReg *s = nx->opr1;
    return nx->opr2 == p && nx->additional_operands == NULL && nx->store.offset == 0 &&
           !nx->store.unaligned && !(s->flag & (VRF_FLONUM | VRF_CONST)) && s->vsize <= VRegSize4 &&
           s->phys != p->phys;
  }
  return false;
}

static int post_step(VReg *addr) {
  IR *nx = emit_next_ir;
  if (nx == NULL || (nx->kind != IR_ADD && nx->kind != IR_SUB) || nx->dst != addr ||
      nx->opr1 != addr || nx->opr2 == NULL || !(nx->opr2->flag & VRF_CONST) ||
      (nx->opr2->flag & VRF_FLONUM) || nx->additional_operands != NULL || nx->bfx.sh_kind != 0 ||
      (addr->flag & VRF_SPILLED) || addr->vsize != VRegSize4)
    return 0;
  int64_t c = nx->kind == IR_ADD ? nx->opr2->fixnum : -nx->opr2->fixnum;
  return c >= -255 && c <= 255 ? (int)c : 0;
}

static void ei_load(IR *ir) {
  assert(!(ir->opr1->flag & VRF_CONST));
  bool spilled = ir->kind != IR_LOAD;
  VReg *dst = ir->dst;
  if (!spilled && ir->load.unaligned && (dst->vsize == VRegSize8 || (dst->flag & VRF_FLONUM))) {
    unaligned_load(ir);
    return;
  }
  if (dst->flag & VRF_FLONUM) {
    const char *src = mem_dst(ir->opr1, ir->load.offset, spilled, MK_VFP);
    EMIT_ASM("vldr", dst->vsize == SZ_DOUBLE ? D(dst) : S(dst), src);
    return;
  }
  if (dst->vsize == VRegSize8) {
    const char *src = mem_dst(ir->opr1, ir->load.offset, spilled, MK_DUAL);
    EMIT_ASM("ldrd", RLO(dst), RHI(dst), src);
    return;
  }
  bool u = (dst->flag & VRF_UNSIGNED) != 0;
  const char *op;
  enum MemKind mk = MK_WORD;
  if (!spilled && ir->opr2 != NULL) {  // [base, index, lsl #n] (fuse_indexed_access)
    switch (dst->vsize) {
    case VRegSize1:  op = u ? "ldrb" : "ldrsb"; break;
    case VRegSize2:  op = u ? "ldrh" : "ldrsh"; break;
    default:         op = "ldr"; break;
    }
    EMIT_ASM(op, R(dst), indexed(ir->opr1, ir->opr2, ir->load.index_shift));
    return;
  }
  switch (dst->vsize) {
  case VRegSize1:  op = u ? "ldrb" : "ldrsb"; if (!u) mk = MK_SIGNED; break;
  case VRegSize2:  op = u ? "ldrh" : "ldrsh"; if (!u) mk = MK_SIGNED; break;
  default:         op = "ldr"; break;
  }
  int step;
  if (ir == s_pre_ir) {  // FPC: pre-indexed, the step before it not emitted
    s_pre_ir = NULL;
    EMIT_ASM(op, R(dst), fmt("[%s, #%d]!", R(ir->opr1), s_pre_step));
    return;
  }
  if (!spilled && ir->load.offset == 0 && !ir->load.unaligned &&
      (step = post_step(ir->opr1)) != 0 && dst->phys != ir->opr1->phys) {
    EMIT_ASM(op, R(dst), fmt("[%s], #%d", R(ir->opr1), step));  // FPC: post-indexed
    s_skip_ir = emit_next_ir;
    return;
  }
  const char *src = mem_dst(ir->opr1, ir->load.offset, spilled, mk);
  EMIT_ASM(op, R(dst), src);
}

#define ei_store_s  ei_store
static void ei_store(IR *ir) {
  assert(!(ir->opr2->flag & VRF_CONST));
  bool spilled = ir->kind != IR_STORE;
  VReg *src = ir->opr1;
  if (!spilled && ir->store.unaligned && !(src->flag & VRF_CONST) &&
      (src->vsize == VRegSize8 || (src->flag & VRF_FLONUM))) {
    unaligned_store(ir);
    return;
  }
  if (src->flag & VRF_FLONUM) {
    assert(!(src->flag & VRF_CONST));
    const char *target = mem_dst(ir->opr2, ir->store.offset, spilled, MK_VFP);
    EMIT_ASM("vstr", src->vsize == SZ_DOUBLE ? D(src) : S(src), target);
    return;
  }
  if (src->vsize == VRegSize8) {
    assert(!(src->flag & VRF_CONST));  // tweak_irs
    const char *target = mem_dst(ir->opr2, ir->store.offset, spilled, MK_DUAL);
    EMIT_ASM("strd", RLO(src), RHI(src), target);
    return;
  }
  const char *op;
  switch (src->vsize) {
  case VRegSize1:  op = "strb"; break;
  case VRegSize2:  op = "strh"; break;
  default:         op = "str"; break;
  }
  if (!spilled && ir->additional_operands != NULL) {  // fuse_indexed_access
    VReg *index = ir->additional_operands->data[0];
    const char *target = indexed(ir->opr2, index, ir->store.index_shift);
    if (src->flag & VRF_CONST) {
      mov_immediate(IP, src->fixnum);
      EMIT_ASM(op, IP, target);
    } else {
      EMIT_ASM(op, R(src), target);
    }
    return;
  }
  int step;
  if (ir == s_pre_ir) {  // FPC: pre-indexed, the step before it not emitted
    s_pre_ir = NULL;
    EMIT_ASM(op, R(src), fmt("[%s, #%d]!", R(ir->opr2), s_pre_step));
    return;
  }
  if (!spilled && ir->store.offset == 0 && !ir->store.unaligned && !(src->flag & VRF_CONST) &&
      (step = post_step(ir->opr2)) != 0 && src->phys != ir->opr2->phys) {
    EMIT_ASM(op, R(src), fmt("[%s], #%d", R(ir->opr2), step));  // FPC: post-indexed
    s_skip_ir = emit_next_ir;
    return;
  }
  if (src->flag & VRF_CONST) {
    // The value in ip, so the address must be computed without it.
    const char *target;
    int64_t offset = ir->store.offset + (spilled ? ir->opr2->frame.offset : 0);
    const char *base = spilled ? FP : R(ir->opr2);
    if (offset >= -255 && offset <= 4095) {
      target = offset != 0 ? fmt("[%s, #%" PRId64 "]", base, offset) : fmt("[%s]", base);
      mov_immediate(IP, src->fixnum);
      EMIT_ASM(op, IP, target);
    } else {
      EMIT_ASM("push", "{r0}");
      mov_immediate("r0", offset);
      EMIT_ASM("add", "r0", "r0", base);
      mov_immediate(IP, src->fixnum);
      EMIT_ASM(op, IP, "[r0]");
      EMIT_ASM("pop", "{r0}");
    }
    return;
  }
  const char *target = mem_dst(ir->opr2, ir->store.offset, spilled, MK_WORD);
  EMIT_ASM(op, R(src), target);
}

// 32-bit operation with an immediate when it encodes, else through ip.
static void alu_rri(const char *op, const char *dst, const char *opr1, int64_t imm,
                    const char *alt_op) {
  uint32_t u = (uint32_t)imm;
  if (is_thumb_modimm(u)) {
    EMIT_ASM(op, dst, opr1, im(u));
  } else if (alt_op != NULL && is_thumb_modimm(-u)) {
    EMIT_ASM(alt_op, dst, opr1, im(-u));
  } else if ((strcmp(op, "add") == 0 || strcmp(op, "sub") == 0) && u <= 4095) {
    EMIT_ASM(op, dst, opr1, im(u));  // addw/subw
  } else {
    mov_immediate(IP, imm);
    EMIT_ASM(op, dst, opr1, IP);
  }
}

static void ei_fbinop(IR *ir, const char *op, const char *helper) {
  VReg *dst = ir->dst, *o1 = ir->opr1, *o2 = ir->opr2;
  assert(!(o1->flag & VRF_CONST) && !(o2->flag & VRF_CONST));
  if (dst->vsize == SZ_FLOAT) {
    EMIT_ASM(op, S(dst), S(o1), S(o2));
  } else {
    EMIT_ASM("vmov", "r0", "r1", D(o1));
    EMIT_ASM("vmov", "r2", "r3", D(o2));
    call_helper(helper);
    EMIT_ASM("vmov", D(dst), "r0", "r1");
  }
}

// 64-bit lo/hi binary operation: lo with op_lo, hi with op_hi.
static void ei_binop64(IR *ir, const char *op_lo, const char *op_hi) {
  VReg *dst = ir->dst, *o1 = ir->opr1, *o2 = ir->opr2;
  assert(!(o1->flag & VRF_CONST) && !(o2->flag & VRF_CONST));
  const char *dlo = RLO(dst), *dhi = RHI(dst);
  bool hazard = dst->phys == o1->phys + 1 || dst->phys == o2->phys + 1;
  const char *lo = hazard ? IP : dlo;
  EMIT_ASM(op_lo, lo, RLO(o1), RLO(o2));
  EMIT_ASM(op_hi, dhi, RHI(o1), RHI(o2));
  if (hazard)
    EMIT_ASM("mov", dlo, IP);
}

static const char *shifted_opr2(IR *ir);  // FPC

static void ei_add(IR *ir) {
  if (ir == s_skip_ir) {  // FPC: done by the access before (post-indexed)
    s_skip_ir = NULL;
    return;
  }
  if (pre_index_ok(ir, emit_next_ir)) {  // FPC: done by the access after (pre-indexed)
    s_pre_ir = emit_next_ir;
    s_pre_step = (int)ir->opr2->fixnum;
    return;
  }
  VReg *dst = ir->dst;
  if (ir->additional_operands != NULL && ir->opr2 == NULL) {  // FPC: fuse_arm_ops
    EMIT_ASM("mla", R(dst), R((VReg*)ir->additional_operands->data[0]),
             R((VReg*)ir->additional_operands->data[1]), R(ir->opr1));
  } else if (ir->bfx.sh_kind != 0) {
    EMIT_ASM("add", R(dst), R(ir->opr1), shifted_opr2(ir));
  } else if (dst->flag & VRF_FLONUM) {
    ei_fbinop(ir, "vadd.f32", "__aeabi_dadd");
  } else if (dst->vsize == VRegSize8) {
    ei_binop64(ir, "adds", "adc");
  } else {
    assert(!(ir->opr1->flag & VRF_CONST));
    if (ir->opr2->flag & VRF_CONST) {
      int64_t v = (int32_t)ir->opr2->fixnum;
      if (v == 0) {
        if (dst->phys != ir->opr1->phys)
          EMIT_ASM("mov", R(dst), R(ir->opr1));
      } else if (v < 0) {
        alu_rri("sub", R(dst), R(ir->opr1), -v, NULL);
      } else {
        alu_rri("add", R(dst), R(ir->opr1), v, NULL);
      }
    } else {
      EMIT_ASM("add", R(dst), R(ir->opr1), R(ir->opr2));
    }
  }
}

static void ei_sub(IR *ir) {
  if (ir == s_skip_ir) {  // FPC: done by the access before (post-indexed)
    s_skip_ir = NULL;
    return;
  }
  if (pre_index_ok(ir, emit_next_ir)) {  // FPC: done by the access after (pre-indexed)
    s_pre_ir = emit_next_ir;
    s_pre_step = -(int)ir->opr2->fixnum;
    return;
  }
  VReg *dst = ir->dst;
  if (ir->additional_operands != NULL && ir->opr2 == NULL) {  // FPC: fuse_arm_ops
    EMIT_ASM("mls", R(dst), R((VReg*)ir->additional_operands->data[0]),
             R((VReg*)ir->additional_operands->data[1]), R(ir->opr1));
  } else if (ir->bfx.sh_kind != 0) {
    EMIT_ASM("sub", R(dst), R(ir->opr1), shifted_opr2(ir));
  } else if (dst->flag & VRF_FLONUM) {
    ei_fbinop(ir, "vsub.f32", "__aeabi_dsub");
  } else if (dst->vsize == VRegSize8) {
    ei_binop64(ir, "subs", "sbc");
  } else {
    if (ir->opr1->flag & VRF_CONST) {
      // const - reg: rsb
      alu_rri("rsb", R(dst), R(ir->opr2), ir->opr1->fixnum, NULL);
    } else if (ir->opr2->flag & VRF_CONST) {
      int64_t v = (int32_t)ir->opr2->fixnum;
      if (v == 0) {
        if (dst->phys != ir->opr1->phys)
          EMIT_ASM("mov", R(dst), R(ir->opr1));
      } else if (v < 0) {
        alu_rri("add", R(dst), R(ir->opr1), -v, NULL);
      } else {
        alu_rri("sub", R(dst), R(ir->opr1), v, NULL);
      }
    } else {
      EMIT_ASM("sub", R(dst), R(ir->opr1), R(ir->opr2));
    }
  }
}

static void ei_mul(IR *ir) {
  VReg *dst = ir->dst, *o1 = ir->opr1, *o2 = ir->opr2;
  if (dst->flag & VRF_FLONUM) {
    ei_fbinop(ir, "vmul.f32", "__aeabi_dmul");
  } else if (dst->vsize == VRegSize8) {
    assert(!(o1->flag & VRF_CONST) && !(o2->flag & VRF_CONST));
    // lo:hi = a_lo * b_lo (64 bits) + ((a_lo * b_hi + a_hi * b_lo) << 32)
    EMIT_ASM("mul", IP, RLO(o1), RHI(o2));
    EMIT_ASM("mla", IP, RHI(o1), RLO(o2), IP);
    EMIT_ASM("umull", RLO(dst), RHI(dst), RLO(o1), RLO(o2));
    EMIT_ASM("add", RHI(dst), RHI(dst), IP);
  } else {
    assert(!(o1->flag & VRF_CONST) && !(o2->flag & VRF_CONST));  // tweak_irs
    EMIT_ASM("mul", R(dst), R(o1), R(o2));
  }
}

// 64-bit division/remainder: __aeabi_[u]ldivmod (r0:r1 / r2:r3: quotient
// r0:r1, remainder r2:r3).  The operands are not in r0-r3 (the allocator).
static void divmod64(IR *ir, bool mod) {
  VReg *dst = ir->dst, *o1 = ir->opr1, *o2 = ir->opr2;
  assert(!(o1->flag & VRF_CONST) && !(o2->flag & VRF_CONST));
  move_pair("r0", "r1", RLO(o1), RHI(o1));
  move_pair("r2", "r3", RLO(o2), RHI(o2));
  call_helper(o1->flag & VRF_UNSIGNED ? "__aeabi_uldivmod" : "__aeabi_ldivmod");
  if (mod)
    move_pair(RLO(dst), RHI(dst), "r2", "r3");
  else
    move_pair(RLO(dst), RHI(dst), "r0", "r1");
}

static void ei_div(IR *ir) {
  VReg *dst = ir->dst;
  if (dst->flag & VRF_FLONUM) {
    ei_fbinop(ir, "vdiv.f32", "__aeabi_ddiv");
  } else if (dst->vsize == VRegSize8) {
    divmod64(ir, false);
  } else {
    assert(!(ir->opr1->flag & VRF_CONST) && !(ir->opr2->flag & VRF_CONST));
    EMIT_ASM(ir->opr1->flag & VRF_UNSIGNED ? "udiv" : "sdiv", R(dst), R(ir->opr1), R(ir->opr2));
  }
}

static void ei_mod(IR *ir) {
  VReg *dst = ir->dst;
  if (dst->vsize == VRegSize8) {
    divmod64(ir, true);
    return;
  }
  assert(!(ir->opr1->flag & VRF_CONST) && !(ir->opr2->flag & VRF_CONST));
  const char *num = R(ir->opr1), *div = R(ir->opr2);
  EMIT_ASM(ir->opr1->flag & VRF_UNSIGNED ? "udiv" : "sdiv", IP, num, div);
  EMIT_ASM("mls", R(dst), IP, div, num);
}

// FPC: opr2 shifted (fuse_arm_ops): "rm, lsl #n"
static const char *shifted_opr2(IR *ir) {
  static const char *kKind[] = {['l'] = "lsl", ['r'] = "lsr", ['a'] = "asr"};
  return fmt("%s, %s #%d", R(ir->opr2), kKind[(int)ir->bfx.sh_kind], ir->bfx.sh_amount);
}

static void ei_logic(IR *ir, const char *op, const char *inv_op) {
  VReg *dst = ir->dst, *o1 = ir->opr1, *o2 = ir->opr2;
  if (ir->bfx.sh_kind != 0) {
    EMIT_ASM(op, R(dst), R(o1), shifted_opr2(ir));
    return;
  }
  if (dst->vsize == VRegSize8) {
    ei_binop64(ir, op, op);
    return;
  }
  if (o1->flag & VRF_CONST) {
    VReg *t = o1; o1 = o2; o2 = t;
  }
  assert(!(o1->flag & VRF_CONST));
  if (o2->flag & VRF_CONST) {
    uint32_t u = (uint32_t)o2->fixnum;
    if (is_thumb_modimm(u))
      EMIT_ASM(op, R(dst), R(o1), im(u));
    else if (inv_op != NULL && is_thumb_modimm(~u))
      EMIT_ASM(inv_op, R(dst), R(o1), im(~u));
    else
      EMIT_ASM(op, R(dst), R(o1), reg_or_ip(o2));
  } else {
    EMIT_ASM(op, R(dst), R(o1), R(o2));
  }
}

static void ei_bitand(IR *ir) {
  if (ir->bfx.shift != 0) {  // FPC: fuse_arm_ops
    int w = 0;
    for (uint64_t m = (uint64_t)ir->opr2->fixnum; m != 0; m >>= 1)
      ++w;
    EMIT_ASM("ubfx", R(ir->dst), R(ir->opr1), im(ir->bfx.shift), im(w));
    return;
  }
  ei_logic(ir, "and", "bic");
}
static void ei_bitor(IR *ir) { ei_logic(ir, "orr", "orn"); }
static void ei_bitxor(IR *ir) { ei_logic(ir, "eor", NULL); }

// 64-bit shifts by a constant, inline.
static void shift64_const(IR *ir, int kind) {  // 0: lsl, 1: lsr, 2: asr
  VReg *dst = ir->dst, *o1 = ir->opr1;
  int n = (int)(ir->opr2->fixnum & 63);
  const char *dlo = RLO(dst), *dhi = RHI(dst), *slo = RLO(o1), *shi = RHI(o1);
  if (n == 0) {
    move_pair(dlo, dhi, slo, shi);
    return;
  }
  switch (kind) {
  case 0:  // lsl
    if (n >= 32) {
      if (n > 32)
        EMIT_ASM("lsl", dhi, slo, im(n - 32));
      else if (strcmp(dhi, slo) != 0)
        EMIT_ASM("mov", dhi, slo);
      EMIT_ASM("mov", dlo, im(0));
    } else {
      EMIT_ASM("lsl", IP, shi, im(n));
      EMIT_ASM("orr", IP, IP, fmt("%s, lsr #%d", slo, 32 - n));
      EMIT_ASM("lsl", dlo, slo, im(n));
      EMIT_ASM("mov", dhi, IP);
    }
    break;
  case 1: case 2:  // lsr, asr
    {
      const char *sh = kind == 1 ? "lsr" : "asr";
      if (n >= 32) {
        if (n > 32)
          EMIT_ASM(sh, dlo, shi, im(n - 32));
        else if (strcmp(dlo, shi) != 0)
          EMIT_ASM("mov", dlo, shi);
        if (kind == 1)
          EMIT_ASM("mov", dhi, im(0));
        else
          EMIT_ASM("asr", dhi, shi, im(31));
      } else {
        EMIT_ASM("lsr", IP, slo, im(n));
        EMIT_ASM("orr", IP, IP, fmt("%s, lsl #%d", shi, 32 - n));
        EMIT_ASM(sh, dhi, shi, im(n));
        EMIT_ASM("mov", dlo, IP);
      }
    }
    break;
  }
}

// 64-bit shifts by a variable: __aeabi_llsl/llsr/lasr (r0:r1 by r2).
static void shift64_var(IR *ir, const char *helper) {
  VReg *dst = ir->dst, *o1 = ir->opr1, *o2 = ir->opr2;
  move_pair("r0", "r1", RLO(o1), RHI(o1));
  // The count is 32 or 64 bits: its low word.
  EMIT_ASM("mov", "r2", RLO(o2));
  call_helper(helper);
  move_pair(RLO(dst), RHI(dst), "r0", "r1");
}

static void ei_lshift(IR *ir) {
  VReg *dst = ir->dst;
  assert(!(ir->opr1->flag & VRF_CONST));
  if (dst->vsize == VRegSize8) {
    if (ir->opr2->flag & VRF_CONST)
      shift64_const(ir, 0);
    else
      shift64_var(ir, "__aeabi_llsl");
    return;
  }
  if (ir->opr2->flag & VRF_CONST)
    EMIT_ASM("lsl", R(dst), R(ir->opr1), im(ir->opr2->fixnum & 31));
  else
    EMIT_ASM("lsl", R(dst), R(ir->opr1), R(ir->opr2));
}

static void ei_rshift(IR *ir) {
  VReg *dst = ir->dst;
  assert(!(ir->opr1->flag & VRF_CONST));
  bool u = (dst->flag & VRF_UNSIGNED) != 0;
  if (dst->vsize == VRegSize8) {
    if (ir->opr2->flag & VRF_CONST)
      shift64_const(ir, u ? 1 : 2);
    else
      shift64_var(ir, u ? "__aeabi_llsr" : "__aeabi_lasr");
    return;
  }
  const char *op = u ? "lsr" : "asr";
  if (ir->opr2->flag & VRF_CONST) {
    int n = (int)(ir->opr2->fixnum & 31);
    if (n == 0 && dst->phys != ir->opr1->phys)
      EMIT_ASM("mov", R(dst), R(ir->opr1));
    else if (n != 0)
      EMIT_ASM(op, R(dst), R(ir->opr1), im(n));
  } else {
    EMIT_ASM(op, R(dst), R(ir->opr1), R(ir->opr2));
  }
}

static void ei_neg(IR *ir) {
  VReg *dst = ir->dst, *o1 = ir->opr1;
  assert(!(o1->flag & VRF_CONST));
  if (ir->bfx.sbfx) {  // FPC: fuse_arm_ops: -(x & 1)
    EMIT_ASM("sbfx", R(dst), R(o1), im(0), im(1));
    return;
  }
  if (dst->flag & VRF_FLONUM) {
    if (dst->vsize == SZ_FLOAT) {
      EMIT_ASM("vneg.f32", S(dst), S(o1));
    } else {
      // The sign is the top bit of the high half.
      if (dst->phys != o1->phys)
        EMIT_ASM("vmov.f32", kFReg32s[dst->phys], kFReg32s[o1->phys]);
      EMIT_ASM("vneg.f32", kFReg32s[dst->phys + 1], kFReg32s[o1->phys + 1]);
    }
  } else if (dst->vsize == VRegSize8) {
    // 0 - x
    EMIT_ASM("rsbs", IP, RLO(o1), im(0));
    EMIT_ASM("sbc", RHI(dst), RHI(o1), fmt("%s, lsl #1", RHI(o1)));
    EMIT_ASM("mov", RLO(dst), IP);
  } else {
    EMIT_ASM("rsb", R(dst), R(o1), im(0));
  }
}

static void ei_bitnot(IR *ir) {
  VReg *dst = ir->dst, *o1 = ir->opr1;
  assert(!(o1->flag & VRF_CONST));
  if (dst->vsize == VRegSize8) {
    if (dst->phys == o1->phys + 1) {
      EMIT_ASM("mvn", RHI(dst), RHI(o1));
      EMIT_ASM("mvn", RLO(dst), RLO(o1));
    } else {
      EMIT_ASM("mvn", RLO(dst), RLO(o1));
      EMIT_ASM("mvn", RHI(dst), RHI(o1));
    }
  } else {
    EMIT_ASM("mvn", R(dst), R(o1));
  }
}

// int <-> int of any size; the 32-bit registers hold values of 1 and 2
// bytes extended to 32 bits as their type says.
static void cast_fix(VReg *dst, VReg *src) {
  int pows = src->vsize, powd = dst->vsize;
  bool du = (dst->flag & VRF_UNSIGNED) != 0, su = (src->flag & VRF_UNSIGNED) != 0;
  if (powd == VRegSize8) {
    const char *lo = RLO(dst), *hi = RHI(dst);
    if (pows == VRegSize8) {
      move_pair(lo, hi, RLO(src), RHI(src));
      return;
    }
    if (strcmp(lo, R(src)) != 0)
      EMIT_ASM("mov", lo, R(src));
    if (su)
      EMIT_ASM("mov", hi, im(0));
    else
      EMIT_ASM("asr", hi, lo, im(31));
    return;
  }
  const char *d = R(dst);
  const char *s = pows == VRegSize8 ? RLO(src) : R(src);
  if (powd < VRegSize4 && (powd < pows || (powd == pows && du != su) || (powd > pows && du && !su))) {
    if (powd == VRegSize1)
      EMIT_ASM(du ? "uxtb" : "sxtb", d, s);
    else
      EMIT_ASM(du ? "uxth" : "sxth", d, s);
    return;
  }
  if (dst->phys != src->phys || pows == VRegSize8)
    if (strcmp(d, s) != 0)
      EMIT_ASM("mov", d, s);
}

static void ei_cast(IR *ir) {
  VReg *dst = ir->dst, *src = ir->opr1;
  assert(!(src->flag & VRF_CONST));
  bool df = (dst->flag & VRF_FLONUM) != 0, sf = (src->flag & VRF_FLONUM) != 0;
  if (ir->cast.trunc != 0) {  // fuse_arm_ops: (float)(int)f, the int in s15
    bool u = ir->cast.trunc == 'u';
    EMIT_ASM(u ? "vcvt.u32.f32" : "vcvt.s32.f32", "s15", S(src));
    EMIT_ASM(u ? "vcvt.f32.u32" : "vcvt.f32.s32", S(dst), "s15");
    return;
  }
  if (!df && !sf && ir->cast.fits && src->vsize <= VRegSize4) {  // mark_fitting_casts
    if (dst->phys != src->phys)
      EMIT_ASM("mov", R(dst), R(src));
    return;
  }
  if (!df && !sf) {
    cast_fix(dst, src);
    return;
  }
  if (df && sf) {
    if (dst->vsize == src->vsize) {
      if (dst->vsize == SZ_FLOAT) {
        if (dst->phys != src->phys)
          EMIT_ASM("vmov.f32", S(dst), S(src));
      } else {
        move_double(dst->phys, src->phys);
      }
    } else if (dst->vsize == SZ_DOUBLE) {  // float -> double
      EMIT_ASM("vmov", "r0", S(src));
      call_helper("__aeabi_f2d");
      EMIT_ASM("vmov", D(dst), "r0", "r1");
    } else {  // double -> float
      EMIT_ASM("vmov", "r0", "r1", D(src));
      call_helper("__aeabi_d2f");
      EMIT_ASM("vmov", S(dst), "r0");
    }
    return;
  }
  if (df) {  // int -> floating point
    bool su = (src->flag & VRF_UNSIGNED) != 0;
    if (src->vsize == VRegSize8) {
      move_pair("r0", "r1", RLO(src), RHI(src));
      if (dst->vsize == SZ_FLOAT) {
        call_helper(su ? "__aeabi_ul2f" : "__aeabi_l2f");
        EMIT_ASM("vmov", S(dst), "r0");
      } else {
        call_helper(su ? "__aeabi_ul2d" : "__aeabi_l2d");
        EMIT_ASM("vmov", D(dst), "r0", "r1");
      }
    } else if (dst->vsize == SZ_FLOAT) {
      EMIT_ASM("vmov", S(dst), R(src));
      EMIT_ASM(su && src->vsize == VRegSize4 ? "vcvt.f32.u32" : "vcvt.f32.s32", S(dst), S(dst));
    } else {
      EMIT_ASM("mov", "r0", R(src));
      call_helper(su && src->vsize == VRegSize4 ? "__aeabi_ui2d" : "__aeabi_i2d");
      EMIT_ASM("vmov", D(dst), "r0", "r1");
    }
    return;
  }
  // floating point -> int
  bool du = (dst->flag & VRF_UNSIGNED) != 0;
  if (dst->vsize == VRegSize8) {
    if (src->vsize == SZ_FLOAT) {
      EMIT_ASM("vmov", "r0", S(src));
      call_helper(du ? "__aeabi_f2ulz" : "__aeabi_f2lz");
    } else {
      EMIT_ASM("vmov", "r0", "r1", D(src));
      call_helper(du ? "__aeabi_d2ulz" : "__aeabi_d2lz");
    }
    move_pair(RLO(dst), RHI(dst), "r0", "r1");
    return;
  }
  if (src->vsize == SZ_FLOAT) {
    const char *t = kFReg32s[FREG_SCRATCH];
    EMIT_ASM(du && dst->vsize == VRegSize4 ? "vcvt.u32.f32" : "vcvt.s32.f32", t, S(src));
    EMIT_ASM("vmov", R(dst), t);
  } else {
    EMIT_ASM("vmov", "r0", "r1", D(src));
    call_helper(du && dst->vsize == VRegSize4 ? "__aeabi_d2uiz" : "__aeabi_d2iz");
    if (dst->phys != 0)
      EMIT_ASM("mov", R(dst), "r0");
  }
  if (dst->vsize < VRegSize4) {
    if (dst->vsize == VRegSize1)
      EMIT_ASM(du ? "uxtb" : "sxtb", R(dst), R(dst));
    else
      EMIT_ASM(du ? "uxth" : "sxth", R(dst), R(dst));
  }
}

// dst <- opr1 into the physical register(s) dphys.
static void emit_mov(int dphys, VReg *opr1) {
  if (opr1->flag & VRF_FLONUM) {
    assert(!(opr1->flag & VRF_CONST));
    if (opr1->vsize == SZ_FLOAT) {
      if (opr1->phys != dphys)
        EMIT_ASM("vmov.f32", kFReg32s[dphys], S(opr1));
    } else {
      move_double(dphys, opr1->phys);
    }
    return;
  }
  if (opr1->vsize == VRegSize8) {
    if (opr1->flag & VRF_CONST)
      mov_immediate64(kReg32s[dphys], kReg32s[dphys + 1], opr1->fixnum);
    else
      move_pair(kReg32s[dphys], kReg32s[dphys + 1], RLO(opr1), RHI(opr1));
    return;
  }
  if (opr1->flag & VRF_CONST)
    mov_immediate(kReg32s[dphys], opr1->fixnum);
  else if (opr1->phys != dphys)
    EMIT_ASM("mov", kReg32s[dphys], R(opr1));
}

static void ei_mov(IR *ir) {
  emit_mov(ir->dst->phys, ir->opr1);
}

// A variadic function returns floating point by the base standard (r0,
// r0:r1), as it takes its arguments.
static bool s_variadic_func;

void arm32_set_variadic_func(bool variadic) {
  s_variadic_func = variadic;
}

static void ei_result(IR *ir) {
  if (ir->opr1 == NULL)  // No effect.
    return;
  // r0 (r0:r1), s0, d0
  assert(ir->result.index == 0);
  VReg *v = ir->opr1;
  if (s_variadic_func && (v->flag & VRF_FLONUM)) {
    if (v->vsize == SZ_FLOAT)
      EMIT_ASM("vmov", "r0", S(v));
    else
      EMIT_ASM("vmov", "r0", "r1", D(v));
    return;
  }
  emit_mov(0, v);
}

static const char *kCondTable[][8] = {
  {  // Signed.
    [COND_EQ] = "eq", [COND_NE] = "ne",
    [COND_LT] = "lt", [COND_GT] = "gt", [COND_LE] = "le", [COND_GE] = "ge",
  },
  {  // Unsigned.
    [COND_EQ] = "eq", [COND_NE] = "ne",
    [COND_LT] = "lo", [COND_GT] = "hi", [COND_LE] = "ls", [COND_GE] = "hs",
  },
};

static const char *kInvCond[][2] = {
  {"eq", "ne"}, {"ne", "eq"}, {"lt", "ge"}, {"ge", "lt"}, {"gt", "le"}, {"le", "gt"},
  {"lo", "hs"}, {"hs", "lo"}, {"hi", "ls"}, {"ls", "hi"}, {"mi", "pl"}, {"pl", "mi"},
};

static const char *inv_cond(const char *c) {
  for (size_t i = 0; i < ARRAY_SIZE(kInvCond); ++i)
    if (strcmp(kInvCond[i][0], c) == 0)
      return kInvCond[i][1];
  assert(false);
  return c;
}

// Compares opr1 with opr2, returning the condition (mnemonic suffix) that
// is true when (opr1 cond opr2) holds.
static const char *cmp_vregs(VReg *opr1, VReg *opr2, int cond) {
  assert(opr1 != NULL && opr2 != NULL);
  if (opr1->flag & VRF_FLONUM) {
    assert(!(opr2->flag & VRF_CONST));
    if (opr1->vsize == SZ_FLOAT) {
      EMIT_ASM("vcmp.f32", S(opr1), S(opr2));
      EMIT_ASM("vmrs", "APSR_nzcv", "fpscr");
      switch (cond) {
      case COND_EQ: return "eq";
      case COND_NE: return "ne";
      case COND_LT: return "mi";
      case COND_LE: return "ls";
      case COND_GT: return "gt";
      case COND_GE: return "ge";
      default: assert(false); return "al";
      }
    }
    // Doubles: a helper answers 0 or 1 in r0.
    const char *helper;
    bool invert = false;
    switch (cond) {
    case COND_EQ: helper = "__aeabi_dcmpeq"; break;
    case COND_NE: helper = "__aeabi_dcmpeq"; invert = true; break;
    case COND_LT: helper = "__aeabi_dcmplt"; break;
    case COND_LE: helper = "__aeabi_dcmple"; break;
    case COND_GT: helper = "__aeabi_dcmpgt"; break;
    case COND_GE: helper = "__aeabi_dcmpge"; break;
    default: assert(false); helper = ""; break;
    }
    EMIT_ASM("vmov", "r0", "r1", D(opr1));
    EMIT_ASM("vmov", "r2", "r3", D(opr2));
    call_helper(helper);
    EMIT_ASM("cmp", "r0", im(0));
    return invert ? "eq" : "ne";
  }

  bool u = (opr1->flag & VRF_UNSIGNED) != 0;
  assert(!(opr1->flag & VRF_CONST));
  if (opr1->vsize == VRegSize8) {
    assert(!(opr2->flag & VRF_CONST));
    switch (cond) {
    case COND_EQ: case COND_NE:
      EMIT_ASM("cmp", RLO(opr1), RLO(opr2));
      EMIT_ASM("it", "eq");
      EMIT_ASM("cmpeq", RHI(opr1), RHI(opr2));
      return cond == COND_EQ ? "eq" : "ne";
    case COND_LT: case COND_GE:
      EMIT_ASM("cmp", RLO(opr1), RLO(opr2));
      EMIT_ASM("sbcs", IP, RHI(opr1), RHI(opr2));
      return kCondTable[u][cond];
    case COND_GT: case COND_LE:
      // b < a, b >= a
      EMIT_ASM("cmp", RLO(opr2), RLO(opr1));
      EMIT_ASM("sbcs", IP, RHI(opr2), RHI(opr1));
      return kCondTable[u][cond == COND_GT ? COND_LT : COND_GE];
    default: assert(false); return "al";
    }
  }

  if (opr2->flag & VRF_CONST) {
    int64_t v = (int32_t)opr2->fixnum;
    if (is_thumb_modimm((uint32_t)v))
      EMIT_ASM("cmp", R(opr1), im((uint32_t)v));
    else if (is_thumb_modimm((uint32_t)-v))
      EMIT_ASM("cmn", R(opr1), im((uint32_t)-v));
    else
      EMIT_ASM("cmp", R(opr1), reg_or_ip(opr2));
  } else {
    EMIT_ASM("cmp", R(opr1), R(opr2));
  }
  return kCondTable[u][cond];
}

static void ei_cond(IR *ir) {
  assert(!(ir->dst->flag & VRF_CONST));
  const char *c = cmp_vregs(ir->opr1, ir->opr2, ir->cond.kind);
  const char *dst = R(ir->dst);
  EMIT_ASM("ite", c);
  EMIT_ASM(fmt("mov%s", c), dst, im(1));
  EMIT_ASM(fmt("mov%s", inv_cond(c)), dst, im(0));
}

// FPC: the last instruction made `reg` with an operation that has an S
// variant (not in an IT block): made to set the flags (Z as a compare with
// 0 would)
static bool set_flags_of_last(const char *reg) {
  const char *line = emit_last_line();
  if (line == NULL || line[0] != '\t')
    return false;
  static const char *kOps[] = {
    "mov", "mvn", "add", "sub", "rsb", "and", "orr", "eor", "bic", "orn", "lsl", "lsr", "asr",
  };
  const char *p = line + 1;
  size_t n = strcspn(p, " ");
  size_t k;
  for (k = 0; k < sizeof(kOps) / sizeof(*kOps); ++k) {
    if (strlen(kOps[k]) == n && strncmp(p, kOps[k], n) == 0)
      break;
  }
  if (k >= sizeof(kOps) / sizeof(*kOps) || p[n] != ' ')
    return false;
  const char *args = p + n + 1;
  size_t rl = strlen(reg);
  if (strncmp(args, reg, rl) != 0 || args[rl] != ',')
    return false;
  // An immediate: one the S variant encodes (a modified immediate; any
  // shift amount, a shifted register's too)
  if (strchr(args, ':') != NULL || strchr(args, '[') != NULL)
    return false;
  const char *imm = strrchr(args, '#');
  bool shift_op = strncmp(p, "lsl", 3) == 0 || strncmp(p, "lsr", 3) == 0 || strncmp(p, "asr", 3) == 0;
  bool shifted_reg = strstr(args, ", lsl #") != NULL || strstr(args, ", lsr #") != NULL ||
                     strstr(args, ", asr #") != NULL;
  if (imm != NULL && !shift_op && !shifted_reg) {
    char *end;
    long long v = strtoll(imm + 1, &end, 0);
    if (!is_thumb_modimm((uint32_t)v))
      return false;
  }
  char buf[256];
  snprintf(buf, sizeof(buf), "\t%.*ss %s", (int)n, p, args);
  emit_replace_last_line(buf);
  return true;
}

static void ei_jmp(IR *ir) {
  const char *label = fmt_name(ir->jmp.bb->label);
  int cond = ir->jmp.cond;
  assert(cond != COND_NONE);
  if (cond == COND_ANY) {
    EMIT_ASM("b", label);
    return;
  }
  VReg *o1 = ir->opr1, *o2 = ir->opr2;
  if ((cond == COND_EQ || cond == COND_NE) && o2 != NULL && (o2->flag & VRF_CONST) &&
      !(o2->flag & VRF_FLONUM) && o2->fixnum == 0 && !(o1->flag & (VRF_CONST | VRF_FLONUM)) &&
      o1->vsize <= VRegSize4) {
    // FPC: a jump back (a loop's): an instruction just before that made
    // the register sets the flags instead of a compare
    if (ir->jmp.bb->index >= 0 && ir->jmp.bb->index <= emit_cur_bb_index &&
        set_flags_of_last(R(o1))) {
      EMIT_ASM(cond == COND_EQ ? "beq" : "bne", label);
      return;
    }
    // cbz/cbnz (the assembler makes it cmp and beq/bne when it can't)
    EMIT_ASM(cond == COND_EQ ? "cbz" : "cbnz", R(o1), label);
    return;
  }
  const char *c = cmp_vregs(ir->opr1, ir->opr2, cond);
  EMIT_ASM(fmt("b%s", c), label);
}

// Table jump: the table of the cases' addresses (Thumb: +1) in .rodata.
static void ei_tjmp(IR *ir) {
  assert(!(ir->opr1->flag & VRF_CONST));
  const Name *table_label = alloc_label();
  char *label = fmt_name(table_label);
  EMIT_ASM("movw", IP, fmt("#:lower16:%s", label));
  EMIT_ASM("movt", IP, fmt("#:upper16:%s", label));
  EMIT_ASM("ldr", PC, fmt("[%s, %s, lsl #2]", IP, RLO(ir->opr1)));

  _RODATA();
  EMIT_ALIGN(4);
  EMIT_LABEL(label);
  for (size_t i = 0, len = ir->tjmp.len; i < len; ++i) {
    BB *bb = ir->tjmp.bbs[i];
    _LONG(fmt("%.*s+1", NAMES(bb->label)));
  }
  _TEXT();
}

static void ei_pusharg(IR *ir) {
  VReg *opr1 = ir->opr1;
  int index = ir->pusharg.index;
  if ((opr1->flag & VRF_FLONUM) && ir->pusharg.fp_as_gp) {
    // Base standard: a float in r<index>, a double in r<index>:r<index+1>
    if (opr1->vsize == SZ_FLOAT)
      EMIT_ASM("vmov", kReg32s[index], S(opr1));
    else
      EMIT_ASM("vmov", kReg32s[index], kReg32s[index + 1], D(opr1));
    return;
  }
  // index is the register itself (aapcs.c): r<index>, s<index>
  emit_mov(index, opr1);
}

static void ei_call(IR *ir) {
  if (ir->call->label != NULL) {
    char *label = fmt_name(ir->call->label);
    if (ir->call->global)
      label = MANGLE(label);
    EMIT_ASM("bl", quote_label(label));
  } else {
    assert(!(ir->opr1->flag & VRF_CONST));
    EMIT_ASM("blx", R(ir->opr1));
  }

  const FrameInfo *fi = ir->call->small_struct_result_frameinfo;
  if (fi != NULL) {
    // A small struct (up to 4 bytes) comes back in r0: to its frame slot.
    int offset = fi->offset;
    size_t size = fi->size;
    assert(size > 0 && size <= 4);
    switch (size) {
    case 1:  EMIT_ASM("strb", "r0", mem_operand(FP, offset, MK_WORD)); break;
    case 2:  EMIT_ASM("strh", "r0", mem_operand(FP, offset, MK_WORD)); break;
    case 3:
      EMIT_ASM("strh", "r0", mem_operand(FP, offset, MK_WORD));
      EMIT_ASM("lsr", "r0", "r0", im(16));
      EMIT_ASM("strb", "r0", mem_operand(FP, offset + 2, MK_WORD));
      break;
    default: EMIT_ASM("str", "r0", mem_operand(FP, offset, MK_WORD)); break;
    }
  } else if (ir->dst != NULL) {
    VReg *dst = ir->dst;
    if ((dst->flag & VRF_FLONUM) && ir->call->vaarg_start >= 0) {
      // A variadic callee returns floating point in core registers.
      if (dst->vsize == SZ_FLOAT)
        EMIT_ASM("vmov", S(dst), "r0");
      else
        EMIT_ASM("vmov", D(dst), "r0", "r1");
    } else if (dst->flag & VRF_FLONUM) {
      if (dst->vsize == SZ_FLOAT) {
        if (dst->phys != 0)
          EMIT_ASM("vmov.f32", S(dst), "s0");
      } else {
        move_double(dst->phys, 0);
      }
    } else if (dst->vsize == VRegSize8) {
      move_pair(RLO(dst), RHI(dst), "r0", "r1");
    } else if (dst->phys != 0) {
      EMIT_ASM("mov", R(dst), "r0");
    }
  }
}

static void ei_subsp(IR *ir) {
  if (ir->opr1->flag & VRF_CONST) {
    int64_t v = ir->opr1->fixnum;
    assert(v % 8 == 0);
    if (v > 0)
      alu_rri("sub", SP, SP, v, NULL);
    else if (v < 0)
      alu_rri("add", SP, SP, -v, NULL);
  } else {
    EMIT_ASM("sub", SP, SP, R(ir->opr1));
  }
  if (ir->dst != NULL)
    EMIT_ASM("mov", R(ir->dst), SP);
}

static void ei_keep(IR *ir) {
  UNUSED(ir);
}

const EmitIrFunc kEmitIrFuncTable[] = {
  [IR_BOFS] = ei_bofs, [IR_IOFS] = ei_iofs, [IR_SOFS] = ei_sofs,
  [IR_LOAD] = ei_load, [IR_LOAD_S] = ei_load_s, [IR_STORE] = ei_store, [IR_STORE_S] = ei_store_s,

  [IR_ADD] = ei_add, [IR_SUB] = ei_sub, [IR_MUL] = ei_mul, [IR_DIV] = ei_div,
  [IR_MOD] = ei_mod, [IR_BITAND] = ei_bitand, [IR_BITOR] = ei_bitor,
  [IR_BITXOR] = ei_bitxor, [IR_LSHIFT] = ei_lshift, [IR_RSHIFT] = ei_rshift,
  [IR_COND] = ei_cond,

  [IR_NEG] = ei_neg, [IR_BITNOT] = ei_bitnot, [IR_CAST] = ei_cast,
  [IR_MOV] = ei_mov, [IR_RESULT] = ei_result,

  [IR_JMP] = ei_jmp, [IR_TJMP] = ei_tjmp,
  [IR_PUSHARG] = ei_pusharg, [IR_CALL] = ei_call,
  [IR_SUBSP] = ei_subsp, [IR_KEEP] = ei_keep,
};

//

// FPC: base + (index << n) for a load or store (an array's element) in its
// addressing mode, [base, index, lsl #n], when the sum and the shift are
// temporaries made for it in the same block.

static IR *def_in_bb(Vector *irs, int before, VReg *v, int *pos) {
  for (int k = before - 1; k >= 0; --k) {
    IR *ir = irs->data[k];
    if (ir->dst == v) {
      *pos = k;
      return ir;
    }
  }
  return NULL;
}

static bool defined_between(Vector *irs, int from, int to, VReg *v) {
  for (int k = from + 1; k < to; ++k) {
    IR *ir = irs->data[k];
    if (ir->dst == v)
      return true;
  }
  return false;
}

static void count_uses_defs(RegAlloc *ra, BBContainer *bbcon, int *uses, int *defs) {
  for (int i = 0; i < ra->vregs->len * 2; ++i)
    uses[i] = 0;
  for (int i = 0; i < bbcon->len; ++i) {
    BB *bb = bbcon->data[i];
    for (int j = 0; j < bb->irs->len; ++j) {
      IR *ir = bb->irs->data[j];
      VReg *operands[] = {ir->opr1, ir->opr2};
      int n = 2 + (ir->additional_operands != NULL ? ir->additional_operands->len : 0);
      for (int k = 0; k < n; ++k) {
        VReg *vreg = k < 2 ? operands[k] : ir->additional_operands->data[k - 2];
        if (vreg != NULL && !(vreg->flag & VRF_CONST))
          ++uses[vreg->virt];
      }
      if (ir->dst != NULL && !(ir->dst->flag & VRF_CONST))
        ++defs[ir->dst->virt];
    }
  }
}

static bool plain_reg(VReg *v) {
  return v != NULL && !(v->flag & (VRF_CONST | VRF_FLONUM | VRF_FORCEMEMORY | VRF_VOLATILEREG)) &&
         v->vsize == VRegSize4;
}

// An address base + constant used by several loads and stores in its block
// (`p->n++`, `a[2] += x`): the constant in each one's offset
static void fuse_offset_uses(Vector *irs, int k, int *uses, int *defs) {
  IR *add = irs->data[k];
  VReg *a = add->dst, *b = add->opr1;
  if ((add->kind != IR_ADD && add->kind != IR_SUB) || !plain_reg(a) || !plain_reg(b) ||
      !(add->opr2->flag & VRF_CONST) || (add->opr2->flag & VRF_FLONUM) || defs[a->virt] != 1 ||
      uses[a->virt] < 2 || (a->flag & VRF_PARAM) ||
      (a->flag & VRF_UNALIGNED) != (b->flag & VRF_UNALIGNED) || a == b)
    return;
  int found = 0;
  bool clobbered = false;
  for (int m = k + 1; m < irs->len && found < uses[a->virt]; ++m) {
    IR *u = irs->data[m];
    bool addr = (u->kind == IR_LOAD && u->opr1 == a && u->opr2 == NULL) ||
                (u->kind == IR_STORE && u->opr2 == a && u->opr1 != a &&
                 u->additional_operands == NULL);
    bool other = (u->opr1 == a && !(u->kind == IR_LOAD && addr)) ||
                 (u->opr2 == a && !(u->kind == IR_STORE && addr));
    if (u->additional_operands != NULL) {
      for (int n = 0; n < u->additional_operands->len; ++n)
        other |= u->additional_operands->data[n] == a;
    }
    if (other || (addr && clobbered))
      return;
    if (addr)
      ++found;
    if (u->dst == b || u->kind == IR_ASM)
      clobbered = true;
  }
  if (found != uses[a->virt])
    return;
  int64_t c = add->kind == IR_ADD ? add->opr2->fixnum : -add->opr2->fixnum;
  for (int m = k + 1, n = 0; n < found; ++m) {
    IR *u = irs->data[m];
    if (u->kind == IR_LOAD && u->opr1 == a) {
      u->opr1 = b;
      u->load.offset += c;
      ++n;
    } else if (u->kind == IR_STORE && u->opr2 == a) {
      u->opr2 = b;
      u->store.offset += c;
      ++n;
    }
  }
  uses[b->virt] += found - 1;
  uses[a->virt] = 0;
  vec_remove_at(irs, k);
}

static void fuse_indexed_access(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  int *uses = malloc_or_die(sizeof(*uses) * vreg_count * 2);
  int *defs = uses + vreg_count;
  count_uses_defs(ra, bbcon, uses, defs);
  for (int i = 0; i < bbcon->len; ++i) {
    Vector *irs = ((BB*)bbcon->data[i])->irs;
    for (int k = irs->len; --k >= 0; )
      fuse_offset_uses(irs, k, uses, defs);
  }
  for (int i = 0; i < bbcon->len; ++i) {
    BB *bb = bbcon->data[i];
    Vector *irs = bb->irs;
    for (int j = 0; j < irs->len; ++j) {
      IR *ir = irs->data[j];
      VReg *addr, *value = NULL;

      // base + constant: the constant in the instruction's offset (any load
      // or store: the emitter copes with an offset out of range)
      for (;;) {
        VReg *a = ir->kind == IR_LOAD && ir->opr2 == NULL ? ir->opr1 :
                  ir->kind == IR_STORE && ir->additional_operands == NULL ? ir->opr2 : NULL;
        if (a == NULL || !plain_reg(a) || uses[a->virt] != 1 || defs[a->virt] != 1 ||
            (a->flag & VRF_PARAM))
          break;
        int k;
        IR *add = def_in_bb(irs, j, a, &k);
        if (add == NULL || (add->kind != IR_ADD && add->kind != IR_SUB) || !plain_reg(add->opr1) ||
            !(add->opr2->flag & VRF_CONST) || defined_between(irs, k, j, add->opr1) ||
            (add->opr1->flag & VRF_UNALIGNED) != (a->flag & VRF_UNALIGNED))
          break;
        int64_t c = add->kind == IR_ADD ? add->opr2->fixnum : -add->opr2->fixnum;
        if (ir->kind == IR_LOAD) {
          ir->opr1 = add->opr1;
          ir->load.offset += c;
        } else {
          ir->opr2 = add->opr1;
          ir->store.offset += c;
        }
        ++uses[add->opr1->virt];
        vec_remove_at(irs, k);
        --j;
      }

      if (ir->kind == IR_LOAD) {
        if (ir->opr2 != NULL || ir->load.unaligned || ir->load.offset != 0 ||
            (ir->dst->flag & VRF_FLONUM) || ir->dst->vsize > VRegSize4)
          continue;
        addr = ir->opr1;
      } else if (ir->kind == IR_STORE) {
        value = ir->opr1;
        if (ir->additional_operands != NULL || ir->store.unaligned || ir->store.offset != 0 ||
            (value->flag & VRF_FLONUM) || value->vsize > VRegSize4)
          continue;
        addr = ir->opr2;
      } else {
        continue;
      }
      if (!plain_reg(addr) || uses[addr->virt] != 1 || defs[addr->virt] != 1 ||
          (addr->flag & (VRF_PARAM | VRF_UNALIGNED)))
        continue;
      int kadd;
      IR *add = def_in_bb(irs, j, addr, &kadd);
      if (add == NULL || add->kind != IR_ADD || !plain_reg(add->opr1) || !plain_reg(add->opr2))
        continue;
      VReg *base = NULL, *index = NULL;
      int shift = 0, kshl = -1;
      for (int side = 0; side < 2 && base == NULL; ++side) {
        VReg *x = side == 0 ? add->opr2 : add->opr1;
        VReg *b = side == 0 ? add->opr1 : add->opr2;
        if (uses[x->virt] != 1 || defs[x->virt] != 1 || (x->flag & VRF_PARAM))
          continue;
        int k;
        IR *shl = def_in_bb(irs, kadd, x, &k);
        if (shl != NULL && shl->kind == IR_LSHIFT && plain_reg(shl->opr1) &&
            (shl->opr2->flag & VRF_CONST) && shl->opr2->fixnum >= 0 && shl->opr2->fixnum <= 3 &&
            !defined_between(irs, k, j, shl->opr1) && shl->opr1 != addr) {
          base = b;
          index = shl->opr1;
          shift = shl->opr2->fixnum;
          kshl = k;
        }
      }
      if (base == NULL) {
        base = add->opr1;
        index = add->opr2;
      }
      if (defined_between(irs, kadd, j, base) || defined_between(irs, kadd, j, index))
        continue;
      if (ir->kind == IR_LOAD) {
        ir->opr1 = base;
        ir->opr2 = index;
        ir->load.index_shift = shift;
      } else {
        ir->opr2 = base;
        ir->additional_operands = new_vector();
        vec_push(ir->additional_operands, index);
        ir->store.index_shift = shift;
      }
      ++uses[base->virt];
      ++uses[index->virt];
      // The sum and the shift go (the later first)
      if (kshl > kadd) {
        vec_remove_at(irs, kshl);
        vec_remove_at(irs, kadd);
      } else {
        vec_remove_at(irs, kadd);
        if (kshl >= 0)
          vec_remove_at(irs, kshl);
      }
      j -= 1 + (kshl >= 0);
    }
  }
  free(uses);
}

// FPC: how many low bits of a 32-bit register value can be set (32: any,
// or negative), from the instructions before `pos` in the block making it.
// A narrow value is kept zero- or sign-extended in its register.
static int vreg_known_bits(Vector *irs, int pos, VReg *v, int depth) {
  if (v->flag & VRF_CONST) {
    if (v->flag & VRF_FLONUM)
      return 32;
    int64_t x = v->fixnum;
    if (x < 0 || x > 0xffffffffLL)
      return 32;
    int n = 0;
    while (x != 0) {
      ++n;
      x >>= 1;
    }
    return n;
  }
  if (v->flag & VRF_FLONUM || v->vsize > VRegSize4)
    return 32;
  int by_type = (v->flag & VRF_UNSIGNED) && v->vsize < VRegSize4 ? 8 << v->vsize : 32;
  if (depth <= 0 || (v->flag & (VRF_FORCEMEMORY | VRF_VOLATILEREG)))
    return by_type;
  int k;
  for (k = pos - 1; k >= 0; --k) {
    if (((IR*)irs->data[k])->dst == v)
      break;
  }
  if (k < 0)
    return by_type;
  IR *ir = irs->data[k];
  int a, b, r;
  switch (ir->kind) {
  case IR_ADD: case IR_SUB: case IR_BITAND: case IR_BITOR: case IR_BITXOR:
    if (ir->bfx.sh_kind != 0)  // (a shifted operand: fuse_arm_ops)
      return by_type;
    break;
  default:
    break;
  }
  switch (ir->kind) {
  case IR_MOV:
    r = vreg_known_bits(irs, k, ir->opr1, depth - 1);
    break;
  case IR_BITAND:
    a = vreg_known_bits(irs, k, ir->opr1, depth - 1);
    b = vreg_known_bits(irs, k, ir->opr2, depth - 1);
    r = a < b ? a : b;
    break;
  case IR_BITOR: case IR_BITXOR:
    a = vreg_known_bits(irs, k, ir->opr1, depth - 1);
    b = vreg_known_bits(irs, k, ir->opr2, depth - 1);
    r = a > b ? a : b;
    break;
  case IR_ADD:
    if (ir->additional_operands != NULL)  // mla
      return by_type;
    a = vreg_known_bits(irs, k, ir->opr1, depth - 1);
    b = vreg_known_bits(irs, k, ir->opr2, depth - 1);
    r = (a > b ? a : b) + 1;
    break;
  case IR_MUL:
    r = vreg_known_bits(irs, k, ir->opr1, depth - 1) + vreg_known_bits(irs, k, ir->opr2, depth - 1);
    break;
  case IR_RSHIFT:
    if (!(ir->opr2->flag & VRF_CONST) || ir->opr2->fixnum < 0 || ir->opr2->fixnum > 31)
      return by_type;
    a = vreg_known_bits(irs, k, ir->opr1, depth - 1);
    if (a >= 32 && !(ir->opr1->flag & VRF_UNSIGNED && ir->opr1->vsize == VRegSize4))
      return by_type;
    r = a - (int)ir->opr2->fixnum;
    if (r < 0)
      r = 0;
    break;
  case IR_LSHIFT:
    if (!(ir->opr2->flag & VRF_CONST) || ir->opr2->fixnum < 0 || ir->opr2->fixnum > 31)
      return by_type;
    r = vreg_known_bits(irs, k, ir->opr1, depth - 1) + (int)ir->opr2->fixnum;
    break;
  case IR_COND:
    r = 1;
    break;
  case IR_CAST:
    if ((ir->opr1->flag & VRF_FLONUM) || ir->opr1->vsize > VRegSize4)
      return by_type;
    a = vreg_known_bits(irs, k, ir->opr1, depth - 1);
    if (v->vsize < VRegSize4) {
      int d = 8 << v->vsize;
      r = (v->flag & VRF_UNSIGNED) ? (a < d ? a : d) : (a < d ? a : 32);
    } else {
      r = a;
    }
    break;
  default:
    return by_type;
  }
  if (r > 32)
    r = 32;
  return r < by_type ? r : by_type;
}

static void mark_fitting_casts(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  int *uses = malloc_or_die(sizeof(*uses) * vreg_count * 2);
  int *defs = uses + vreg_count;
  count_uses_defs(ra, bbcon, uses, defs);
  for (int i = 0; i < bbcon->len; ++i) {
    BB *bb = bbcon->data[i];
    Vector *irs = bb->irs;
    for (int j = 0; j < irs->len; ++j) {
      IR *ir = irs->data[j];
      if (ir->kind != IR_CAST)
        continue;
      ir->cast.fits = false;
      VReg *dst = ir->dst, *src = ir->opr1;
      if ((dst->flag | src->flag) & VRF_FLONUM || dst->vsize >= VRegSize4 ||
          src->vsize > VRegSize4 || (src->flag & VRF_CONST))
        continue;
      int d = 8 << dst->vsize;
      int k = vreg_known_bits(irs, j, src, 6);
      ir->cast.fits = (dst->flag & VRF_UNSIGNED) ? k <= d : k < d;
      // FPC: used only by the next store of its size (which keeps its low
      // bits only): no extension
      if (!ir->cast.fits && j + 1 < irs->len && !(dst->flag & (VRF_FORCEMEMORY | VRF_VOLATILEREG)) &&
          dst->virt < vreg_count && uses[dst->virt] == 1 && defs[dst->virt] == 1) {
        IR *st = irs->data[j + 1];
        if (st->kind == IR_STORE && st->opr1 == dst && st->opr2 != dst &&
            (st->additional_operands == NULL || st->additional_operands->data[0] != dst))
          ir->cast.fits = true;
      }
    }
  }
  free(uses);
}

// FPC: a multiplication added to (subtracted from) a register: mla (mls);
// a right shift masked to its low bits: ubfx.  The multiplication or the
// shift a temporary of the same block used only there.
static bool reg32(VReg *v) {  // (a narrow value kept extended to 32 bits)
  return v != NULL && !(v->flag & (VRF_CONST | VRF_FLONUM | VRF_FORCEMEMORY | VRF_VOLATILEREG)) &&
         v->vsize <= VRegSize4;
}

static void fuse_arm_ops(RegAlloc *ra, BBContainer *bbcon) {
  int vreg_count = ra->vregs->len;
  int *uses = malloc_or_die(sizeof(*uses) * vreg_count * 2);
  int *defs = uses + vreg_count;
  count_uses_defs(ra, bbcon, uses, defs);
  for (int i = 0; i < bbcon->len; ++i) {
    Vector *irs = ((BB*)bbcon->data[i])->irs;
    for (int j = 0; j < irs->len; ++j) {
      IR *ir = irs->data[j];
      int k;
      if (ir->kind == IR_BITAND && ir->bfx.shift == 0 && reg32(ir->dst) && reg32(ir->opr1) && (ir->opr2->flag & VRF_CONST) && !(ir->opr2->flag & VRF_FLONUM)) {
        VReg *t = ir->opr1;
        int64_t m = ir->opr2->fixnum;
        IR *sh;
        if (m <= 0 || m > 0xffffffffLL || (m & (m + 1)) != 0 || uses[t->virt] != 1 ||
            defs[t->virt] != 1 || (t->flag & VRF_PARAM) || (sh = def_in_bb(irs, j, t, &k)) == NULL ||
            sh->kind != IR_RSHIFT || !reg32(sh->opr1) || !(sh->opr2->flag & VRF_CONST) ||
            sh->opr2->fixnum < 1 || sh->opr2->fixnum > 31 || defined_between(irs, k, j, sh->opr1))
          continue;
        int w = 0;
        for (int64_t x = m; x != 0; x >>= 1)
          ++w;
        if (sh->opr2->fixnum + w > 32)
          continue;
        ir->opr1 = sh->opr1;
        ir->bfx.shift = (int)sh->opr2->fixnum;
        vec_remove_at(irs, k);
        --j;
        continue;
      }
      if (ir->kind == IR_CAST && (ir->dst->flag & VRF_FLONUM) && ir->dst->vsize == VRegSize4 &&
          plain_reg(ir->opr1) && uses[ir->opr1->virt] == 1 && defs[ir->opr1->virt] == 1 &&
          !(ir->opr1->flag & VRF_PARAM)) {
        // (float)(int)f: the int not out of the FPU
        VReg *t = ir->opr1, *f;
        IR *cv = def_in_bb(irs, j, t, &k);
        if (cv != NULL && cv->kind == IR_CAST && ((f = cv->opr1)->flag & VRF_FLONUM) &&
            f->vsize == VRegSize4 && !(f->flag & (VRF_CONST | VRF_FORCEMEMORY)) &&
            !defined_between(irs, k, j, f)) {
          ir->opr1 = f;
          ir->cast.trunc = (t->flag & VRF_UNSIGNED) ? 'u' : 's';
          vec_remove_at(irs, k);
          --j;
          continue;
        }
      }
      if (ir->kind == IR_NEG && plain_reg(ir->dst) && reg32(ir->opr1) &&
          uses[ir->opr1->virt] == 1 && defs[ir->opr1->virt] == 1 && !(ir->opr1->flag & VRF_PARAM)) {
        // -(x & 1): sbfx
        IR *a = def_in_bb(irs, j, ir->opr1, &k);
        VReg *x;
        if (a != NULL && a->kind == IR_BITAND && a->bfx.shift == 0 && a->bfx.sh_kind == 0 &&
            (a->opr2->flag & VRF_CONST) && !(a->opr2->flag & VRF_FLONUM) && a->opr2->fixnum == 1 &&
            reg32(x = a->opr1) && !defined_between(irs, k, j, x)) {
          ir->opr1 = x;
          ir->bfx.sbfx = true;
          vec_remove_at(irs, k);
          --j;
          continue;
        }
      }
      // A shifted operand: t = x << / >> n; d = y OP t -> OP d, y, x, lsl / lsr / asr #n
      if ((ir->kind == IR_ADD || ir->kind == IR_SUB || ir->kind == IR_BITAND ||
           ir->kind == IR_BITOR || ir->kind == IR_BITXOR) && ir->additional_operands == NULL &&
          ir->bfx.shift == 0 && ir->bfx.sh_kind == 0 && reg32(ir->dst) && ir->opr2 != NULL &&
          !((ir->dst->flag | ir->opr1->flag | ir->opr2->flag) & VRF_FLONUM) &&
          (ir->dst->vsize == VRegSize4 || ir->kind != IR_ADD)) {
        bool done = false;
        for (int side = ir->kind == IR_SUB ? 1 : 0; side < 2 && !done; ++side) {
          VReg *t = side == 0 ? ir->opr1 : ir->opr2, *y = side == 0 ? ir->opr2 : ir->opr1;
          IR *sh;
          if (!reg32(t) || !reg32(y) || uses[t->virt] != 1 || defs[t->virt] != 1 ||
              (t->flag & VRF_PARAM) || (sh = def_in_bb(irs, j, t, &k)) == NULL ||
              (sh->kind != IR_LSHIFT && sh->kind != IR_RSHIFT) || !reg32(sh->opr1) ||
              !(sh->opr2->flag & VRF_CONST) || sh->opr2->fixnum < 1 || sh->opr2->fixnum > 31 ||
              sh->dst->vsize != VRegSize4 || defined_between(irs, k, j, sh->opr1))
            continue;
          ir->opr1 = y;
          ir->opr2 = sh->opr1;
          ir->bfx.sh_kind = sh->kind == IR_LSHIFT ? 'l' : (sh->dst->flag & VRF_UNSIGNED) ? 'r' : 'a';
          ir->bfx.sh_amount = (int)sh->opr2->fixnum;
          vec_remove_at(irs, k);
          --j;
          done = true;
        }
        if (done)
          continue;
      }
      if ((ir->kind == IR_ADD || ir->kind == IR_SUB) && ir->additional_operands == NULL &&
          ir->bfx.sh_kind == 0 && plain_reg(ir->dst) && ir->opr2 != NULL) {
        for (int side = ir->kind == IR_ADD ? 0 : 1; side < 2; ++side) {
          VReg *t = side == 0 ? ir->opr1 : ir->opr2, *x = side == 0 ? ir->opr2 : ir->opr1;
          IR *mul;
          if (!plain_reg(t) || !reg32(x) || uses[t->virt] != 1 || defs[t->virt] != 1 ||
              (t->flag & VRF_PARAM) || (mul = def_in_bb(irs, j, t, &k)) == NULL ||
              mul->kind != IR_MUL || !reg32(mul->opr1) || !reg32(mul->opr2) ||
              defined_between(irs, k, j, mul->opr1) || defined_between(irs, k, j, mul->opr2))
            continue;
          ir->opr1 = x;
          ir->opr2 = NULL;
          ir->additional_operands = new_vector();
          vec_push(ir->additional_operands, mul->opr1);
          vec_push(ir->additional_operands, mul->opr2);
          vec_remove_at(irs, k);
          --j;
          break;
        }
      }
    }
  }
  free(uses);
}

void tweak_irs(FuncBackend *fnbe) {
  BBContainer *bbcon = fnbe->bbcon;
  RegAlloc *ra = fnbe->ra;
  fuse_indexed_access(ra, bbcon);  // FPC
  for (int i = 0; i < bbcon->len; ++i) {
    BB *bb = bbcon->data[i];
    Vector *irs = bb->irs;
    for (int j = 0; j < irs->len; ++j) {
      IR *ir = irs->data[j];

#ifndef __NO_FLONUM
      // Floating-point constants: loaded from .rodata.
      VReg **operands[] = {&ir->opr1, &ir->opr2};
      for (int k = 0; k < 2; ++k) {
        VReg **pp = operands[k], *opr = *pp;
        if (opr != NULL && (opr->flag & (VRF_FLONUM | VRF_CONST)) == (VRF_FLONUM | VRF_CONST))
          j = insert_const_float(pp, irs, j);
      }
#endif

      // Arithmetic under 4 bytes: done in 32 bits, then cast.
      {
        VReg *dst = ir->dst;
        if (dst != NULL && !(dst->flag & VRF_FLONUM) && dst->vsize < VRegSize4 &&
            !(ir->kind == IR_ADD && ir->bfx.nowrap)) {  // (FPC: a counter that fits)
          switch (ir->kind) {
          default: break;
          case IR_ADD: case IR_SUB: case IR_MUL: case IR_LSHIFT:
          case IR_NEG: case IR_BITNOT:
            {
              VReg *tmp = reg_alloc_spawn(ra, VRegSize4, dst->flag & VRF_MASK);
              IR *cast = new_ir_bop_raw(IR_CAST, dst, tmp, NULL);
              vec_insert(irs, j + 1, cast);
              ir->dst = tmp;
            }
            break;
          }
        }
      }

      bool wide = ir->dst != NULL && is_i64(ir->dst);
      switch (ir->kind) {
      case IR_LOAD:
        if (ir->opr1->flag & VRF_CONST)
          insert_tmp_mov(&ir->opr1, irs, j++);
        break;
      case IR_STORE:
        if (ir->opr2->flag & VRF_CONST)
          insert_tmp_mov(&ir->opr2, irs, j++);
        if ((ir->opr1->flag & VRF_CONST) && ir->opr1->vsize == VRegSize8)
          insert_tmp_mov(&ir->opr1, irs, j++);
        break;
      case IR_ADD: case IR_SUB: case IR_MUL: case IR_BITAND: case IR_BITOR: case IR_BITXOR:
        if (wide) {
          if (ir->opr1->flag & VRF_CONST)
            insert_tmp_mov(&ir->opr1, irs, j++);
          if (ir->opr2->flag & VRF_CONST)
            insert_tmp_mov(&ir->opr2, irs, j++);
        } else if ((ir->opr1->flag & VRF_CONST) && (ir->kind == IR_ADD || ir->kind == IR_MUL ||
                                                    ir->kind == IR_BITAND ||
                                                    ir->kind == IR_BITOR ||
                                                    ir->kind == IR_BITXOR)) {
          swap_opr12(ir);
        }
        if (!wide && ir->kind == IR_MUL && (ir->opr2->flag & VRF_CONST))
          insert_tmp_mov(&ir->opr2, irs, j++);
        break;
      case IR_DIV: case IR_MOD:
        if (ir->opr1->flag & VRF_CONST)
          insert_tmp_mov(&ir->opr1, irs, j++);
        if (ir->opr2->flag & VRF_CONST)
          insert_tmp_mov(&ir->opr2, irs, j++);
        break;
      case IR_LSHIFT: case IR_RSHIFT:
        if (ir->opr1->flag & VRF_CONST)
          insert_tmp_mov(&ir->opr1, irs, j++);
        break;
      case IR_NEG: case IR_BITNOT: case IR_CAST:
        if (ir->opr1->flag & VRF_CONST)
          insert_tmp_mov(&ir->opr1, irs, j++);
        break;
      case IR_COND: case IR_JMP:
        if (ir->opr1 != NULL && (ir->opr1->flag & VRF_CONST)) {
          if (ir->opr2 != NULL && !(ir->opr2->flag & VRF_CONST)) {
            swap_opr12(ir);
            if (ir->kind == IR_COND)
              ir->cond.kind = swap_cond(ir->cond.kind);
            else
              ir->jmp.cond = swap_cond(ir->jmp.cond);
          } else {
            insert_tmp_mov(&ir->opr1, irs, j++);
          }
        }
        if (ir->opr2 != NULL && (ir->opr2->flag & VRF_CONST) && ir->opr1 != NULL &&
            ir->opr1->vsize == VRegSize8)
          insert_tmp_mov(&ir->opr2, irs, j++);
        break;
      case IR_TJMP:
        assert(!(ir->opr1->flag & VRF_CONST));
        break;
      case IR_CALL:
        if (ir->opr1 != NULL && (ir->opr1->flag & VRF_CONST))
          insert_tmp_mov(&ir->opr1, irs, j++);
        break;
      default: break;
      }
    }
  }
  fuse_arm_ops(ra, bbcon);  // FPC
  mark_fitting_casts(ra, bbcon);  // FPC
}
