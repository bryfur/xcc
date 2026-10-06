// FPC: ARM32 assembler: Thumb-2 encoding (ARMv8-M Mainline, FPv5-SP).
//
// A 16-bit encoding is used where it means the same as the 32-bit one:
// inside an IT block, or for the forms that never set the flags (moves
// between registers, loads and stores, sp adjustments, cmp, push/pop).
// Branches are encoded short here and widened by ir_asm_arm32.c when
// their targets are too far.

#include "../../../config.h"
#include "asm_code.h"

#include <assert.h>
#include <string.h>  // memcpy

#include "inst.h"
#include "parse_asm.h"
#include "table.h"
#include "util.h"

#define SP  13
#define LR  14
#define PC  15

void make_code16(Inst *inst, Code *code, unsigned short *buf, int len) {
  assert(code->len + len <= (int)sizeof(code->buf));
  code->inst = inst;
  memcpy(code->buf + code->len, buf, len);
  code->len += len;
}

void make_code32(Inst *inst, Code *code, unsigned int *buf, int len) {
  assert(code->len + len <= (int)sizeof(code->buf));
  code->inst = inst;
  memcpy(code->buf + code->len, buf, len);
  code->len += len;
}

static void put16(Code *code, uint32_t hw) {
  assert(code->len + 2 <= (int)sizeof(code->buf));
  int n = code->len;
  code->buf[n] = hw;
  code->buf[n + 1] = hw >> 8;
  code->len = n + 2;
}

static void put32(Code *code, uint32_t hw1, uint32_t hw2) {
  put16(code, hw1);
  put16(code, hw2);
}

// IT block state: the instructions still inside one.
static int s_it_remaining;

static bool asm_error(ParseInfo *parser, const char *msg) {
  parse_error(parser, msg);
  return false;
}

static inline bool low(int r) { return r >= 0 && r < 8; }

// ThumbExpandImm encoding of x (12 bits: i:imm3:imm8), or -1
int arm32_modimm(uint32_t x) {
  if (x <= 0xff)
    return x;
  uint32_t b = x & 0xff, b2 = (x >> 8) & 0xff;
  if (x == (b | (b << 16)))
    return 0x100 | b;
  if (x == ((b2 << 8) | (b2 << 24)))
    return 0x200 | b2;
  if (x == (b | (b << 8) | (b << 16) | (b << 24)))
    return 0x300 | b;
  for (int rot = 8; rot < 32; ++rot) {
    uint32_t v = (x << rot) | (x >> (32 - rot));
    if (v >= 0x80 && v <= 0xff)
      return (rot << 7) | (v & 0x7f);
  }
  return -1;
}

static void put_dp_imm(Code *code, int op, int s, int rn, int rd, int imm12) {
  uint32_t i = (imm12 >> 11) & 1, imm3 = (imm12 >> 8) & 7, imm8 = imm12 & 0xff;
  put32(code, 0xf000 | (i << 10) | (op << 5) | (s << 4) | rn, (imm3 << 12) | (rd << 8) | imm8);
}

static void put_dp_reg(Code *code, int op, int s, int rn, int rd, int rm, int shtype, int amount) {
  uint32_t imm3 = (amount >> 2) & 7, imm2 = amount & 3;
  put32(code, 0xea00 | (op << 5) | (s << 4) | rn,
        (imm3 << 12) | (rd << 8) | (imm2 << 6) | (shtype << 4) | rm);
}

// The 4-bit data processing opcodes (modified immediate / shifted register)
enum {
  DP_AND = 0, DP_BIC = 1, DP_ORR = 2, DP_ORN = 3, DP_EOR = 4,
  DP_ADD = 8, DP_ADC = 10, DP_SBC = 11, DP_SUB = 13, DP_RSB = 14,
};

static int dp_opcode(enum Opcode op) {
  switch (op) {
  case AND: case TST: return DP_AND;
  case BIC: return DP_BIC;
  case ORR: case MOV: return DP_ORR;
  case ORN: case MVN: return DP_ORN;
  case EOR: case TEQ: return DP_EOR;
  case ADD: case CMN: return DP_ADD;
  case ADC: return DP_ADC;
  case SBC: return DP_SBC;
  case SUB: case CMP: return DP_SUB;
  case RSB: return DP_RSB;
  default: return -1;
  }
}

// Operand 2 (register, shifted register, immediate) of a data processing
// instruction: rd = rn op opr2 (rn 15 for mov/mvn, rd 15 for tst/cmp...).
static bool asm_dp(Inst *inst, ParseInfo *parser, Code *code, int rd, int rn, Operand *opr2,
                   int s) {
  int op = dp_opcode(inst->op);
  assert(op >= 0);
  switch (opr2->type) {
  case IMMEDIATE:
    {
      int imm12 = arm32_modimm((uint32_t)opr2->immediate);
      if (imm12 < 0) {
        // add/sub: a plain 12-bit immediate (addw/subw), no flags.
        uint32_t u = (uint32_t)opr2->immediate;
        if (!s && (inst->op == ADD || inst->op == SUB) && u <= 4095) {
          uint32_t i = (u >> 11) & 1, imm3 = (u >> 8) & 7, imm8 = u & 0xff;
          put32(code, 0xf200 | (i << 10) | ((inst->op == SUB ? 0x5 : 0x0) << 5) | rn,
                (imm3 << 12) | (rd << 8) | imm8);
          return true;
        }
        return asm_error(parser, "immediate out of range");
      }
      put_dp_imm(code, op, s, rn, rd, imm12);
      return true;
    }
  case REG:
    put_dp_reg(code, op, s, rn, rd, opr2->reg, 0, 0);
    return true;
  case SHIFTREG:
    put_dp_reg(code, op, s, rn, rd, opr2->shiftreg.reg, opr2->shiftreg.shift,
               opr2->shiftreg.amount & 31);
    return true;
  default:
    return asm_error(parser, "illegal operand");
  }
}

static bool asm_data_processing(Inst *inst, ParseInfo *parser, Code *code, bool in_it) {
  Operand *o = inst->opr;
  int s = inst->setflags ? 1 : 0;
  switch (inst->op) {
  case MOV: case MVN:
    {
      if (o[0].type != REG)
        return asm_error(parser, "register expected");
      int rd = o[0].reg;
      if (inst->op == MOV && o[1].type == REG && !s && inst->width != 4) {
        put16(code, 0x4600 | ((rd & 8) << 4) | (o[1].reg << 3) | (rd & 7));  // MOV (register) T1
        return true;
      }
      if (inst->op == MOV && o[1].type == IMMEDIATE && low(rd) && inst->width != 4 &&
          (uint64_t)o[1].immediate <= 0xff && (in_it ? !s : s)) {
        put16(code, 0x2000 | (rd << 8) | (o[1].immediate & 0xff));  // MOVS T1 / MOV in IT
        return true;
      }
      if (inst->op == MOV && o[1].type == IMMEDIATE && arm32_modimm((uint32_t)o[1].immediate) < 0 &&
          !s && (uint64_t)o[1].immediate <= 0xffff) {
        // mov r, #imm16: movw
        uint32_t u = (uint32_t)o[1].immediate;
        put32(code, 0xf240 | (((u >> 11) & 1) << 10) | (u >> 12), (((u >> 8) & 7) << 12) | (rd << 8) | (u & 0xff));
        return true;
      }
      return asm_dp(inst, parser, code, rd, 15, &o[1], s);
    }
  case CMP: case CMN: case TST: case TEQ:
    {
      int rn = o[0].reg;
      if (inst->op == CMP && inst->width != 4) {
        if (o[1].type == IMMEDIATE && low(rn) && (uint64_t)o[1].immediate <= 0xff) {
          put16(code, 0x2800 | (rn << 8) | (o[1].immediate & 0xff));
          return true;
        }
        if (o[1].type == REG) {
          int rm = o[1].reg;
          if (low(rn) && low(rm))
            put16(code, 0x4280 | (rm << 3) | rn);
          else
            put16(code, 0x4500 | ((rn & 8) << 4) | (rm << 3) | (rn & 7));
          return true;
        }
      }
      return asm_dp(inst, parser, code, 15, rn, &o[1], 1);
    }
  default:
    break;
  }

  // rd, rn, opr2 (or rd, opr2 meaning rd, rd, opr2)
  int rd = o[0].reg;
  int rn;
  Operand *opr2;
  if (o[2].type == NOOPERAND) {
    rn = rd;
    opr2 = &o[1];
  } else {
    rn = o[1].reg;
    opr2 = &o[2];
  }
  if (o[0].type != REG || (o[2].type != NOOPERAND && o[1].type != REG))
    return asm_error(parser, "register expected");

  // sp adjustments, 16-bit
  if ((inst->op == ADD || inst->op == SUB) && rd == SP && rn == SP && opr2->type == IMMEDIATE &&
      !s && opr2->immediate >= 0 && opr2->immediate <= 508 && (opr2->immediate & 3) == 0 &&
      inst->width != 4) {
    put16(code, (inst->op == ADD ? 0xb000 : 0xb080) | (opr2->immediate >> 2));
    return true;
  }
  // add rdn, rm (no flags), 16-bit (either operand being rd)
  if (inst->op == ADD && !s && opr2->type == REG && (rd == rn || rd == opr2->reg) &&
      inst->width != 4 && opr2->reg != SP && rn != SP && rd != SP && rd != PC) {
    int rm = rd == rn ? opr2->reg : rn;
    put16(code, 0x4400 | ((rd & 8) << 4) | (rm << 3) | (rd & 7));
    return true;
  }
  return asm_dp(inst, parser, code, rd, rn, opr2, s);
}

static bool asm_shift(Inst *inst, ParseInfo *parser, Code *code) {
  Operand *o = inst->opr;
  int s = inst->setflags ? 1 : 0;
  int rd = o[0].reg;
  int rm = o[1].reg;
  Operand *amount = o[2].type != NOOPERAND ? &o[2] : &o[1];
  if (o[2].type == NOOPERAND)
    rm = rd;
  int type = inst->op == LSL ? 0 : inst->op == LSR ? 1 : inst->op == ASR ? 2 : 3;
  if (amount->type == IMMEDIATE) {
    // MOV with a shift (shifted register form, Rn = 15)
    int n = amount->immediate & 31;
    if (n == 0 && type == 0) {
      put_dp_reg(code, DP_ORR, s, 15, rd, rm, 0, 0);
      return true;
    }
    put_dp_reg(code, DP_ORR, s, 15, rd, rm, type, n);
    return true;
  }
  if (amount->type != REG)
    return asm_error(parser, "illegal shift");
  // Register shift: 11111010 0 type S Rn | 1111 Rd 0000 Rm
  put32(code, 0xfa00 | (type << 5) | (s << 4) | rm, 0xf000 | (rd << 8) | amount->reg);
  return true;
}

static bool asm_movwt(Inst *inst, ParseInfo *parser, Code *code) {
  Operand *o = inst->opr;
  int rd = o[0].reg;
  uint32_t u = 0;
  if (o[1].type == IMMEDIATE)
    u = (uint32_t)o[1].immediate & 0xffff;
  else if (o[1].type != LABEL16)
    return asm_error(parser, "immediate expected");
  // The label's part is put in by the relocation (ir_asm_arm32.c).
  uint32_t hw1 = (inst->op == MOVW ? 0xf240 : 0xf2c0) | (((u >> 11) & 1) << 10) | (u >> 12);
  uint32_t hw2 = (((u >> 8) & 7) << 12) | (rd << 8) | (u & 0xff);
  put32(code, hw1, hw2);
  return true;
}

static bool asm_multiply(Inst *inst, ParseInfo *parser, Code *code) {
  Operand *o = inst->opr;
  UNUSED(parser);
  switch (inst->op) {
  case MUL:
    {
      int rd = o[0].reg, rn = o[1].reg, rm = o[2].type == REG ? o[2].reg : o[0].reg;
      put32(code, 0xfb00 | rn, 0xf000 | (rd << 8) | rm);
    }
    return true;
  case MLA: case MLS:
    put32(code, 0xfb00 | o[1].reg,
          (o[3].reg << 12) | (o[0].reg << 8) | (inst->op == MLS ? 0x10 : 0) | o[2].reg);
    return true;
  case UMULL: case SMULL:
    put32(code, (inst->op == UMULL ? 0xfba0 : 0xfb80) | o[2].reg,
          (o[0].reg << 12) | (o[1].reg << 8) | o[3].reg);
    return true;
  case SDIV: case UDIV:
    put32(code, (inst->op == SDIV ? 0xfb90 : 0xfbb0) | o[1].reg, 0xf0f0 | (o[0].reg << 8) | o[2].reg);
    return true;
  default:
    assert(false);
    return false;
  }
}

static bool asm_misc(Inst *inst, ParseInfo *parser, Code *code) {
  Operand *o = inst->opr;
  UNUSED(parser);
  int rd = o[0].reg, rm = o[1].reg;
  switch (inst->op) {
  case SXTB: put32(code, 0xfa4f, 0xf080 | (rd << 8) | rm); return true;
  case SXTH: put32(code, 0xfa0f, 0xf080 | (rd << 8) | rm); return true;
  case UXTB: put32(code, 0xfa5f, 0xf080 | (rd << 8) | rm); return true;
  case UXTH: put32(code, 0xfa1f, 0xf080 | (rd << 8) | rm); return true;
  case CLZ:  put32(code, 0xfab0 | rm, 0xf080 | (rd << 8) | rm); return true;
  case RBIT: put32(code, 0xfa90 | rm, 0xf0a0 | (rd << 8) | rm); return true;
  default: assert(false); return false;
  }
}

// Loads and stores of a core register.
static bool asm_ldst(Inst *inst, ParseInfo *parser, Code *code) {
  Operand *o = inst->opr;
  if (o[0].type != REG || o[1].type != INDIRECT)
    return asm_error(parser, "illegal operand");
  int rt = o[0].reg;
  int rn = o[1].indirect.base;
  int64_t ofs = o[1].indirect.offset;
  int prepost = o[1].indirect.prepost;

  // (T3 op: imm12; T4/register base op)
  int t3, t4;
  int shift16 = -1, op16 = 0;  // 16-bit imm5 form: scale, opcode
  switch (inst->op) {
  case LDR:   t3 = 0xf8d0; t4 = 0xf850; shift16 = 2; op16 = 0x6800; break;
  case STR:   t3 = 0xf8c0; t4 = 0xf840; shift16 = 2; op16 = 0x6000; break;
  case LDRB:  t3 = 0xf890; t4 = 0xf810; shift16 = 0; op16 = 0x7800; break;
  case STRB:  t3 = 0xf880; t4 = 0xf800; shift16 = 0; op16 = 0x7000; break;
  case LDRH:  t3 = 0xf8b0; t4 = 0xf830; shift16 = 1; op16 = 0x8800; break;
  case STRH:  t3 = 0xf8a0; t4 = 0xf820; shift16 = 1; op16 = 0x8000; break;
  case LDRSB: t3 = 0xf990; t4 = 0xf910; break;
  case LDRSH: t3 = 0xf9b0; t4 = 0xf930; break;
  default: assert(false); return false;
  }

  if (o[1].indirect.index >= 0) {
    if (prepost != 0)
      return asm_error(parser, "illegal addressing");
    int rm = o[1].indirect.index;
    int sh = o[1].indirect.shift;
    if (sh < 0 || sh > 3)
      return asm_error(parser, "shift out of range");
    if (sh == 0 && low(rt) && low(rn) && low(rm) && inst->width != 4) {  // 16-bit T1
      static const struct { int op; uint16_t code; } k16[] = {
        {STR, 0x5000}, {STRH, 0x5200}, {STRB, 0x5400}, {LDRSB, 0x5600},
        {LDR, 0x5800}, {LDRH, 0x5a00}, {LDRB, 0x5c00}, {LDRSH, 0x5e00},
      };
      for (size_t i = 0; i < sizeof(k16) / sizeof(k16[0]); ++i) {
        if (k16[i].op == (int)inst->op) {
          put16(code, k16[i].code | (rm << 6) | (rn << 3) | rt);
          return true;
        }
      }
    }
    put32(code, t4 | rn, (rt << 12) | (sh << 4) | rm);
    return true;
  }

  if (prepost == 0) {
    // 16-bit: low registers, small scaled offset
    if (shift16 >= 0 && low(rt) && low(rn) && ofs >= 0 && (ofs & ((1 << shift16) - 1)) == 0 &&
        (ofs >> shift16) < 32 && inst->width != 4) {
      put16(code, op16 | ((ofs >> shift16) << 6) | (rn << 3) | rt);
      return true;
    }
    // 16-bit sp-relative word
    if ((inst->op == LDR || inst->op == STR) && rn == SP && low(rt) && ofs >= 0 && ofs <= 1020 &&
        (ofs & 3) == 0 && inst->width != 4) {
      put16(code, (inst->op == LDR ? 0x9800 : 0x9000) | (rt << 8) | (ofs >> 2));
      return true;
    }
    if (ofs >= 0 && ofs <= 4095) {
      put32(code, t3 | rn, (rt << 12) | (uint32_t)ofs);
      return true;
    }
    if (ofs < 0 && ofs >= -255) {
      put32(code, t4 | rn, (rt << 12) | 0xc00 | (uint32_t)(-ofs));  // P=1 U=0 W=0
      return true;
    }
    return asm_error(parser, "offset out of range");
  }
  // Pre/post-indexed (writeback): T4, P U W
  if (ofs < -255 || ofs > 255)
    return asm_error(parser, "offset out of range");
  uint32_t p = prepost == 1 ? 1 : 0;
  uint32_t u = ofs >= 0 ? 1 : 0;
  uint32_t imm8 = ofs >= 0 ? (uint32_t)ofs : (uint32_t)-ofs;
  put32(code, t4 | rn, (rt << 12) | 0x800 | (p << 10) | (u << 9) | (1 << 8) | imm8);
  return true;
}

static bool asm_ldstd(Inst *inst, ParseInfo *parser, Code *code) {
  Operand *o = inst->opr;
  if (o[0].type != REG || o[1].type != REG || o[2].type != INDIRECT || o[2].indirect.index >= 0)
    return asm_error(parser, "illegal operand");
  int64_t ofs = o[2].indirect.offset;
  if (ofs < -1020 || ofs > 1020 || (ofs & 3) != 0)
    return asm_error(parser, "offset out of range");
  int prepost = o[2].indirect.prepost;
  uint32_t p = prepost == 2 ? 0 : 1;
  uint32_t w = prepost != 0 ? 1 : 0;
  uint32_t u = ofs >= 0 ? 1 : 0;
  uint32_t imm8 = (ofs >= 0 ? ofs : -ofs) >> 2;
  uint32_t l = inst->op == LDRD ? 1 : 0;
  put32(code, 0xe840 | (p << 8) | (u << 7) | (w << 5) | (l << 4) | o[2].indirect.base,
        (o[0].reg << 12) | (o[1].reg << 8) | imm8);
  return true;
}

static int popcount16(uint32_t x) {
  int n = 0;
  for (; x != 0; x &= x - 1)
    ++n;
  return n;
}

static int lowest_bit(uint32_t x) {
  int n = 0;
  while (!(x & 1)) {
    x >>= 1;
    ++n;
  }
  return n;
}

static bool asm_pushpop(Inst *inst, ParseInfo *parser, Code *code) {
  Operand *o = inst->opr;
  if (o[0].type != REGLIST)
    return asm_error(parser, "register list expected");
  uint32_t mask = o[0].list.mask;
  bool push = inst->op == PUSH;
  uint32_t extra = push ? (1U << LR) : (1U << PC);
  if ((mask & ~(0xffU | extra)) == 0 && inst->width != 4) {
    put16(code, (push ? 0xb400 : 0xbc00) | ((mask & extra) ? 0x100 : 0) | (mask & 0xff));
    return true;
  }
  if (popcount16(mask) == 1) {
    int rt = lowest_bit(mask);
    if (push)
      put32(code, 0xf84d, (rt << 12) | 0xd04);  // str rt, [sp, #-4]!
    else
      put32(code, 0xf85d, (rt << 12) | 0xb04);  // ldr rt, [sp], #4
    return true;
  }
  if (push)
    put32(code, 0xe92d, mask & 0x5fff);
  else
    put32(code, 0xe8bd, mask & 0xdfff);
  return true;
}

static bool asm_ldmstm(Inst *inst, ParseInfo *parser, Code *code) {
  Operand *o = inst->opr;
  // ldmia rn!, {list} / stmia rn, {list}: the '!' was parsed as part of the
  // register operand?  Only "rn" or "rn!" forms: see parse (REG + list).
  if (o[0].type != REG || o[1].type != REGLIST)
    return asm_error(parser, "illegal operand");
  int rn = o[0].reg;
  uint32_t w = inst->writeback ? 1 : 0;
  if (inst->op == STM)
    put32(code, 0xe880 | (w << 5) | rn, o[1].list.mask & 0x5fff);
  else
    put32(code, 0xe890 | (w << 5) | rn, o[1].list.mask & 0xdfff);
  return true;
}

// A branch, short; the offset is patched (and the branch widened) later.
static bool asm_branch(Inst *inst, ParseInfo *parser, Code *code) {
  Operand *o = inst->opr;
  switch (inst->op) {
  case B:
    if (o[0].type != DIRECT)
      return asm_error(parser, "label expected");
    if (inst->cond == NOCOND || inst->cond == AL)
      put16(code, 0xe000);
    else
      put16(code, 0xd000 | (inst->cond << 8));
    return true;
  case BL:
    if (o[0].type != DIRECT)
      return asm_error(parser, "label expected");
    put32(code, 0xf000, 0xd000);
    return true;
  case BX:
    put16(code, 0x4700 | (o[0].reg << 3));
    return true;
  case BLX:
    if (o[0].type != REG)
      return asm_error(parser, "register expected");
    put16(code, 0x4780 | (o[0].reg << 3));
    return true;
  case CBZ: case CBNZ:
    put16(code, (inst->op == CBZ ? 0xb100 : 0xb900) | o[0].reg);
    return true;
  case SVC:
    put16(code, 0xdf00 | (o[0].immediate & 0xff));
    return true;
  case NOP:
    put16(code, 0xbf00);
    return true;
  default:
    assert(false);
    return false;
  }
}

static bool asm_it(Inst *inst, ParseInfo *parser, Code *code) {
  if (inst->opr[0].type != DIRECT)
    return asm_error(parser, "condition expected");
  // The condition was parsed as a label expression: its name.
  const Expr *e = inst->opr[0].direct.expr;
  if (e->kind != EX_LABEL || e->label.name->bytes != 2)
    return asm_error(parser, "condition expected");
  static const char *kConds[] = {
    "eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al",
  };
  int cond = -1;
  for (int i = 0; i < (int)ARRAY_SIZE(kConds); ++i)
    if (strncmp(e->label.name->chars, kConds[i], 2) == 0)
      cond = i;
  if (strncmp(e->label.name->chars, "cs", 2) == 0) cond = HS;
  if (strncmp(e->label.name->chars, "cc", 2) == 0) cond = LO;
  if (cond < 0)
    return asm_error(parser, "condition expected");
  int n = (int)strnlen(inst->itpattern, 3);
  // mask: for each following instruction a bit (cond[0] for then, !cond[0]
  // for else), then a 1, then zeros
  uint32_t c0 = cond & 1;
  uint32_t mask = 0;
  for (int i = 0; i < n; ++i) {
    uint32_t bit = inst->itpattern[i] == 't' ? c0 : !c0;
    mask |= bit << (3 - i);
  }
  mask |= 1U << (3 - n);
  put16(code, 0xbf00 | (cond << 4) | mask);
  s_it_remaining = n + 1;
  return true;
}

// VFP

static void vreg_bits(Operand *o, uint32_t *v, uint32_t *b) {
  if (o->type == SREG) {
    *v = o->reg >> 1;
    *b = o->reg & 1;
  } else {
    *v = o->reg & 15;
    *b = (o->reg >> 4) & 1;
  }
}

static bool asm_vfp(Inst *inst, ParseInfo *parser, Code *code) {
  Operand *o = inst->opr;
  uint32_t vd, d, vn, n, vm, m;
  switch (inst->op) {
  case VLDR: case VSTR:
    {
      if ((o[0].type != SREG && o[0].type != DREG) || o[1].type != INDIRECT ||
          o[1].indirect.index >= 0 || o[1].indirect.prepost != 0)
        return asm_error(parser, "illegal operand");
      int64_t ofs = o[1].indirect.offset;
      if (ofs < -1020 || ofs > 1020 || (ofs & 3) != 0)
        return asm_error(parser, "offset out of range");
      vreg_bits(&o[0], &vd, &d);
      uint32_t u = ofs >= 0 ? 1 : 0;
      uint32_t sz = o[0].type == DREG ? 1 : 0;
      put32(code, 0xed00 | (u << 7) | (d << 6) | (inst->op == VLDR ? 0x10 : 0) | o[1].indirect.base,
            (vd << 12) | 0xa00 | (sz << 8) | ((ofs >= 0 ? ofs : -ofs) >> 2));
      return true;
    }
  case VMOV:
    if (o[0].type == REG && o[1].type == SREG) {  // vmov r, s
      vreg_bits(&o[1], &vn, &n);
      put32(code, 0xee10 | vn, (o[0].reg << 12) | 0xa10 | (n << 7));
      return true;
    }
    if (o[0].type == SREG && o[1].type == REG) {  // vmov s, r
      vreg_bits(&o[0], &vn, &n);
      put32(code, 0xee00 | vn, (o[1].reg << 12) | 0xa10 | (n << 7));
      return true;
    }
    if (o[0].type == REG && o[1].type == REG && o[2].type == DREG) {  // vmov r, r, d
      vreg_bits(&o[2], &vm, &m);
      put32(code, 0xec50 | o[1].reg, (o[0].reg << 12) | 0xb10 | (m << 5) | vm);
      return true;
    }
    if (o[0].type == DREG && o[1].type == REG && o[2].type == REG) {  // vmov d, r, r
      vreg_bits(&o[0], &vm, &m);
      put32(code, 0xec40 | o[2].reg, (o[1].reg << 12) | 0xb10 | (m << 5) | vm);
      return true;
    }
    if (o[0].type == SREG && o[1].type == SREG) {  // vmov.f32 s, s
      vreg_bits(&o[0], &vd, &d);
      vreg_bits(&o[1], &vm, &m);
      put32(code, 0xeeb0 | (d << 6), (vd << 12) | 0xa40 | (m << 5) | vm);
      return true;
    }
    return asm_error(parser, "illegal vmov");
  case VADD: case VSUB: case VMUL: case VDIV:
    {
      if (o[0].type != SREG || o[1].type != SREG || o[2].type != SREG)
        return asm_error(parser, "single precision registers expected");
      vreg_bits(&o[0], &vd, &d);
      vreg_bits(&o[1], &vn, &n);
      vreg_bits(&o[2], &vm, &m);
      uint32_t hw1, op6 = 0;
      switch (inst->op) {
      default:
      case VADD: hw1 = 0xee30; break;
      case VSUB: hw1 = 0xee30; op6 = 1; break;
      case VMUL: hw1 = 0xee20; break;
      case VDIV: hw1 = 0xee80; break;
      }
      put32(code, hw1 | (d << 6) | vn, (vd << 12) | 0xa00 | (n << 7) | (op6 << 6) | (m << 5) | vm);
      return true;
    }
  case VNEG: case VABS: case VSQRT:
    {
      vreg_bits(&o[0], &vd, &d);
      vreg_bits(&o[1], &vm, &m);
      uint32_t hw1 = inst->op == VABS ? 0xeeb0 : 0xeeb1;
      uint32_t op = inst->op == VNEG ? 0x40 : 0xc0;
      put32(code, hw1 | (d << 6), (vd << 12) | 0xa00 | op | (m << 5) | vm);
      return true;
    }
  case VCMP: case VCMPE:
    {
      vreg_bits(&o[0], &vd, &d);
      uint32_t e = inst->op == VCMPE ? 0x80 : 0;
      if (o[1].type == IMMEDIATE) {  // vcmp s, #0
        put32(code, 0xeeb5 | (d << 6), (vd << 12) | 0xa40 | e);
      } else {
        vreg_bits(&o[1], &vm, &m);
        put32(code, 0xeeb4 | (d << 6), (vd << 12) | 0xa40 | e | (m << 5) | vm);
      }
      return true;
    }
  case VMRS:
    put32(code, 0xeef1, 0xfa10);  // vmrs APSR_nzcv, fpscr
    return true;
  case VCVT:
    {
      vreg_bits(&o[0], &vd, &d);
      vreg_bits(&o[1], &vm, &m);
      enum VType to = inst->vt[0], from = inst->vt[1];
      uint32_t opc2, op;
      if (to == VT_F32 && (from == VT_S32 || from == VT_U32)) {
        opc2 = 0;
        op = from == VT_S32 ? 1 : 0;
      } else if ((to == VT_S32 || to == VT_U32) && from == VT_F32) {
        opc2 = to == VT_S32 ? 5 : 4;
        op = 1;  // round towards zero
      } else {
        return asm_error(parser, "vcvt: f32<->s32/u32 only");
      }
      put32(code, 0xeeb8 | (d << 6) | opc2, (vd << 12) | 0xa40 | (op << 7) | (m << 5) | vm);
      return true;
    }
  case VPUSH: case VPOP: case VLDM: case VSTM:
    {
      Operand *list = inst->op == VPUSH || inst->op == VPOP ? &o[0] : &o[1];
      if (list->type != FREGLIST)
        return asm_error(parser, "register list expected");
      Operand first = {.type = list->list.dbl ? DREG : SREG, .reg = (int)list->list.mask};
      vreg_bits(&first, &vd, &d);
      uint32_t sz = list->list.dbl ? 1 : 0;
      uint32_t imm8 = list->list.count * (list->list.dbl ? 2 : 1);
      uint32_t hw2 = (vd << 12) | 0xa00 | (sz << 8) | imm8;
      switch (inst->op) {
      default:
      case VPUSH: put32(code, 0xed2d | (d << 6), hw2); break;  // vstmdb sp!
      case VPOP:  put32(code, 0xecbd | (d << 6), hw2); break;  // vldmia sp!
      case VSTM:
      case VLDM:
        {
          uint32_t w = inst->writeback ? 1 : 0;
          uint32_t l = inst->op == VLDM ? 1 : 0;
          put32(code, 0xec80 | (d << 6) | (w << 5) | (l << 4) | o[0].reg, hw2);
        }
        break;
      }
      return true;
    }
  default:
    assert(false);
    return false;
  }
}

void assemble_inst(Inst *inst, ParseInfo *parser, Code *code) {
  code->flag = 0;
  code->len = 0;
  code->inst = inst;

  bool in_it = s_it_remaining > 0;
  if (inst->op != IT && s_it_remaining > 0)
    --s_it_remaining;
  if (inst->cond != NOCOND && inst->op != B && !in_it) {
    asm_error(parser, "conditional instruction outside an IT block");
    return;
  }

  bool ok;
  switch (inst->op) {
  case AND: case BIC: case ORR: case ORN: case EOR: case ADD: case ADC: case SBC: case SUB:
  case RSB: case MOV: case MVN: case CMP: case CMN: case TST: case TEQ:
    ok = asm_data_processing(inst, parser, code, in_it);
    break;
  case LSL: case LSR: case ASR: case ROR:
    ok = asm_shift(inst, parser, code);
    break;
  case MOVW: case MOVT:
    ok = asm_movwt(inst, parser, code);
    break;
  case MUL: case MLA: case MLS: case UMULL: case SMULL: case SDIV: case UDIV:
    ok = asm_multiply(inst, parser, code);
    break;
  case SXTB: case SXTH: case UXTB: case UXTH: case CLZ: case RBIT:
    ok = asm_misc(inst, parser, code);
    break;
  case LDR: case LDRB: case LDRH: case LDRSB: case LDRSH: case STR: case STRB: case STRH:
    ok = asm_ldst(inst, parser, code);
    break;
  case LDRD: case STRD:
    ok = asm_ldstd(inst, parser, code);
    break;
  case PUSH: case POP:
    ok = asm_pushpop(inst, parser, code);
    break;
  case LDM: case STM:
    ok = asm_ldmstm(inst, parser, code);
    break;
  case B: case BL: case BX: case BLX: case CBZ: case CBNZ: case SVC: case NOP:
    ok = asm_branch(inst, parser, code);
    break;
  case IT:
    ok = asm_it(inst, parser, code);
    break;
  default:
    ok = asm_vfp(inst, parser, code);
    break;
  }
  UNUSED(ok);
}
