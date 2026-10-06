// FPC: ARM32 (Thumb-2, ARMv8-M Mainline with the FPv5 single precision FPU)
// instructions, as xcc's assembler parses and encodes them.

#pragma once

#include <stdbool.h>
#include <stdint.h>  // int64_t

typedef struct Expr Expr;

enum Opcode {
  NOOP,
  // Data processing (an operand 2 of a register, a shifted register or an
  // immediate)
  AND, BIC, ORR, ORN, EOR, ADD, ADC, SBC, SUB, RSB,
  MOV, MVN, CMP, CMN, TST, TEQ,
  LSL, LSR, ASR, ROR,
  MOVW, MOVT,
  MUL, MLA, MLS, UMULL, SMULL, SDIV, UDIV,
  SXTB, SXTH, UXTB, UXTH, CLZ, RBIT,
  // Loads and stores
  LDR, LDRB, LDRH, LDRSB, LDRSH, STR, STRB, STRH, LDRD, STRD,
  PUSH, POP, LDM, STM,
  // Branches
  B, BL, BX, BLX, CBZ, CBNZ, IT, SVC, NOP,
  // VFP
  VLDR, VSTR, VMOV, VADD, VSUB, VMUL, VDIV, VNEG, VABS, VSQRT, VCMP, VCMPE,
  VMRS, VCVT, VPUSH, VPOP, VLDM, VSTM,
};

enum CondType {
  NOCOND = -1,
  EQ, NE, HS, LO, MI, PL, VS, VC,
  HI, LS, GE, LT, GT, LE, AL,
};

enum OperandType {
  NOOPERAND,
  REG,        // r0-r15
  SREG,       // s0-s31
  DREG,       // d0-d15
  IMMEDIATE,  // #nn
  DIRECT,     // label + nn (a branch target, data)
  LABEL16,    // #:lower16:label+nn, #:upper16:label+nn
  SHIFTREG,   // r, lsl #n (operand 2)
  INDIRECT,   // [r], [r, #nn], [r, #nn]!, [r], #nn, [r, r], [r, r, lsl #n]
  REGLIST,    // {r0, r4-r7, lr}
  FREGLIST,   // {s16-s31}, {d8-d15}
  SPECIAL,    // APSR_nzcv, fpscr
  ITMASK,     // the conditions of an IT block (x, t, e: in the mnemonic)
};

enum ShiftType {
  SH_LSL, SH_LSR, SH_ASR, SH_ROR,
};

typedef struct Operand {
  enum OperandType type;
  union {
    int reg;               // REG, SREG, DREG
    int64_t immediate;     // IMMEDIATE
    struct {
      Expr *expr;
      int upper;           // LABEL16: 1 for :upper16:
    } direct;              // DIRECT, LABEL16
    struct {
      int reg;
      enum ShiftType shift;
      int amount;
    } shiftreg;            // SHIFTREG
    struct {
      int base;
      int index;           // -1: an immediate offset
      int shift;           // index's lsl
      int64_t offset;
      int prepost;         // 0: offset, 1: pre-indexed (!), 2: post-indexed
    } indirect;            // INDIRECT
    struct {
      uint32_t mask;       // REGLIST: bit n = rn; FREGLIST: the first register
      int count;           // FREGLIST: number of registers
      bool dbl;            // FREGLIST: d registers
    } list;
  };
} Operand;

// VFP data types (the mnemonic's .f32, .s32...)
enum VType {
  VT_NONE, VT_F32, VT_F64, VT_S32, VT_U32,
};

typedef struct Inst {
  enum Opcode op;
  enum CondType cond;      // the mnemonic's condition (inside an IT block)
  bool setflags;           // the S suffix
  int width;               // 0, or .n (2), .w (4)
  enum VType vt[2];        // VFP types (vcvt has two)
  bool writeback;          // ldm/stm rn!
  Operand opr[4];
  // IT: the first condition and the then/else pattern ('t'/'e' x 0-3)
  char itpattern[4];
} Inst;
