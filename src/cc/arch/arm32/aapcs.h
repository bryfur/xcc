// FPC: the ARM procedure call standard (AAPCS) for xcc's ARM32 target:
// where each argument of a call goes, for the caller and the callee alike.
//
// Hard-float (the VFP variant): float and double arguments of a call that
// is not variadic go in s0-s15 (floats, back-filled) and d0-d7 (doubles);
// a variadic call passes everything by the base standard (core registers,
// doubles in an even pair).  Core registers r0-r3: a 64-bit value takes an
// even pair (r0:r1, r2:r3); when one does not fit, it and everything after
// it go on the stack (8-byte aligned).  A struct (by value) takes the core
// registers left, its rest going on the stack (split), if nothing is on the
// stack yet; else all of it goes on the stack.  A function with a struct
// parameter in registers (or variadic) pushes r0-r3 on entry, so that the
// struct (and the variadic arguments) are whole in memory.

#pragma once

#include <stdbool.h>

typedef struct Type Type;

enum ArgLocKind {
  AL_GP,     // core registers r[reg] .. r[reg + nregs - 1]
  AL_FP,     // s[reg] (a float), d[reg / 2] (a double: nregs 2)
  AL_STACK,  // the stack, at offset
};

typedef struct {
  enum ArgLocKind kind;
  int reg;
  int nregs;
  int offset;   // AL_STACK: where; AL_GP: a struct's rest on the stack (split), else -1
} ArgLoc;

typedef struct {
  int ncrn;           // next core register
  unsigned sfree;     // free s-registers (bits 0-15)
  bool fp_stacked;    // a VFP argument went to the stack: no more back-filling
  int nsaa;           // next stacked argument offset
  bool vfp;           // the VFP variant (not a variadic call)
} AapcsState;

void aapcs_init(AapcsState *st, bool variadic, bool hidden_ret);
ArgLoc aapcs_next(AapcsState *st, const Type *type);
// Where a function returns this type: 0 core registers (r0, r0:r1),
// 1 s0/d0, -1 memory (the caller's buffer, its address in r0)
int aapcs_return_kind(const Type *type, bool variadic);

typedef struct Function Function;
// Does the function push r0-r3 on entry (variadic, or a struct parameter
// in core registers)?
bool aapcs_spills_regargs(const Function *func);
