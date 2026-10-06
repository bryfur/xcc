// FPC: xcc's ARM32 target: ARMv8-M Mainline (Cortex-M33), Thumb-2, the
// AAPCS with the VFP variant (hard float, single precision FPU: doubles in
// software, through the run-time ABI's helpers).

#pragma once

#include <stdbool.h>
#include <stdint.h>  // int64_t

#include "emit_code.h"

// Core registers, by allocation index (arm32.c kReg32s): r0-r3 temporaries
// (the arguments), then r4-r6, r8-r11 and r7 (callee-saved; r7 is the
// frame pointer when there is a frame).  r12 (ip) is the emitter's scratch,
// never allocated; lr is pushed by every function.
#define PHYSICAL_REG_TEMPORARY   (4)
#define PHYSICAL_REG_MAX         (12)
// VFP: s0-s15 temporaries, s16-s31 callee-saved; s15 the emitter's scratch.
#define PHYSICAL_FREG_TEMPORARY  (16)
#define PHYSICAL_FREG_MAX        (32)

#define GET_FPREG_INDEX()  11  // r7
#define FREG_SCRATCH       15  // s15

#define FP    "r7"
#define SP    "sp"
#define LR    "lr"
#define PC    "pc"
#define IP    "ip"

extern const char *kReg32s[PHYSICAL_REG_MAX];
extern const char *kFReg32s[PHYSICAL_FREG_MAX];

char *im(int64_t x);  // #x
void mov_immediate(const char *dst, int64_t value);
bool is_thumb_modimm(uint32_t x);
const char *dreg(int phys);  // d<phys/2>

#define ALIGN8(x)  (((x) + 7) & ~7)
