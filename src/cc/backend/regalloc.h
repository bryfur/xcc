// Register allocation

#pragma once

#include <stdbool.h>
#include <stddef.h>  // size_t

#include "ir.h"  // enum VRegSize, RegBits

typedef struct Vector BBContainer;
typedef struct Function Function;
typedef struct IR IR;
typedef struct VReg VReg;
typedef struct Vector Vector;

enum LiveIntervalState {
  LI_NORMAL,
  LI_SPILL,
};

typedef struct LiveInterval {
  RegBits occupied_reg_bit;  // Represent occupied registers in bit.
  enum LiveIntervalState state;
  int start;
  int end;
  int virt;  // Virtual register no.
  int phys;  // Mapped physical register no.
  int weight;  // FPC: its uses and defs, 10x each loop around them (what spilling it costs)
} LiveInterval;

typedef struct RegAllocSettings {
  RegBits (*detect_extra_occupied)(RegAlloc *ra, IR *ir);
  const int *reg_param_mapping;
  const int *reg_return_mapping;
  struct {
    int phys_max;              // Max physical register count.
    int phys_temporary_count;  // Temporary register count (= start index for saved registers)
    // FPC: 8-byte values in a pair of registers (phys, phys + 1) on a 32-bit
    // target: 0 none, 1 any two adjacent, 2 an even phys only.
    int pair_align;
    RegBits reserved;          // FPC: registers never allocated (the emitter's scratch).
  } regset[2];
  // FPC: registers occupied on the FP side by an IR (a helper call), or NULL.
  RegBits (*detect_extra_occupied_fp)(RegAlloc *ra, IR *ir);
} RegAllocSettings;

#define RAF_STACK_FRAME  (1 << 0)  // Require stack frame

typedef struct RegAlloc {
  const RegAllocSettings *settings;
  Vector *vregs;   // <VReg*>, non-const vregs
  Vector *consts;  // <VReg*>, const vregs
  Vector **vreg_table;  // [original_vreg_count]
  LiveInterval *intervals;  // size=vregs->len
  LiveInterval **sorted_intervals;

  RegBits used_reg_bits[2];
  int flag;
  int original_vreg_count;
} RegAlloc;

RegAlloc *new_reg_alloc(const RegAllocSettings *settings);
VReg *reg_alloc_spawn_raw(enum VRegSize vsize, int vflag);
VReg *reg_alloc_spawn(RegAlloc *ra, enum VRegSize vsize, int vflag);
VReg *reg_alloc_with_original(RegAlloc *ra, VReg *original);
VReg *reg_alloc_spawn_const(RegAlloc *ra, int64_t value, enum VRegSize vsize, int vflag);
#ifndef __NO_FLONUM
VReg *reg_alloc_spawn_fconst(RegAlloc *ra, double value, enum VRegSize vsize);
#endif
void alloc_physical_registers(RegAlloc *ra, BBContainer *bbcon);

// FPC: registers a vreg takes: 2 for an 8-byte value in a pair, else 1.
int vreg_reg_width(const RegAlloc *ra, const VReg *vreg);
