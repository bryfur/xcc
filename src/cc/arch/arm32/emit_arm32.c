// FPC: xcc's ARM32 target: functions (prologue, parameters, epilogue).
//
// The frame (r7, when there is one, points at the bottom of the saved
// registers; locals and spills below it):
//
//   incoming stack arguments          <- r7 + param_bottom
//   r0-r3 (variadic functions only)
//   r4..r11 used, r7, lr              (push)
//   s16..sN used                      (vpush)
//   locals, spills                    <- r7
//   outgoing stack arguments          <- sp
//
// sp stays 8-byte aligned (the AAPCS).

#include "../../../config.h"

#include <assert.h>
#include <inttypes.h>  // PRId64
#include <stdlib.h>
#include <string.h>

#include "aapcs.h"
#include "arm32.h"
#include "ast.h"
#include "be_aux.h"
#include "cc_misc.h"
#include "codegen.h"
#include "ir.h"
#include "regalloc.h"
#include "table.h"
#include "type.h"
#include "util.h"
#include "var.h"

#define MAX_REG_ARGS   (4)
#define MAX_FREG_ARGS  (16)

const ArchSetting kArchSetting = {
  .max_reg_args = {MAX_REG_ARGS, MAX_FREG_ARGS},
};

extern bool arm32_ir_calls_helper(const IR *ir);

// Callee-saved core registers by allocation index (r4, r5, r6, r8-r11);
// r7 and lr are pushed by every function.
static const int kCalleeSaveRegs[] = {4, 5, 6, 7, 8, 9, 10};

// The registers a function pushes, in ascending register number: a
// register list "{r4, r5, r7, lr}" and their count.
static int saved_core_regs_leaf(RegBits used, char *buf, size_t bufsize, bool leaf) {
  static const int kRegNo[] = {[4] = 4, [5] = 5, [6] = 6, [7] = 8, [8] = 9, [9] = 10, [10] = 11};
  bool regs[16] = {false};
  for (size_t i = 0; i < ARRAY_SIZE(kCalleeSaveRegs); ++i) {
    int idx = kCalleeSaveRegs[i];
    if (used & REGBIT(idx))
      regs[kRegNo[idx]] = true;
  }
  // FPC: a leaf function (no calls, no frame) saves r7 only when it uses it
  // (allocation index 11), and not lr
  regs[7] = !leaf || (used & REGBIT(11));
  regs[14] = !leaf;
  if (!leaf) {
    // An even count keeps the stack 8-byte aligned without a sub sp: r3
    // (scratch: no result is in it) pads the list
    int n = 0;
    for (int r = 0; r < 16; ++r)
      n += regs[r];
    if (n & 1)
      regs[3] = true;
  }
  int count = 0;
  size_t len = 0;
  buf[0] = '\0';
  for (int r = 0; r < 16; ++r) {
    if (!regs[r])
      continue;
    const char *name = r == 14 ? "lr" : fmt("r%d", r);
    len += snprintf(buf + len, bufsize - len, "%s%s", count == 0 ? "" : ", ", name);
    ++count;
  }
  return count;
}

static int saved_core_regs(RegBits used, char *buf, size_t bufsize) {
  return saved_core_regs_leaf(used, buf, bufsize, false);
}

// FPC: whether a function calls anything (a function, a run-time helper)
static bool makes_calls(BBContainer *bbcon) {
  for (int i = 0; i < bbcon->len; ++i) {
    BB *bb = bbcon->data[i];
    for (int j = 0; j < bb->irs->len; ++j) {
      IR *ir = bb->irs->data[j];
      if (ir->kind == IR_CALL || ir->kind == IR_ASM || ir->kind == IR_SUBSP ||
          arm32_ir_calls_helper(ir))
        return true;
    }
  }
  return false;
}

// VFP callee-saved: s16 up to the highest used (vpush takes a range).
static int saved_vfp_count(RegBits fused) {
  int top = -1;
  for (int i = PHYSICAL_FREG_TEMPORARY; i < PHYSICAL_FREG_MAX; ++i)
    if (fused & REGBIT(i))
      top = i;
  return top < 0 ? 0 : top - PHYSICAL_FREG_TEMPORARY + 1;
}

int calculate_func_param_bottom(Function *func) {
  FuncBackend *fnbe = func->extra;
  char buf[64];
  int ncore = saved_core_regs(fnbe->ra->used_reg_bits[GPREG], buf, sizeof(buf));
  int nvfp = saved_vfp_count(fnbe->ra->used_reg_bits[FPREG]);
  return 4 * ncore + 4 * nvfp + (aapcs_spills_regargs(func) ? 16 : 0);
}

static const char *fp_mem(int offset) {
  if (offset >= -255 && offset <= 4095)
    return offset != 0 ? fmt("[%s, #%d]", FP, offset) : fmt("[%s]", FP);
  mov_immediate(IP, offset);
  return fmt("[%s, %s]", FP, IP);
}

static const char *fp_mem_wide(int offset) {  // ldrd/strd, vldr/vstr
  if (offset >= -1020 && offset <= 1020 && (offset & 3) == 0)
    return offset != 0 ? fmt("[%s, #%d]", FP, offset) : fmt("[%s]", FP);
  mov_immediate(IP, offset);
  EMIT_ASM("add", IP, IP, FP);
  return fmt("[%s]", IP);
}

// A parameter from where the caller put it to where the function keeps it.
static void move_param(VReg *vreg, const ArgLoc *loc) {
  bool wide = vreg->vsize == VRegSize8;
  bool spilled = (vreg->flag & VRF_SPILLED) != 0;
  if (loc->kind == AL_FP) {
    if (spilled) {
      EMIT_ASM("vstr", wide ? dreg(loc->reg) : kFReg32s[loc->reg], fp_mem_wide(vreg->frame.offset));
    } else if (vreg->phys != loc->reg) {
      EMIT_ASM("vmov.f32", kFReg32s[vreg->phys], kFReg32s[loc->reg]);
      if (wide)
        EMIT_ASM("vmov.f32", kFReg32s[vreg->phys + 1], kFReg32s[loc->reg + 1]);
    }
    return;
  }

  assert(loc->kind == AL_GP);
  const char *lo = kReg32s[loc->reg];
  const char *hi = wide ? kReg32s[loc->reg + 1] : NULL;
  if (spilled) {
    if (wide)
      EMIT_ASM("strd", lo, hi, fp_mem_wide(vreg->frame.offset));
    else {
      const char *op = vreg->vsize == VRegSize1 ? "strb" : vreg->vsize == VRegSize2 ? "strh" : "str";
      EMIT_ASM(op, lo, fp_mem(vreg->frame.offset));
    }
    return;
  }
  if (vreg->flag & VRF_FLONUM) {  // floating point in core registers
    if (wide)
      EMIT_ASM("vmov", dreg(vreg->phys), lo, hi);
    else
      EMIT_ASM("vmov", kFReg32s[vreg->phys], lo);
    return;
  }
  if (vreg->phys == loc->reg)
    return;
  EMIT_ASM("mov", kReg32s[vreg->phys], lo);
  if (wide)
    EMIT_ASM("mov", kReg32s[vreg->phys + 1], hi);
}

static void move_params_to_assigned(Function *func) {
  FuncBackend *fnbe = func->extra;
  // The struct result's buffer, in r0.
  if (fnbe->struct_retvar != NULL) {
    VReg *vreg = fnbe->struct_retvar->local.vreg;
    ArgLoc loc = {.kind = AL_GP, .reg = 0, .nregs = 1};
    move_param(vreg, &loc);
  }

  AapcsState st;
  aapcs_init(&st, func->type->func.vaargs, fnbe->struct_retvar != NULL);
  const Vector *params = func->params;
  for (int i = 0; params != NULL && i < params->len; ++i) {
    VarInfo *varinfo = params->data[i];
    ArgLoc loc = aapcs_next(&st, varinfo->type);
    if (loc.kind == AL_STACK)
      continue;  // Used where it is.
    VReg *vreg = varinfo->local.vreg;
    if (vreg == NULL)
      continue;  // A struct: where r0-r3 were pushed on entry.
    move_param(vreg, &loc);
  }
}

static size_t detect_funcall_work_size(Function *func) {
  FuncBackend *fnbe = func->extra;
  Vector *funcalls = fnbe->funcalls;
  size_t max = 0;
  if (funcalls != NULL) {
    for (int i = 0; i < funcalls->len; ++i) {
      Expr *funcall = funcalls->data[i];
      FuncallInfo *fcinfo = funcall->funcall.fcinfo;
      IR *ir = fcinfo->call;
      ir->call->caller_saves = new_vector();  // none: r0-r3 are not kept live across
      max = MAX(max, ir->call->stack_args_size);
    }
  }
  return max;
}

static bool is_asm(Stmt *stmt) {
  return stmt->kind == ST_ASM;
}

void emit_defun_body(Function *func) {
  emit_comment(NULL);
  _TEXT();

  bool global = true;
  const Name *name = func->ident->ident;
  const VarInfo *varinfo = scope_find(global_scope, name, NULL);
  if (varinfo != NULL)
    global = (varinfo->storage & VS_STATIC) == 0;

  {
    char *label = format_func_name(name, global);
    if (varinfo->storage & VS_WEAK)
      _WEAK(label);
    else if (global)
      _GLOBL(label);
    else
      _LOCAL(label);
    EMIT_ALIGN(2);
    EMIT_ASM(".type", quote_label(fmt_name(name)), "%function");
    EMIT_ASM(".thumb_func");
    EMIT_LABEL(label);
  }

  bool no_stmt = true;
  if (func->body_block != NULL) {
    Vector *stmts = func->body_block->block.stmts;
    for (int i = 0; i < stmts->len; ++i) {
      Stmt *stmt = stmts->data[i];
      if (stmt == NULL)
        continue;
      if (!is_asm(stmt)) {
        no_stmt = false;
        break;
      }
    }
  }

  extern void arm32_set_variadic_func(bool variadic);
  arm32_set_variadic_func(func->type->func.vaargs);

  FuncBackend *fnbe = func->extra;
  size_t funcall_work_size = detect_funcall_work_size(func);
  fnbe->stack_work_size = funcall_work_size;
  {
    VReg *vreg = fnbe->stack_work_size_vreg;
    if (vreg != NULL) {
      assert(vreg->flag & VRF_CONST);
      vreg->fixnum = funcall_work_size;
    }
  }

  bool vaargs = aapcs_spills_regargs(func);  // r0-r3 pushed on entry
  bool fp_saved = fnbe->frame_size > 0 || (fnbe->ra->flag & RAF_STACK_FRAME);
  // FPC: a leaf (no calls, nothing on the stack) pushes only the callee-saved
  // registers it uses; no lr, no 8-byte alignment (it calls nothing)
  bool leaf = !vaargs && !fp_saved && funcall_work_size == 0 && !makes_calls(fnbe->bbcon);
  char saves[64];
  int ncore = saved_core_regs_leaf(fnbe->ra->used_reg_bits[GPREG], saves, sizeof(saves), leaf);
  int nvfp = saved_vfp_count(fnbe->ra->used_reg_bits[FPREG]);
  int pushed = 4 * ncore + 4 * nvfp + (vaargs ? 16 : 0);
  size_t frame_size = leaf ? 0 : ALIGN8(fnbe->frame_size + funcall_work_size + pushed) - pushed;

  if (!no_stmt) {
    // Prologue
    if (vaargs)
      EMIT_ASM("push", "{r0, r1, r2, r3}");
    if (ncore > 0)
      EMIT_ASM("push", fmt("{%s}", saves));
    if (nvfp > 0)
      EMIT_ASM("vpush", nvfp == 1 ? "{s16}" : fmt("{s16-s%d}", 16 + nvfp - 1));
    if (fp_saved)
      EMIT_ASM("mov", FP, SP);
    if (frame_size > 0) {
      if (frame_size <= 4095) {
        EMIT_ASM("sub", SP, SP, im(frame_size));
      } else {
        mov_immediate(IP, frame_size);
        EMIT_ASM("sub", SP, SP, IP);
      }
    }

    move_params_to_assigned(func);
  }

  emit_bb_irs(fnbe->bbcon);

  if (!function_not_returned(fnbe)) {
    // Epilogue
    if (!no_stmt) {
      if (fp_saved) {
        EMIT_ASM("mov", SP, FP);
      } else if (frame_size > 0) {
        if (frame_size <= 4095) {
          EMIT_ASM("add", SP, SP, im(frame_size));
        } else {
          mov_immediate(IP, frame_size);
          EMIT_ASM("add", SP, SP, IP);
        }
      }
      if (nvfp > 0)
        EMIT_ASM("vpop", nvfp == 1 ? "{s16}" : fmt("{s16-s%d}", 16 + nvfp - 1));
      if (leaf) {
        if (ncore > 0)
          EMIT_ASM("pop", fmt("{%s}", saves));
        EMIT_ASM("bx", LR);
        return;
      }
      if (!vaargs) {
        // lr comes back as pc: the return.
        size_t len = strlen(saves);
        assert(len >= 2 && strcmp(saves + len - 2, "lr") == 0);
        saves[len - 2] = '\0';
        EMIT_ASM("pop", fmt("{%spc}", saves));
        return;
      }
      EMIT_ASM("pop", fmt("{%s}", saves));
      EMIT_ASM("add", SP, SP, im(16));
    }
    EMIT_ASM("bx", LR);
  }
}

// The start of every assembly file.
void emit_arm32_preamble(void) {
  EMIT_ASM(".syntax", "unified");
  EMIT_ASM(".cpu", "cortex-m33");
  EMIT_ASM(".fpu", "fpv5-sp-d16");
  EMIT_ASM(".thumb");
  // The AAPCS attributes GNU ld checks: hard float (VFP arguments), 8-byte
  // aligned stack, enums of 4 bytes, wchar_t of 4 bytes.
  EMIT_ASM(".eabi_attribute", "28, 1");   // Tag_ABI_VFP_args
  EMIT_ASM(".eabi_attribute", "24, 1");   // Tag_ABI_align_needed
  EMIT_ASM(".eabi_attribute", "25, 1");   // Tag_ABI_align_preserved
  EMIT_ASM(".eabi_attribute", "26, 2");   // Tag_ABI_enum_size: int
  EMIT_ASM(".eabi_attribute", "18, 4");   // Tag_ABI_PCS_wchar_t
}
