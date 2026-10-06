// FPC: the AAPCS for xcc's ARM32 target (aapcs.h)
#include "../../../config.h"
#include "aapcs.h"

#include <assert.h>

#include "ast.h"  // Function
#include "fe_misc.h"  // is_small_struct
#include "type.h"
#include "util.h"
#include "var.h"

#define MAX_CORE_REGS  4
#define MAX_S_REGS     16

void aapcs_init(AapcsState *st, bool variadic, bool hidden_ret) {
  st->ncrn = hidden_ret ? 1 : 0;
  st->sfree = (1U << MAX_S_REGS) - 1;
  st->fp_stacked = false;
  st->nsaa = 0;
  st->vfp = !variadic;
}

static ArgLoc on_stack(AapcsState *st, size_t size, size_t align) {
  if (align < 4)
    align = 4;
  ArgLoc loc = {.kind = AL_STACK, .reg = -1, .nregs = 0};
  st->nsaa = ALIGN(st->nsaa, align);
  loc.offset = st->nsaa;
  st->nsaa += ALIGN(size, 4);
  return loc;
}

ArgLoc aapcs_next(AapcsState *st, const Type *type) {
  size_t size = type_size(type);
  size_t align = align_size(type);
  ArgLoc loc = {.kind = AL_GP, .reg = -1, .nregs = 0, .offset = -1};

  if (st->vfp && is_flonum(type)) {
    if (!st->fp_stacked) {
      if (size <= 4) {
        for (int i = 0; i < MAX_S_REGS; ++i) {
          if (st->sfree & (1U << i)) {
            st->sfree &= ~(1U << i);
            loc.kind = AL_FP;
            loc.reg = i;
            loc.nregs = 1;
            return loc;
          }
        }
      } else {
        for (int i = 0; i < MAX_S_REGS; i += 2) {
          if ((st->sfree & (3U << i)) == (3U << i)) {
            st->sfree &= ~(3U << i);
            loc.kind = AL_FP;
            loc.reg = i;
            loc.nregs = 2;
            return loc;
          }
        }
      }
    }
    st->fp_stacked = true;
    st->sfree = 0;
    return on_stack(st, size, align);
  }

  if (type->kind == TY_STRUCT) {
    int nwords = (int)((size + 3) / 4);
    if (align > 4)
      st->ncrn = ALIGN(st->ncrn, 2);
    if (st->ncrn < MAX_CORE_REGS && (st->nsaa == 0 || st->ncrn + nwords <= MAX_CORE_REGS)) {
      int n = MAX_CORE_REGS - st->ncrn;
      if (n > nwords)
        n = nwords;
      loc.kind = AL_GP;
      loc.reg = st->ncrn;
      loc.nregs = n;
      st->ncrn += n;
      if (n < nwords) {  // Split: the rest on the stack.
        st->ncrn = MAX_CORE_REGS;
        ArgLoc rest = on_stack(st, size - 4 * n, 4);
        loc.offset = rest.offset;
      }
      return loc;
    }
    st->ncrn = MAX_CORE_REGS;
    return on_stack(st, size, align);
  }

  int n = size > 4 ? 2 : 1;
  if (n == 2)
    st->ncrn = ALIGN(st->ncrn, 2);
  if (st->ncrn + n <= MAX_CORE_REGS) {
    loc.kind = AL_GP;
    loc.reg = st->ncrn;
    loc.nregs = n;
    st->ncrn += n;
    return loc;
  }
  st->ncrn = MAX_CORE_REGS;
  return on_stack(st, size, align);
}

bool aapcs_spills_regargs(const Function *func) {
  if (func->type->func.vaargs)
    return true;
  const Vector *params = func->params;
  if (params == NULL)
    return false;
  const Type *rettype = func->type->func.ret;
  AapcsState st;
  aapcs_init(&st, false, rettype->kind == TY_STRUCT && !is_small_struct(rettype) &&
                         type_size(rettype) > 0);
  for (int i = 0; i < params->len; ++i) {
    const VarInfo *varinfo = params->data[i];
    ArgLoc loc = aapcs_next(&st, varinfo->type);
    if (varinfo->type->kind == TY_STRUCT && loc.kind == AL_GP)
      return true;
  }
  return false;
}

int aapcs_return_kind(const Type *type, bool variadic) {
  if (type->kind == TY_STRUCT)
    return is_small_struct(type) ? 0 : -1;
  if (is_flonum(type) && !variadic)
    return 1;
  return 0;
}
