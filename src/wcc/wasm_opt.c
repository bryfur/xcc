// FPC: wcc's optimiser over a function's finished WebAssembly code.
//
// The code generator gives every variable and temporary a local of its own
// and passes values through them (`local.set t; local.get t`).  A wasm
// runtime keeps only a few locals in registers (the JIT: the 5-7 most used,
// the rest in its frame), so this pass, from the locals' liveness over the
// structured control flow:
//   - leaves a value on the stack where it is set and read at once and not
//     read again (`local.set t; local.get t`: nothing);
//   - drops stores to locals nobody reads (a `local.set` to `drop`, which
//     goes with a pure value before it), and `local.tee`s of them;
//   - gives locals whose lives do not overlap the same index (an
//     interference graph, coloured from the most used local down, a copy's
//     two locals together where they can be): fewer, busier locals;
//   - removes the copies that became `local.get x; local.set x`.
// A function with exception handling (setjmp) is left as it is.

#include "../config.h"
#include "wcc.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "table.h"
#include "util.h"
#include "wasm.h"
#include "wasm_obj.h"

typedef uint32_t Bits;  // bit sets of locals
#define BWORDS(n)  (((n) + 31) / 32)
#define BTEST(b, i)  ((b)[(i) >> 5] & (1u << ((i) & 31)))
#define BSET(b, i)   ((b)[(i) >> 5] |= 1u << ((i) & 31))
#define BCLR(b, i)   ((b)[(i) >> 5] &= ~(1u << ((i) & 31)))

#define MAX_COALESCE_LOCALS  1024  // interference matrix: n * n bits at most

typedef struct {
  uint32_t pos;    // in the old code
  uint32_t len;    // bytes
  uint32_t imm;    // local index; branch: index in tgts of its target(s)
  int32_t match;   // block/loop/if: its end; end/else: its opener
  int32_t alt;     // if: its else (-1: none); br_table: its target count
  uint8_t op;      // the opcode (OP_DROP, OP_LOCAL_TEE when changed)
  uint8_t del;     // removed
  uint8_t reloc;   // a relocation points into it (kept as it is)
  uint8_t loops;   // loops around it
  uint8_t eqz;     // an i32.eqz before it
} WI;

typedef struct {
  unsigned char *code;
  uint32_t body;      // offset of the body (after the locals' declarations)
  uint32_t size;
  WI *wi;
  int n;
  int32_t *tgts;      // branch targets (instruction indices; n: the function's end)
  int ntgts;
  uint32_t nparams, nlocals;
  unsigned char *ltype;
  // basic blocks
  int nbb;
  int *bb_start;      // [nbb + 1]
  int *bb_of;         // [n]
  Bits *live_in, *live_out, *use, *def;  // [nbb * words]
  uint32_t words;
} WOpt;

static bool skip_blocktype(unsigned char **pp) {
  unsigned char c = **pp;
  if (c == WT_VOID || (c >= 0x6f && c <= 0x7f)) {
    ++*pp;
  } else {
    read_leb128(*pp, pp);
  }
  return true;
}

// Decode the body: false when it has something this pass does not know.
static bool decode(WOpt *o, Vector *relocs) {
  unsigned char *p = o->code + o->body, *end = o->code + o->size;
  int cap = 64, tcap = 16;
  o->wi = malloc(sizeof(WI) * cap);
  o->tgts = malloc(sizeof(int32_t) * tcap);
  o->n = o->ntgts = 0;
  int32_t stack[256];
  int sp = 0;
  int loops = 0;
  bool ok = true;
  while (p < end && ok) {
    if (o->n >= cap) {
      cap *= 2;
      o->wi = realloc(o->wi, sizeof(WI) * cap);
    }
    WI *w = &o->wi[o->n];
    memset(w, 0, sizeof(*w));
    w->pos = p - o->code;
    w->match = w->alt = -1;
    unsigned char op = *p++;
    w->op = op;
    w->loops = loops > 255 ? 255 : loops;
    switch (op) {
    case OP_BLOCK: case OP_LOOP: case OP_IF:
      skip_blocktype(&p);
      if (sp >= (int)(sizeof(stack) / sizeof(*stack))) {
        ok = false;
        break;
      }
      stack[sp++] = o->n;
      if (op == OP_LOOP)
        ++loops;
      break;
    case OP_ELSE:
      if (sp <= 0 || o->wi[stack[sp - 1]].op != OP_IF) {
        ok = false;
        break;
      }
      w->match = stack[sp - 1];
      o->wi[stack[sp - 1]].alt = o->n;
      break;
    case OP_END:
      if (sp > 0) {
        int k = stack[--sp];
        w->match = k;
        o->wi[k].match = o->n;
        if (o->wi[k].op == OP_LOOP)
          --loops;
      } else if (p != end) {
        ok = false;
      }
      break;
    case OP_BR: case OP_BR_IF: case OP_BR_TABLE:
      {
        uint32_t cnt = op == OP_BR_TABLE ? read_uleb128(p, &p) + 1 : 1;
        if (o->ntgts + (int)cnt > tcap) {
          while (o->ntgts + (int)cnt > tcap)
            tcap *= 2;
          o->tgts = realloc(o->tgts, sizeof(int32_t) * tcap);
        }
        w->imm = o->ntgts;
        w->alt = cnt;
        for (uint32_t i = 0; i < cnt; ++i) {
          uint32_t depth = read_uleb128(p, &p);
          // The opener for now (-1: the function); its target later.
          o->tgts[o->ntgts++] = depth < (uint32_t)sp ? stack[sp - 1 - depth] : -1;
          if (depth > (uint32_t)sp)
            ok = false;
        }
      }
      break;
    case OP_CALL: case OP_LOCAL_GET: case OP_LOCAL_SET: case OP_LOCAL_TEE:
    case OP_GLOBAL_GET: case OP_GLOBAL_SET: case 0x25: case 0x26: case 0xd2:
      w->imm = read_uleb128(p, &p);
      break;
    case OP_CALL_INDIRECT:
      read_uleb128(p, &p);
      read_uleb128(p, &p);
      break;
    case OP_I32_CONST: case OP_I64_CONST:
      read_leb128(p, &p);
      break;
    case OP_F32_CONST: p += 4; break;
    case OP_F64_CONST: p += 8; break;
    case OP_MEMORY_SIZE: case OP_MEMORY_GROW: case 0xd0: p += 1; break;
    case OP_0xFC:
      {
        uint32_t sub = read_uleb128(p, &p);
        if (sub <= 7) {
        } else if (sub == 8) {
          read_uleb128(p, &p);
          p += 1;
        } else if (sub == 9 || sub == 13 || (sub >= 15 && sub <= 17)) {
          read_uleb128(p, &p);
        } else if (sub == OPFC_MEMORY_COPY) {
          p += 2;
        } else if (sub == OPFC_MEMORY_FILL) {
          p += 1;
        } else if (sub == 12 || sub == 14) {
          read_uleb128(p, &p);
          read_uleb128(p, &p);
        } else {
          ok = false;
        }
      }
      break;
    default:
      if (op >= 0x28 && op <= 0x3e) {  // loads and stores: memarg
        read_uleb128(p, &p);
        read_uleb128(p, &p);
      } else if (op == OP_UNREACHABLE || op == OP_NOP || op == OP_RETURN ||
                 op == OP_DROP || op == OP_SELECT || (op >= 0x45 && op <= 0xc4)) {
      } else {
        ok = false;  // exception handling, typed select, ...
      }
      break;
    }
    w->len = (p - o->code) - w->pos;
    ++o->n;
  }
  if (!ok || p != end || sp != 0 || o->n == 0 || o->wi[o->n - 1].op != OP_END)
    return false;

  // Branch targets: a loop's start, a block's or if's end; the function's end.
  for (int i = 0; i < o->ntgts; ++i) {
    int k = o->tgts[i];
    o->tgts[i] = k < 0 ? o->n - 1 : o->wi[k].op == OP_LOOP ? k : o->wi[k].match;
  }

  // Instructions relocations point into stay as they are.
  if (relocs != NULL) {
    for (int i = 0; i < relocs->len; ++i) {
      RelocInfo *ri = relocs->data[i];
      int lo = 0, hi = o->n - 1;
      while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (o->wi[mid].pos <= ri->offset)
          lo = mid;
        else
          hi = mid - 1;
      }
      o->wi[lo].reloc = 1;
    }
  }
  return true;
}

static bool is_control(unsigned char op) {
  switch (op) {
  case OP_BLOCK: case OP_LOOP: case OP_IF: case OP_ELSE: case OP_END:
  case OP_BR: case OP_BR_IF: case OP_BR_TABLE: case OP_RETURN: case OP_UNREACHABLE:
    return true;
  default:
    return false;
  }
}

static void build_blocks(WOpt *o) {
  int n = o->n;
  char *leader = calloc(n + 1, 1);
  leader[0] = 1;
  for (int i = 0; i < n; ++i) {
    unsigned char op = o->wi[i].op;
    if (op == OP_LOOP || op == OP_END)
      leader[i] = 1;
    if (is_control(op))
      leader[i + 1] = 1;
  }
  int nbb = 0;
  for (int i = 0; i < n; ++i)
    nbb += leader[i];
  o->nbb = nbb;
  o->bb_start = malloc(sizeof(int) * (nbb + 1));
  o->bb_of = malloc(sizeof(int) * n);
  int b = -1;
  for (int i = 0; i < n; ++i) {
    if (leader[i])
      o->bb_start[++b] = i;
    o->bb_of[i] = b;
  }
  o->bb_start[nbb] = n;
  free(leader);
}

// Successor blocks of block b; returns the count (into succ, which has room
// for the largest br_table + 2).
static int successors(WOpt *o, int b, int *succ) {
  int last = o->bb_start[b + 1] - 1;
  WI *w = &o->wi[last];
  int ns = 0;
  switch (w->op) {
  case OP_BR:
    succ[ns++] = o->bb_of[o->tgts[w->imm]];
    break;
  case OP_BR_IF:
    succ[ns++] = o->bb_of[o->tgts[w->imm]];
    if (last + 1 < o->n)
      succ[ns++] = b + 1;
    break;
  case OP_BR_TABLE:
    for (int i = 0; i < w->alt; ++i)
      succ[ns++] = o->bb_of[o->tgts[w->imm + i]];
    break;
  case OP_RETURN: case OP_UNREACHABLE:
    break;
  case OP_IF:
    succ[ns++] = b + 1;
    succ[ns++] = o->bb_of[w->alt >= 0 ? w->alt + 1 : w->match];
    break;
  case OP_ELSE:
    succ[ns++] = o->bb_of[o->wi[w->match].match];
    break;
  default:
    if (last + 1 < o->n)
      succ[ns++] = b + 1;
    break;
  }
  return ns;
}

static int max_succ(WOpt *o) {
  int m = 2;
  for (int i = 0; i < o->n; ++i)
    if (o->wi[i].op == OP_BR_TABLE && o->wi[i].alt > m)
      m = o->wi[i].alt;
  return m;
}

static inline bool is_local_op(const WI *w) {
  return !w->del && (w->op == OP_LOCAL_GET || w->op == OP_LOCAL_SET || w->op == OP_LOCAL_TEE);
}

static void liveness(WOpt *o) {
  uint32_t W = o->words = BWORDS(o->nlocals);
  size_t sz = sizeof(Bits) * W * o->nbb;
  free(o->live_in); free(o->live_out); free(o->use); free(o->def);
  o->live_in = calloc(1, sz + sizeof(Bits));
  o->live_out = calloc(1, sz + sizeof(Bits));
  o->use = calloc(1, sz + sizeof(Bits));
  o->def = calloc(1, sz + sizeof(Bits));
  for (int b = 0; b < o->nbb; ++b) {
    Bits *use = o->use + b * W, *def = o->def + b * W;
    for (int i = o->bb_start[b]; i < o->bb_start[b + 1]; ++i) {
      WI *w = &o->wi[i];
      if (!is_local_op(w))
        continue;
      if (w->op == OP_LOCAL_GET) {
        if (!BTEST(def, w->imm))
          BSET(use, w->imm);
      } else {
        BSET(def, w->imm);
      }
    }
  }
  int *succ = malloc(sizeof(int) * (max_succ(o) + 2));
  for (bool changed = true; changed; ) {
    changed = false;
    for (int b = o->nbb; --b >= 0; ) {
      Bits *out = o->live_out + b * W, *in = o->live_in + b * W;
      int ns = successors(o, b, succ);
      for (int k = 0; k < ns; ++k) {
        Bits *sin = o->live_in + succ[k] * W;
        for (uint32_t j = 0; j < W; ++j)
          out[j] |= sin[j];
      }
      Bits *use = o->use + b * W, *def = o->def + b * W;
      for (uint32_t j = 0; j < W; ++j) {
        Bits v = use[j] | (out[j] & ~def[j]);
        if (v != in[j]) {
          in[j] = v;
          changed = true;
        }
      }
    }
  }
  free(succ);
}

static int prev_live(WOpt *o, int i, int lo) {
  while (--i >= lo && o->wi[i].del)
    ;
  return i;
}

static bool pure_push(const WI *w) {
  if (w->del || w->reloc)
    return false;
  switch (w->op) {
  case OP_LOCAL_GET: case OP_GLOBAL_GET:
  case OP_I32_CONST: case OP_I64_CONST: case OP_F32_CONST: case OP_F64_CONST:
    return true;
  default:
    return false;
  }
}

// Stores nobody reads, values read once right where they are set.
// Returns whether something changed.
static bool cleanup(WOpt *o, bool use_tee) {
  uint32_t W = o->words;
  Bits *live = malloc(sizeof(Bits) * (W + 1));
  bool changed = false;
  for (int b = 0; b < o->nbb; ++b) {
    int lo = o->bb_start[b];
    memcpy(live, o->live_out + b * W, sizeof(Bits) * W);
    for (int i = o->bb_start[b + 1]; --i >= lo; ) {
      WI *w = &o->wi[i];
      if (!is_local_op(w))
        continue;
      uint32_t x = w->imm;
      if (w->op == OP_LOCAL_GET) {
        int k = prev_live(o, i, lo);
        if (k >= 0 && o->wi[k].op == OP_LOCAL_SET && o->wi[k].imm == x) {
          if (!BTEST(live, x)) {
            w->del = o->wi[k].del = 1;  // the value stays on the stack
            changed = true;
            i = k;
            continue;
          } else if (use_tee) {
            w->del = 1;
            o->wi[k].op = OP_LOCAL_TEE;
            changed = true;
            i = k;
            continue;
          }
        }
        BSET(live, x);
      } else {
        if (!BTEST(live, x)) {
          if (w->op == OP_LOCAL_TEE) {
            w->del = 1;
          } else {
            int k = prev_live(o, i, lo);
            if (k >= 0 && pure_push(&o->wi[k])) {
              w->del = o->wi[k].del = 1;
            } else {
              w->op = OP_DROP;
            }
          }
          changed = true;
          continue;
        }
        if (w->op == OP_LOCAL_SET) {
          // `local.get x; local.set x`
          int k = prev_live(o, i, lo);
          if (k >= 0 && o->wi[k].op == OP_LOCAL_GET && o->wi[k].imm == x && !o->wi[k].del) {
            w->del = o->wi[k].del = 1;
            changed = true;
            i = k;
            continue;
          }
        }
        BCLR(live, x);
      }
    }
  }
  free(live);
  return changed;
}

// Locals whose lives do not overlap share an index.
static void coalesce(WOpt *o) {
  uint32_t nl = o->nlocals, W = o->words;
  if (nl <= o->nparams || nl > MAX_COALESCE_LOCALS)
    return;
  Bits *ig = calloc((size_t)nl * W + 1, sizeof(Bits));  // interference rows
  uint32_t *weight = calloc(nl, sizeof(uint32_t));
  int32_t *partner = malloc(sizeof(int32_t) * nl);
  for (uint32_t i = 0; i < nl; ++i)
    partner[i] = -1;
  Bits *live = malloc(sizeof(Bits) * (W + 1));

#define EDGE(a, b)  do { BSET(ig + (size_t)(a) * W, (b)); BSET(ig + (size_t)(b) * W, (a)); } while (0)
  for (int b = 0; b < o->nbb; ++b) {
    int lo = o->bb_start[b];
    memcpy(live, o->live_out + b * W, sizeof(Bits) * W);
    for (int i = o->bb_start[b + 1]; --i >= lo; ) {
      WI *w = &o->wi[i];
      if (!is_local_op(w))
        continue;
      uint32_t x = w->imm;
      uint32_t lw = w->loops * 3 > 24 ? 1u << 24 : 1u << (w->loops * 3);
      weight[x] += lw;
      if (w->op == OP_LOCAL_GET) {
        BSET(live, x);
        continue;
      }
      int src = -1;
      int k = prev_live(o, i, lo);
      if (k >= 0 && o->wi[k].op == OP_LOCAL_GET && o->wi[k].imm != x) {
        src = o->wi[k].imm;
        if (o->ltype[src] == o->ltype[x]) {
          partner[x] = src;
          if (partner[src] < 0)
            partner[src] = x;
        }
      }
      for (uint32_t j = 0; j < W; ++j) {
        Bits m = live[j];
        while (m != 0) {
          uint32_t y = j * 32 + __builtin_ctz(m);
          m &= m - 1;
          if (y != x && (int)y != src)
            EDGE(x, y);
        }
      }
      BCLR(live, x);
    }
  }
  // At the entry every local is set: the parameters to the arguments, the
  // others to 0 (which those read before they are set need).
  {
    Bits *in = o->live_in;
    for (uint32_t x = o->nparams; x < nl; ++x) {
      if (!BTEST(in, x))
        continue;
      for (uint32_t p = 0; p < o->nparams; ++p)
        EDGE(x, p);
      for (uint32_t y = o->nparams; y < nl; ++y)
        if (y != x && BTEST(in, y))
          EDGE(x, y);
    }
  }
#undef EDGE

  // Colours: the parameters keep theirs; the others from the most used.
  uint32_t *order = malloc(sizeof(uint32_t) * nl);
  uint32_t no = 0;
  for (uint32_t x = o->nparams; x < nl; ++x)
    if (weight[x] > 0)
      order[no++] = x;
  for (uint32_t i = 1; i < no; ++i) {  // (insertion sort: stable)
    uint32_t x = order[i], j = i;
    while (j > 0 && weight[order[j - 1]] < weight[x]) {
      order[j] = order[j - 1];
      --j;
    }
    order[j] = x;
  }
  int32_t *color = malloc(sizeof(int32_t) * nl);
  Bits *crow = calloc((size_t)nl * W + 1, sizeof(Bits));  // a colour's interference
  unsigned char *ctype = malloc(nl);
  uint32_t ncolors = o->nparams;
  for (uint32_t x = 0; x < nl; ++x)
    color[x] = -1;
  for (uint32_t p = 0; p < o->nparams; ++p) {
    color[p] = p;
    ctype[p] = o->ltype[p];
    memcpy(crow + (size_t)p * W, ig + (size_t)p * W, sizeof(Bits) * W);
  }
  for (uint32_t i = 0; i < no; ++i) {
    uint32_t x = order[i];
    int c = -1;
    int pt = partner[x];
    if (pt >= 0 && color[pt] >= 0 && ctype[color[pt]] == o->ltype[x] &&
        !BTEST(crow + (size_t)color[pt] * W, x))
      c = color[pt];
    for (uint32_t k = 0; c < 0 && k < ncolors; ++k)
      if (ctype[k] == o->ltype[x] && !BTEST(crow + (size_t)k * W, x))
        c = k;
    if (c < 0) {
      c = ncolors++;
      ctype[c] = o->ltype[x];
    }
    color[x] = c;
    Bits *cr = crow + (size_t)c * W, *xr = ig + (size_t)x * W;
    for (uint32_t j = 0; j < W; ++j)
      cr[j] |= xr[j];
  }

  // New indices: the parameters, then the colours by type.
  static const unsigned char kTypes[] = {WT_I32, WT_I64, WT_F32, WT_F64};
  int32_t *cindex = malloc(sizeof(int32_t) * (ncolors + 1));
  uint32_t idx = o->nparams;
  for (uint32_t c = 0; c < o->nparams; ++c)
    cindex[c] = c;
  for (int t = 0; t < 4; ++t)
    for (uint32_t c = o->nparams; c < ncolors; ++c)
      if (ctype[c] == kTypes[t])
        cindex[c] = idx++;
  unsigned char *ntype = malloc(idx + 1);
  memcpy(ntype, o->ltype, o->nparams);
  for (uint32_t c = o->nparams; c < ncolors; ++c)
    ntype[cindex[c]] = ctype[c];

  for (int i = 0; i < o->n; ++i) {
    WI *w = &o->wi[i];
    if (is_local_op(w)) {
      assert(color[w->imm] >= 0);
      w->imm = cindex[color[w->imm]];
    }
  }
  free(o->ltype);
  o->ltype = ntype;
  o->nlocals = idx;

  free(cindex); free(ctype); free(crow); free(color); free(order);
  free(live); free(partner); free(weight); free(ig);
}

// `if (then nothing) else B end`: `if B end` on the opposite condition
// (a comparison just before it inverted, or i32.eqz); `if A else end`:
// `if A end`
static void drop_empty_branches(WOpt *o) {
  static const unsigned char kInv[][2] = {
    {OP_I32_EQ, OP_I32_NE}, {OP_I32_LT_S, OP_I32_GE_S}, {OP_I32_LT_U, OP_I32_GE_U},
    {OP_I32_GT_S, OP_I32_LE_S}, {OP_I32_GT_U, OP_I32_LE_U},
    {OP_I64_EQ, OP_I64_NE}, {OP_I64_LT_S, OP_I64_GE_S}, {OP_I64_LT_U, OP_I64_GE_U},
    {OP_I64_GT_S, OP_I64_LE_S}, {OP_I64_GT_U, OP_I64_LE_U},
  };
  for (int i = 0; i < o->n; ++i) {
    WI *w = &o->wi[i];
    if (w->del || w->op != OP_IF || w->alt < 0 || o->code[w->pos + 1] != WT_VOID)
      continue;
    WI *el = &o->wi[w->alt];
    int k = i + 1;
    while (k < o->n && o->wi[k].del)
      ++k;
    if (k == w->alt) {  // (then nothing)
      int c = prev_live(o, i, 0);
      bool done = false;
      if (c >= 0 && !o->wi[c].reloc && o->wi[c].len == 1) {
        unsigned char op = o->wi[c].op;
        for (size_t m = 0; m < sizeof(kInv) / sizeof(*kInv) && !done; ++m) {
          if (op == kInv[m][0] || op == kInv[m][1]) {
            o->wi[c].op = op == kInv[m][0] ? kInv[m][1] : kInv[m][0];
            done = true;
          }
        }
        if (!done && op == OP_I32_EQZ) {
          o->wi[c].del = 1;
          done = true;
        }
      }
      if (!done)
        w->eqz = 1;
      el->del = 1;
      continue;
    }
    k = w->alt + 1;
    while (k < o->n && o->wi[k].del)
      ++k;
    if (k == w->match)  // (else nothing)
      el->del = 1;
  }
}

static void encode(WOpt *o, DataStorage *code, Vector *relocs) {
  DataStorage out;
  data_init(&out);
  data_reserve(&out, o->size);
  // Locals: runs of one type.
  uint32_t nruns = 0;
  for (uint32_t i = o->nparams; i < o->nlocals; ++i)
    if (i == o->nparams || o->ltype[i] != o->ltype[i - 1])
      ++nruns;
  data_uleb128(&out, -1, nruns);
  for (uint32_t i = o->nparams; i < o->nlocals; ) {
    uint32_t j = i;
    while (j < o->nlocals && o->ltype[j] == o->ltype[i])
      ++j;
    data_uleb128(&out, -1, j - i);
    data_push(&out, o->ltype[i]);
    i = j;
  }
  uint32_t *npos = malloc(sizeof(uint32_t) * (o->n + 1));
  for (int i = 0; i < o->n; ++i) {
    WI *w = &o->wi[i];
    npos[i] = out.len;
    if (w->del)
      continue;
    switch (w->op) {
    case OP_LOCAL_GET: case OP_LOCAL_SET: case OP_LOCAL_TEE:
      data_push(&out, w->op);
      data_uleb128(&out, -1, w->imm);
      break;
    case OP_DROP:
      data_push(&out, OP_DROP);
      break;
    default:
      if (w->eqz)
        data_push(&out, OP_I32_EQZ);
      if (w->len == 1) {  // (an operation without immediates, maybe changed)
        data_push(&out, w->op);
        break;
      }
      data_append(&out, o->code + w->pos, w->len);
      break;
    }
  }
  if (relocs != NULL) {
    for (int i = 0; i < relocs->len; ++i) {
      RelocInfo *ri = relocs->data[i];
      int lo = 0, hi = o->n - 1;
      while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (o->wi[mid].pos <= ri->offset)
          lo = mid;
        else
          hi = mid - 1;
      }
      assert(!o->wi[lo].del);
      ri->offset = npos[lo] + (ri->offset - o->wi[lo].pos);
    }
  }
  free(npos);
  code->len = 0;
  data_append(code, out.buf, out.len);
  data_release(&out);
}

// code: the function's locals' declarations and body (from offset 0);
// ptypes: its parameters' wasm types.
void wasm_opt_func(DataStorage *code, Vector *relocs, const unsigned char *ptypes,
                   uint32_t nparams) {
  static int disabled = -1;
  if (disabled < 0) {
    const char *e = getenv("WCC_NOWOPT");
    disabled = e != NULL && *e != '\0' && *e != '0';
  }
  if (disabled)
    return;

  WOpt o;
  memset(&o, 0, sizeof(o));
  o.code = code->buf;
  o.size = code->len;
  o.nparams = nparams;

  // The locals' declarations
  unsigned char *p = o.code;
  uint32_t nruns = read_uleb128(p, &p);
  uint32_t nl = nparams;
  unsigned char *q = p;
  for (uint32_t i = 0; i < nruns; ++i) {
    nl += read_uleb128(q, &q);
    ++q;
  }
  o.nlocals = nl;
  o.ltype = malloc(nl + 1);
  memcpy(o.ltype, ptypes, nparams);
  uint32_t k = nparams;
  for (uint32_t i = 0; i < nruns; ++i) {
    uint32_t cnt = read_uleb128(p, &p);
    unsigned char t = *p++;
    while (cnt-- > 0)
      o.ltype[k++] = t;
  }
  o.body = p - o.code;

  if (decode(&o, relocs)) {
    build_blocks(&o);
    static int tee = -1;
    if (tee < 0) {
      const char *e = getenv("WCC_TEE");
      tee = e != NULL && *e == '1';
    }
    liveness(&o);
    for (int i = 0; i < 4 && cleanup(&o, false); ++i)
      liveness(&o);
    coalesce(&o);
    liveness(&o);
    for (int i = 0; i < 4 && cleanup(&o, tee); ++i)
      liveness(&o);
    drop_empty_branches(&o);
    encode(&o, code, relocs);
  }

  free(o.live_in); free(o.live_out); free(o.use); free(o.def);
  free(o.bb_start); free(o.bb_of);
  free(o.wi); free(o.tgts); free(o.ltype);
}
