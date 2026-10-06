// FPC: precompiled prefixes.
//
// A source file starts with its #includes, and preprocessing them is most
// of the work (NuttX's <stdio.h> is 90 files), the same each time for the
// same headers, options and directory.  The prefix of a file (its leading
// lines of directives, comments and blank lines, ending at #if depth 0) is
// preprocessed once and the result kept in the cache directory:
//   - the macros defined at its end, the include guards and #pragma once
//     files, the files read (their size and time, and a sum of the
//     contents of those not under an -isystem directory);
//   - its output split into top-level declarations, with the names each
//     declares and those it uses.
// The same prefix again (the same text, options, working and source
// directories) with the same files restores that state, and only the rest
// of the file is preprocessed.  With pch_prune the declarations of the
// system headers (-isystem) are output only when the rest of the file uses
// a name they declare (transitively): a hello world needs 2 of NuttX's
// 1261.  The output says so on its first line ("# pruned"): cc1 then stops
// at the first error with exit code 2, for the driver to try again without
// pruning (a declaration missed is an error, never different code: there
// are no implicit declarations in xcc).

#include "../config.h"
#include "pch.h"

#include <assert.h>
#include <ctype.h>
#include <libgen.h>  // dirname
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "lexer.h"
#include "macro.h"
#include "preprocessor.h"
#include "table.h"
#include "util.h"

#define PCH_MAGIC    0x48435058  // "XPCH"
#define PCH_VERSION  3
#define TOKSEP       '\x01'      // Between a macro's tokens

const char *pch_dir;
bool pch_prune = true;
bool pch_trust_system;
int pch_stats;

extern Lexer lexer;

// --pch-stats (or XCC_PCH_STATS set): what is done, and when
static void stamp(const char *what) {
  static struct timespec t0;
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  if (t0.tv_sec == 0 && t0.tv_nsec == 0)
    t0 = t;
  if (pch_stats)
    fprintf(stderr, "pch: %6ld ms %s\n",
            (long)((t.tv_sec - t0.tv_sec) * 1000 + (t.tv_nsec - t0.tv_nsec) / 1000000), what);
}

// Byte buffers, written and read: numbers in LEB128, strings with their
// length before and a '\0' after

typedef struct {
  char *p;
  size_t len, cap;
} Buf;

static void b_put(Buf *b, const void *d, size_t n) {
  if (b->len + n > b->cap) {
    size_t cap = b->cap != 0 ? b->cap * 2 : 1024;
    while (cap < b->len + n)
      cap *= 2;
    b->p = realloc_or_die(b->p, cap);
    b->cap = cap;
  }
  memcpy(b->p + b->len, d, n);
  b->len += n;
}

static void b_u32(Buf *b, uint32_t v) { b_put(b, &v, sizeof(v)); }

static void b_uv(Buf *b, uint32_t v) {
  unsigned char c[5];
  int n = 0;
  do {
    c[n] = v & 0x7f;
    v >>= 7;
    if (v != 0)
      c[n] |= 0x80;
    ++n;
  } while (v != 0);
  b_put(b, c, n);
}

static void b_str(Buf *b, const char *s, size_t n) {
  b_uv(b, n);
  b_put(b, s, n);
  b_put(b, "", 1);
}

static void b_cstr(Buf *b, const char *s) { b_str(b, s, strlen(s)); }

static uint32_t r_u32(const char **pp) {
  uint32_t v;
  memcpy(&v, *pp, sizeof(v));
  *pp += sizeof(v);
  return v;
}

static uint32_t r_uv(const char **pp) {
  const unsigned char *p = (const unsigned char*)*pp;
  uint32_t v = 0;
  for (int shift = 0;; shift += 7) {
    unsigned char c = *p++;
    v |= (uint32_t)(c & 0x7f) << shift;
    if (!(c & 0x80))
      break;
  }
  *pp = (const char*)p;
  return v;
}

static const char *r_str(const char **pp, uint32_t *pn) {
  uint32_t n = r_uv(pp);
  const char *s = *pp;
  *pp += n + 1;
  if (pn != NULL)
    *pn = n;
  return s;
}

static uint32_t fnv32(const char *s, size_t n, uint32_t h) {
  for (size_t i = 0; i < n; ++i)
    h = (h ^ (unsigned char)s[i]) * 16777619u;
  return h;
}

// Strings to indices (open addressing)

typedef struct {
  const char **keys;
  uint32_t *lens;
  int *vals;
  int cap, count;
} StrMap;

static int strmap_find(StrMap *m, const char *s, size_t n, uint32_t *pslot) {
  if (m->cap == 0) {
    *pslot = 0;
    return -1;
  }
  uint32_t i = fnv32(s, n, 2166136261u) & (m->cap - 1);
  for (;; i = (i + 1) & (m->cap - 1)) {
    if (m->keys[i] == NULL) {
      *pslot = i;
      return -1;
    }
    if (m->lens[i] == n && memcmp(m->keys[i], s, n) == 0) {
      *pslot = i;
      return m->vals[i];
    }
  }
}

static int strmap_get(StrMap *m, const char *s, size_t n) {
  uint32_t slot;
  return strmap_find(m, s, n, &slot);
}

static void strmap_put(StrMap *m, const char *s, size_t n, int v) {
  if ((m->count + 1) * 2 > m->cap) {
    StrMap old = *m;
    m->cap = old.cap != 0 ? old.cap * 2 : 256;
    m->keys = calloc_or_die(m->cap * sizeof(*m->keys));
    m->lens = malloc_or_die(m->cap * sizeof(*m->lens));
    m->vals = malloc_or_die(m->cap * sizeof(*m->vals));
    m->count = 0;
    for (int i = 0; i < old.cap; ++i) {
      if (old.keys[i] != NULL)
        strmap_put(m, old.keys[i], old.lens[i], old.vals[i]);
    }
    free(old.keys);
    free(old.lens);
    free(old.vals);
  }
  uint32_t slot;
  if (strmap_find(m, s, n, &slot) >= 0) {
    m->vals[slot] = v;
    return;
  }
  m->keys[slot] = s;
  m->lens[slot] = n;
  m->vals[slot] = v;
  ++m->count;
}

// The key: the options, the directories and the prefix's text

static Buf key;

void pch_set_args(int argc, char *argv[]) {
  for (int i = 0; i < argc; ++i) {
    b_put(&key, argv[i], strlen(argv[i]));
    b_put(&key, "\n", 1);
  }
}

static bool is_system_path(const char *path) {
  static Vector sysdirs;  // <char*> full paths
  static bool init;
  if (!init) {
    init = true;
    char *cwd = getcwd(NULL, 0);
    for (int ord = INC_SYSTEM; ord <= INC_AFTER; ++ord) {
      Vector *v = pp_inc_paths(ord);
      for (int i = 0; i < v->len; ++i)
        vec_push(&sysdirs, JOIN_PATHS(cwd, v->data[i]));
    }
  }
  for (int i = 0; i < sysdirs.len; ++i) {
    const char *d = sysdirs.data[i];
    size_t n = strlen(d);
    if (strncmp(path, d, n) == 0 && path[n] == '/')
      return true;
  }
  return false;
}

static uint32_t file_sum(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL)
    return 0;
  char buf[512];
  uint32_t h = 2166136261u;
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), fp)) > 0)
    h = fnv32(buf, n, h);
  fclose(fp);
  return h | 1;  // never 0 (0: not summed)
}

// The prefix: directives, comments, blank lines

// The rest of a line is blank or comments (a block comment may go on)
static bool rest_blank(const char *p, bool *incomment) {
  for (;;) {
    p = skip_whitespaces(p);
    if (*p == '\0' || (p[0] == '/' && p[1] == '/'))
      return true;
    if (p[0] != '/' || p[1] != '*')
      return false;
    const char *q = block_comment_end(p + 2);
    if (q == NULL) {
      *incomment = true;
      return true;
    }
    p = q;
  }
}

static bool word(const char *p, const char *w) {
  size_t n = strlen(w);
  return strncmp(p, w, n) == 0 && !isalnum_(p[n]);
}

// The prefix's lines (0: none worth it) and its text (into the key)
static int scan_prefix(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (fp == NULL)
    return 0;
  Buf text = {0};
  size_t keep_len = 0;
  int prefix = 0, depth = 0, lineno = 0;
  bool incomment = false, has_include = false, keep_include = false, local = false;
  char *line = NULL;
  size_t capa = 0;
  for (;;) {
    ssize_t len = getline_cont(&line, &capa, fp, &lineno);
    if (len == -1)
      break;
    const char *p = line;
    bool directive = false;
    if (incomment) {
      const char *q = block_comment_end(p);
      if (q != NULL) {
        incomment = false;
        if (!rest_blank(q, &incomment))
          break;
      }
    } else {
      p = skip_whitespaces(p);
      if (*p == '#') {
        directive = true;
        const char *d = skip_whitespaces(p + 1);
        // A block comment left open on a directive's line: the end
        for (const char *q = d; (q = strstr(q, "/*")) != NULL; ) {
          q = strstr(q + 2, "*/");
          if (q == NULL)
            goto end;
          q += 2;
        }
        if (word(d, "include") || word(d, "include_next")) {
          has_include = true;
          const char *q = skip_whitespaces(d + (word(d, "include") ? 7 : 12));
          if (*q != '<')
            local = true;  // "file" (or a macro): from the source's directory
        } else if (word(d, "if") || word(d, "ifdef") || word(d, "ifndef")) {
          ++depth;
        } else if (word(d, "endif")) {
          if (--depth < 0)
            break;
        } else if (!(word(d, "elif") || word(d, "else") || word(d, "define") ||
                     word(d, "undef") || word(d, "pragma") || *d == '\0')) {
          break;  // #line, #error...: the end
        }
      } else if (!rest_blank(p, &incomment)) {
        break;
      }
    }
    if (directive) {  // The key: the directives (not the comments, not the lines)
      b_put(&text, line, len);
      b_put(&text, "\n", 1);
    }
    if (!incomment && depth == 0) {
      prefix = lineno;
      keep_len = text.len;
      keep_include = has_include;
    }
  }
end:
  fclose(fp);
  free(line);
  if (!keep_include) {
    free(text.p);
    return 0;
  }
  // The directories, if the files found depend on them: an include
  // "file" or an include path not absolute
  for (int ord = INC_NORMAL; ord <= INC_AFTER && !local; ++ord) {
    Vector *v = pp_inc_paths(ord);
    for (int i = 0; i < v->len; ++i) {
      if (*(const char*)v->data[i] != '/')
        local = true;
    }
  }
  b_put(&key, "\n", 1);
  if (local) {
    char *cwd = getcwd(NULL, 0);
    char *dir = dirname(strdup(filename));
    b_cstr(&key, cwd);
    b_cstr(&key, dir);
  }
  b_put(&key, text.p, keep_len);
  free(text.p);
  return prefix;
}

// The declarations of the prefix's output

typedef struct {
  uint32_t off, len, file, line, flags;
  int ndefs, nuses;
  int *defs, *uses;
} Decl;

#define DF_SYSTEM  1  // From a system header: may be left out

typedef struct {
  uint32_t nnames, nfiles, ndecls;
  StrMap names;
  const char **files;
  Decl *decls;
  const char *text;  // NULL: still in the file, from text_pos
  FILE *fp;
  long text_pos;
  int *defs_of_start;  // CSR: the declarations declaring each name
  int *defs_of;
} DeclTable;

static DeclTable *decl_table;

// The cache file being made, or read
static char *cache_path;
static Vector deps;  // <char*>

// Declarations split, and their names

enum {
  KW_NONE,
  KW_OTHER,
  KW_STRUCT,  // struct, union
  KW_ENUM,
  KW_ATTR,    // __attribute__, __asm
};

static StrMap keywords;

static void init_keywords(void) {
  static const char *kOther[] = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
    "extern", "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict",
    "return", "short", "signed", "sizeof", "static", "switch", "typedef", "unsigned", "void",
    "volatile", "while", "_Bool", "_Complex", "_Imaginary", "_Noreturn", "_Alignas", "_Alignof",
    "_Atomic", "_Generic", "_Static_assert", "_Thread_local", "__inline", "__inline__",
    "__restrict", "__restrict__", "__volatile__", "__const", "__signed", "__signed__",
    "__extension__", "__typeof__", "__typeof", "typeof",
  };
  for (size_t i = 0; i < sizeof(kOther) / sizeof(*kOther); ++i)
    strmap_put(&keywords, kOther[i], strlen(kOther[i]), KW_OTHER);
  strmap_put(&keywords, "struct", 6, KW_STRUCT);
  strmap_put(&keywords, "union", 5, KW_STRUCT);
  strmap_put(&keywords, "enum", 4, KW_ENUM);
  static const char *kAttr[] = {"__attribute__", "__attribute", "__asm", "__asm__", "asm"};
  for (size_t i = 0; i < sizeof(kAttr) / sizeof(*kAttr); ++i)
    strmap_put(&keywords, kAttr[i], strlen(kAttr[i]), KW_ATTR);
}

typedef struct {
  char kind;  // 'i' identifier, 'n' number, 's' string, 'E' other operator, or the char
  int kw;
  int name;
} PTok;

typedef struct {
  StrMap names, files;
  Buf name_strs, file_strs;  // b_str'd
  Vector file_names;  // <const char*> in the text
  Vector file_lens;
  int nnames, nfiles;
  Buf decls;  // off len file line flags ndefs nuses defs... uses... (b_uv)
  Buf text;   // The declarations' text
  int ndecls;
  int *seen;  // per name: the declaration that used it last (+1)
  int seen_cap;
} Build;

static int build_name(Build *bd, const char *s, size_t n) {
  int i = strmap_get(&bd->names, s, n);
  if (i < 0) {
    i = bd->nnames++;
    strmap_put(&bd->names, s, n, i);
    b_str(&bd->name_strs, s, n);
    if (i >= bd->seen_cap) {
      int cap = bd->seen_cap != 0 ? bd->seen_cap * 2 : 1024;
      bd->seen = realloc_or_die(bd->seen, cap * sizeof(int));
      memset(bd->seen + bd->seen_cap, 0, (cap - bd->seen_cap) * sizeof(int));
      bd->seen_cap = cap;
    }
  }
  return i;
}

static int build_file(Build *bd, const char *s, size_t n) {
  int i = strmap_get(&bd->files, s, n);
  if (i < 0) {
    i = bd->nfiles++;
    strmap_put(&bd->files, s, n, i);
    b_str(&bd->file_strs, s, n);
    char *z = strndup(s, n);
    vec_push(&bd->file_names, z);
    vec_push(&bd->file_lens, INT2VOIDP(is_system_path(z) ? DF_SYSTEM : 0));
  }
  return i;
}

// The prefix's output split; -1: something not ended
static int split_decls(Build *bd, const char *text, size_t size) {
  const char *p = text, *end = text + size;
  int line = 1, file = -1;
  bool bol = true;

  bool indecl = false, fnbody = false;
  const char *dstart = NULL;
  int dfile = 0, dline = 0, paren = 0, brace = 0, enum_level = 0;
  PTok prev1 = {0}, prev2 = {0};
  struct {
    bool valid;
    int name, paren, brace, enum_level;
    PTok prev1, prev2;
  } pend = {0};
  Vector defs, uses;  // <int> of the declaration
  vec_init(&defs);
  vec_init(&uses);

  while (p < end) {
    char c = *p;
    if (c == '\n') {
      ++line;
      ++p;
      bol = true;
      continue;
    }
    if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v') {
      ++p;
      continue;
    }
    if (bol && c == '#') {  // Linemarker
      const char *q = skip_whitespaces(p + 1);
      char *e;
      long num = strtol(q, &e, 10);
      const char *eol = memchr(p, '\n', end - p);
      if (eol == NULL)
        eol = end;
      if (e > q) {
        q = strchr(e, '"');
        if (q != NULL && q < eol) {
          const char *qe = memchr(q + 1, '"', eol - (q + 1));
          if (qe != NULL)
            file = build_file(bd, q + 1, qe - (q + 1));
        }
        line = num - 1;
      }
      p = eol;
      continue;
    }
    bol = false;

    PTok t = {0};
    const char *ts = p;
    if (isalpha((unsigned char)c) || c == '_') {
      while (p < end && isalnum_((unsigned char)*p))
        ++p;
      t.kind = 'i';
      t.kw = strmap_get(&keywords, ts, p - ts);
      if (t.kw < 0) {
        t.kw = KW_NONE;
        t.name = build_name(bd, ts, p - ts);
      }
    } else if (isdigit((unsigned char)c) ||
               (c == '.' && p + 1 < end && isdigit((unsigned char)p[1]))) {
      while (p < end && (isalnum_((unsigned char)*p) || *p == '.' ||
                         ((*p == '+' || *p == '-') && strchr("eEpP", p[-1]) != NULL)))
        ++p;
      t.kind = 'n';
    } else if (c == '"' || c == '\'') {
      for (++p; p < end && *p != c && *p != '\n'; ++p) {
        if (*p == '\\' && p + 1 < end)
          ++p;
      }
      if (p < end && *p == c)
        ++p;
      t.kind = 's';
    } else {
      ++p;
      t.kind = c;
      if (p < end && ((*p == '=' && strchr("!<>=+-*/%&|^", c) != NULL) ||
                      (c == '-' && *p == '>'))) {
        ++p;
        t.kind = 'E';
      }
    }

    if (!indecl) {
      indecl = true;
      dstart = ts;
      dfile = file;
      dline = line;
      fnbody = false;
      paren = brace = enum_level = 0;
      vec_clear(&defs);
      vec_clear(&uses);
      prev1.kind = prev2.kind = 0;
    }

    if (pend.valid) {  // Whether the identifier before is declared
      bool def = false;
      bool attr_next = t.kind == 'i' && t.kw == KW_ATTR;
      if (pend.prev1.kind == 'i' && (pend.prev1.kw == KW_STRUCT || pend.prev1.kw == KW_ENUM) &&
          (t.kind == '{' || (t.kind == ';' && pend.paren == 0 && pend.brace == 0)))
        def = true;  // A tag
      else if (pend.paren == 0 && pend.brace == 0 &&
               (strchr("(;,=[{", t.kind) != NULL || attr_next))
        def = true;  // What a declaration declares
      else if (pend.paren == 1 && pend.brace == 0 && pend.prev1.kind == '*' &&
               pend.prev2.kind == '(' && strchr(")([", t.kind) != NULL)
        def = true;  // (*name)
      else if (pend.enum_level > 0 && pend.brace == pend.enum_level && pend.paren == 0 &&
               (pend.prev1.kind == '{' || pend.prev1.kind == ',') &&
               strchr("=,}", t.kind) != NULL)
        def = true;  // An enumerator
      if (def)
        vec_push(&defs, INT2VOIDP(pend.name));
      pend.valid = false;
    }

    bool done = false;
    switch (t.kind) {
    case 'i':
      if (t.kw == KW_NONE) {
        int ndecl = bd->ndecls + 1;
        if (bd->seen[t.name] != ndecl) {
          bd->seen[t.name] = ndecl;
          vec_push(&uses, INT2VOIDP(t.name));
        }
        pend.valid = true;
        pend.name = t.name;
        pend.paren = paren;
        pend.brace = brace;
        pend.enum_level = enum_level;
        pend.prev1 = prev1;
        pend.prev2 = prev2;
      }
      break;
    case '(': case '[':
      ++paren;
      break;
    case ')': case ']':
      --paren;
      break;
    case '{':
      if (brace == 0 && paren == 0 && prev1.kind == ')')
        fnbody = true;
      if ((prev1.kind == 'i' && prev1.kw == KW_ENUM) ||
          (prev1.kind == 'i' && prev2.kind == 'i' && prev2.kw == KW_ENUM))
        enum_level = brace + 1;
      ++brace;
      break;
    case '}':
      --brace;
      if (enum_level > 0 && brace < enum_level)
        enum_level = 0;
      if (brace == 0 && fnbody)
        done = true;
      break;
    case ';':
      if (brace == 0 && paren == 0)
        done = true;
      break;
    }
    prev2 = prev1;
    prev1 = t;
    if (paren < 0 || brace < 0)
      return -1;

    if (done) {
      if (dfile < 0)
        return -1;
      Buf *b = &bd->decls;
      b_uv(b, bd->text.len);
      b_uv(b, p - dstart);
      b_uv(b, dfile);
      b_uv(b, dline);
      b_uv(b, VOIDP2INT(bd->file_lens.data[dfile]));
      b_uv(b, defs.len);
      b_uv(b, uses.len);
      for (int i = 0; i < defs.len; ++i)
        b_uv(b, VOIDP2INT(defs.data[i]));
      for (int i = 0; i < uses.len; ++i)
        b_uv(b, VOIDP2INT(uses.data[i]));
      b_put(&bd->text, dstart, p - dstart);
      ++bd->ndecls;
      indecl = false;
      pend.valid = false;
    }
  }
  return indecl ? -1 : 0;
}

// The table read from a blob (from the cache file, or just made)
static DeclTable *load_decls(const char **pp) {
  DeclTable *dt = calloc_or_die(sizeof(*dt));
  const char *p = *pp;
  dt->nnames = r_uv(&p);
  for (uint32_t i = 0; i < dt->nnames; ++i) {
    uint32_t n;
    const char *s = r_str(&p, &n);
    strmap_put(&dt->names, s, n, i);
  }
  dt->nfiles = r_uv(&p);
  dt->files = malloc_or_die((dt->nfiles + 1) * sizeof(*dt->files));
  for (uint32_t i = 0; i < dt->nfiles; ++i)
    dt->files[i] = r_str(&p, NULL);
  dt->ndecls = r_uv(&p);
  uint32_t nints = r_uv(&p);
  dt->decls = malloc_or_die((dt->ndecls + 1) * sizeof(*dt->decls));
  int *ints = malloc_or_die((nints + 1) * sizeof(int));
  dt->defs_of_start = calloc_or_die((dt->nnames + 2) * sizeof(int));
  for (uint32_t i = 0; i < dt->ndecls; ++i) {
    Decl *d = &dt->decls[i];
    d->off = r_uv(&p);
    d->len = r_uv(&p);
    d->file = r_uv(&p);
    d->line = r_uv(&p);
    d->flags = r_uv(&p);
    d->ndefs = r_uv(&p);
    d->nuses = r_uv(&p);
    d->defs = ints;
    for (int j = 0; j < d->ndefs; ++j) {
      int nm = r_uv(&p);
      *ints++ = nm;
      ++dt->defs_of_start[nm + 1];
    }
    d->uses = ints;
    for (int j = 0; j < d->nuses; ++j)
      *ints++ = r_uv(&p);
  }
  for (uint32_t i = 0; i < dt->nnames; ++i)
    dt->defs_of_start[i + 1] += dt->defs_of_start[i];
  dt->defs_of = malloc_or_die((dt->defs_of_start[dt->nnames] + 1) * sizeof(int));
  int *fill = malloc_or_die((dt->nnames + 1) * sizeof(int));
  memcpy(fill, dt->defs_of_start, dt->nnames * sizeof(int));
  for (uint32_t i = 0; i < dt->ndecls; ++i) {
    const Decl *d = &dt->decls[i];
    for (int j = 0; j < d->ndefs; ++j)
      dt->defs_of[fill[d->defs[j]]++] = i;
  }
  free(fill);
  *pp = p;
  return dt;
}

// Macros

static bool write_macro(Buf *b, const Name *name, Macro *m) {
  b_str(b, name->chars, name->bytes);
  b_uv(b, (uint32_t)(m->params_len + 1));
  if (m->params_len > 0) {
    const Name **params = calloc_or_die(m->params_len * sizeof(*params));
    const Name *pn;
    void *v;
    for (int it = 0; (it = table_iterate(m->param_table, it, &pn, &v)) != -1; ) {
      intptr_t idx = VOIDP2INT(v);
      if (idx >= 0 && idx < m->params_len)
        params[idx] = pn;
    }
    for (int i = 0; i < m->params_len; ++i) {
      if (params[i] == NULL)
        return false;
      b_str(b, params[i]->chars, params[i]->bytes);
    }
    free(params);
  }
  if (m->vaargs_ident != NULL)
    b_str(b, m->vaargs_ident->chars, m->vaargs_ident->bytes);
  else
    b_str(b, "", 0);
  if (m->body == NULL) {
    b_uv(b, 0);
    return true;
  }
  // The tokens' texts between TOKSEPs, a space as nothing
  b_uv(b, m->body->len + 1);
  Buf t = {0};
  for (int i = 0; i < m->body->len; ++i) {
    const Token *tok = m->body->data[i];
    if (i > 0)
      b_put(&t, (char[]){TOKSEP}, 1);
    if (tok->kind != PPTK_SPACE) {
      size_t n = tok->end - tok->begin;
      if (n == 0 || memchr(tok->begin, TOKSEP, n) != NULL || memchr(tok->begin, '\0', n) != NULL)
        return false;
      b_put(&t, tok->begin, n);
    }
  }
  b_str(b, t.p != NULL ? t.p : "", t.len);
  free(t.p);
  return true;
}

static const char *skip_macro(const char *p) {
  int32_t plen = (int32_t)r_uv(&p) - 1;
  for (int32_t i = 0; i < plen; ++i)
    r_str(&p, NULL);
  r_str(&p, NULL);
  if (r_uv(&p) != 0)
    r_str(&p, NULL);
  return p;
}

void pch_materialize(Macro *m) {
  const char *p = m->lazy;
  m->lazy = NULL;
  int32_t plen = (int32_t)r_uv(&p) - 1;
  Vector *params = NULL;
  if (plen >= 0) {
    params = new_vector();
    for (int32_t i = 0; i < plen; ++i) {
      uint32_t n;
      const char *s = r_str(&p, &n);
      vec_push(params, alloc_name(s, s + n, false));
    }
  }
  uint32_t n;
  const char *va = r_str(&p, &n);
  const Name *vaargs = n > 0 ? alloc_name(va, va + n, false) : NULL;
  uint32_t ntok = r_uv(&p);
  Vector *body = NULL;
  if (ntok != 0) {
    --ntok;
    const char *s = r_str(&p, &n);
    body = new_vector();
    Lexer saved = lexer;
    TokenPoolMark mark = token_pool_enter(false);
    Token *space = NULL;
    for (uint32_t i = 0; i < ntok; ++i) {
      const char *e = strchr(s, TOKSEP);
      if (e == NULL)
        e = s + strlen(s);
      if (e == s) {
        if (space == NULL)
          space = alloc_token(PPTK_SPACE, NULL, " ", NULL);
        vec_push(body, space);
      } else {
        // The token by itself, to lex
        char *z = strndup(s, e - s);
        set_source_string(z, "<pch>", 0);
        vec_push(body, match(-1));
      }
      s = *e != '\0' ? e + 1 : e;
    }
    token_pool_leave(mark);
    lexer = saved;
  }
  Macro *made = new_macro(params, vaargs, body);
  *m = *made;
}

// Before the main file: the prefix; restored from the cache if it is there
int pch_begin(const char *filename, bool *phit) {
  *phit = false;
  if (pch_dir == NULL)
    return 0;
  if (getenv("XCC_PCH_STATS") != NULL)
    pch_stats = 1;
  stamp("start");
  int prefix = scan_prefix(filename);
  stamp("prefix scanned");
  if (prefix <= 0)
    return 0;

  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < key.len; ++i)
    h = (h ^ (unsigned char)key.p[i]) * 1099511628211ull;
  char name[32];
  snprintf(name, sizeof(name), "%08x%08x.pch", (unsigned)(h >> 32), (unsigned)h);
  cache_path = JOIN_PATHS(pch_dir, name);

  // Header: magic, version, the size of what is read whole, the key
  FILE *fp = fopen(cache_path, "rb");
  if (fp == NULL)
    return prefix;
  char hdr[12];
  const char *p = hdr;
  uint32_t head_size;
  if (fread(hdr, 1, sizeof(hdr), fp) != sizeof(hdr) || r_u32(&p) != PCH_MAGIC ||
      r_u32(&p) != PCH_VERSION || (head_size = r_u32(&p)) < sizeof(hdr) ||
      head_size > (64u << 20)) {
    fclose(fp);
    return prefix;
  }
  char *blob = malloc_or_die(head_size);
  if (fread(blob + sizeof(hdr), 1, head_size - sizeof(hdr), fp) != head_size - sizeof(hdr)) {
    free(blob);
    fclose(fp);
    return prefix;
  }
  p = blob + sizeof(hdr);
  uint32_t n;
  const char *k = r_str(&p, &n);
  if (n != key.len || memcmp(k, key.p, n) != 0) {
    free(blob);
    fclose(fp);
    return prefix;
  }

  stamp("head read");
  // The files read, as they were?
  uint32_t ndeps = r_uv(&p);
  for (uint32_t i = 0; i < ndeps; ++i) {
    const char *path = r_str(&p, NULL);
    uint32_t size = r_uv(&p);
    uint32_t mtime = r_u32(&p);
    uint32_t sum = r_u32(&p);
    struct stat ds;
    if (sum == 0 && pch_trust_system)
      continue;  // A system header: the key says which (--pch-key)
    if (stat(path, &ds) != 0 || (uint32_t)ds.st_size != size || (uint32_t)ds.st_mtime != mtime ||
        (sum != 0 && file_sum(path) != sum)) {
      if (pch_stats)
        fprintf(stderr, "pch: %s changed\n", path);
      free(blob);
      fclose(fp);
      return prefix;
    }
  }

  stamp("files checked");
  // The state at the prefix's end
  macro_clear();
  uint32_t nmacros = r_uv(&p);
  for (uint32_t i = 0; i < nmacros; ++i) {
    const char *s = r_str(&p, &n);
    Macro *m = calloc_or_die(sizeof(*m));
    m->lazy = p;
    macro_add(alloc_name(s, s + n, false), m);
    p = skip_macro(p);
  }
  uint32_t nonce = r_uv(&p);
  for (uint32_t i = 0; i < nonce; ++i)
    pp_add_pragma_once(r_str(&p, NULL));
  uint32_t nguard = r_uv(&p);
  for (uint32_t i = 0; i < nguard; ++i) {
    const char *f = r_str(&p, NULL);
    const char *g = r_str(&p, &n);
    pp_add_guard(f, g, n);
  }
  stamp("macros");
  decl_table = load_decls(&p);
  stamp("declarations");
  decl_table->fp = fp;  // The declarations' text: read when known which
  decl_table->text_pos = head_size;
  if (pch_stats)
    fprintf(stderr, "pch: %s: %s, %d macros, %d declarations\n", filename, name, nmacros,
            decl_table->ndecls);
  *phit = true;
  return prefix;
}

void pch_add_dep(const char *path) {
  for (int i = 0; i < deps.len; ++i) {
    if (strcmp(deps.data[i], path) == 0)
      return;
  }
  vec_push(&deps, strdup(path));
}

// At the prefix's end (not in the cache): the cache file made
void pch_snapshot(const char *text, size_t size) {
  Build bd = {0};
  init_keywords();
  if (split_decls(&bd, text, size) != 0) {
    if (pch_stats)
      fprintf(stderr, "pch: the prefix ends inside a declaration\n");
    return;
  }

  Buf b = {0};
  b_u32(&b, PCH_MAGIC);
  b_u32(&b, PCH_VERSION);
  b_u32(&b, 0);  // The head's size, below
  b_str(&b, key.p, key.len);

  b_uv(&b, deps.len);
  for (int i = 0; i < deps.len; ++i) {
    const char *path = deps.data[i];
    struct stat st;
    if (stat(path, &st) != 0)
      return;
    b_cstr(&b, path);
    b_uv(&b, st.st_size);
    b_u32(&b, st.st_mtime);
    b_u32(&b, is_system_path(path) ? 0 : file_sum(path));
  }

  const Name *name;
  Macro *m;
  uint32_t nmacros = 0;
  Buf mb = {0};
  for (int it = 0; (it = macro_iterate(it, &name, &m)) != -1; ) {
    if (m == NULL || (name->bytes == 8 && (memcmp(name->chars, "__FILE__", 8) == 0 ||
                                           memcmp(name->chars, "__LINE__", 8) == 0)))
      continue;
    if (m->lazy != NULL)
      pch_materialize(m);
    if (!write_macro(&mb, name, m)) {
      if (pch_stats)
        fprintf(stderr, "pch: macro %.*s not kept\n", NAMES(name));
      return;
    }
    ++nmacros;
  }
  b_uv(&b, nmacros);
  b_put(&b, mb.p, mb.len);
  free(mb.p);

  Vector *once = pp_pragma_once_files();
  b_uv(&b, once->len);
  for (int i = 0; i < once->len; ++i)
    b_cstr(&b, once->data[i]);
  Vector *guards = pp_guarded_files();
  b_uv(&b, guards->len / 2);
  for (int i = 0; i < guards->len; i += 2) {
    const Name *g = guards->data[i + 1];
    b_cstr(&b, guards->data[i]);
    b_str(&b, g->chars, g->bytes);
  }

  size_t decls_at = b.len;
  b_uv(&b, bd.nnames);
  b_put(&b, bd.name_strs.p, bd.name_strs.len);
  b_uv(&b, bd.nfiles);
  b_put(&b, bd.file_strs.p, bd.file_strs.len);
  b_uv(&b, bd.ndecls);
  b_uv(&b, bd.decls.len);  // At least the number of defs and uses
  b_put(&b, bd.decls.p, bd.decls.len);
  uint32_t head_size = b.len;
  memcpy(b.p + 8, &head_size, sizeof(head_size));

  char *tmp = malloc_or_die(strlen(cache_path) + 32);
  sprintf(tmp, "%s.%d.tmp", cache_path, (int)getpid());
  FILE *fp = fopen(tmp, "wb");
  if (fp != NULL) {
    bool ok = fwrite(b.p, 1, b.len, fp) == b.len;
    ok = ok && fwrite(bd.text.p, 1, bd.text.len, fp) == bd.text.len;
    ok = fclose(fp) == 0 && ok;
    remove(cache_path);
    if (!ok || rename(tmp, cache_path) != 0)
      remove(tmp);
  }
  free(tmp);

  const char *p = b.p + decls_at;
  decl_table = load_decls(&p);
  decl_table->text = bd.text.p;
  if (pch_stats)
    fprintf(stderr, "pch: made %s: %d macros, %d declarations, %d + %d bytes\n", cache_path,
            nmacros, bd.ndecls, (int)b.len, (int)bd.text.len);
}

// The output: the declarations needed, then the rest of the file
void pch_end(FILE *ofp, const char *prefix, size_t prefix_size, const char *body,
             size_t body_size) {
  DeclTable *dt = decl_table;
  stamp("end of the file");
  if (dt == NULL) {
    fwrite(prefix, 1, prefix_size, ofp);
    fwrite(body, 1, body_size, ofp);
    return;
  }

  char *keep = calloc_or_die(dt->ndecls + 1);
  int nkeep = 0;
  if (!pch_prune) {
    memset(keep, 1, dt->ndecls);
    nkeep = dt->ndecls;
  } else {
    char *needed = calloc_or_die(dt->nnames + 1);
    int *stack = malloc_or_die((dt->nnames + 1) * sizeof(int));
    int sp = 0;
#define NEED(i)  do { int i_ = (i); if (!needed[i_]) { needed[i_] = 1; stack[sp++] = i_; } } while (0)
    // The names the rest of the file has
    const char *p = body, *end = body + body_size;
    bool bol = true;
    while (p < end) {
      char c = *p;
      if (c == '\n') {
        bol = true;
        ++p;
        continue;
      }
      if (bol && c == '#') {
        const char *eol = memchr(p, '\n', end - p);
        p = eol != NULL ? eol : end;
        continue;
      }
      bol = false;
      if (isalpha((unsigned char)c) || c == '_') {
        const char *s = p;
        while (p < end && isalnum_((unsigned char)*p))
          ++p;
        int i = strmap_get(&dt->names, s, p - s);
        if (i >= 0)
          NEED(i);
      } else if (isdigit((unsigned char)c)) {
        while (p < end && (isalnum_((unsigned char)*p) || *p == '.'))
          ++p;
      } else if (c == '"' || c == '\'') {
        for (++p; p < end && *p != c && *p != '\n'; ++p) {
          if (*p == '\\' && p + 1 < end)
            ++p;
        }
        if (p < end && *p == c)
          ++p;
      } else {
        ++p;
      }
    }
    // What the declarations kept anyway use
    for (uint32_t i = 0; i < dt->ndecls; ++i) {
      const Decl *d = &dt->decls[i];
      if (!(d->flags & DF_SYSTEM)) {
        keep[i] = 1;
        ++nkeep;
        for (int j = 0; j < d->nuses; ++j)
          NEED(d->uses[j]);
      }
    }
    while (sp > 0) {
      int nm = stack[--sp];
      for (int k = dt->defs_of_start[nm]; k < dt->defs_of_start[nm + 1]; ++k) {
        int di = dt->defs_of[k];
        if (keep[di])
          continue;
        keep[di] = 1;
        ++nkeep;
        const Decl *d = &dt->decls[di];
        for (int j = 0; j < d->nuses; ++j)
          NEED(d->uses[j]);
      }
    }
#undef NEED
    free(needed);
    free(stack);
  }

  stamp("pruned");
  // Their text, from the file if it is not here
  const char *text = dt->text;
  char *buf = NULL;
  if (text == NULL) {
    size_t total = 0;
    for (uint32_t i = 0; i < dt->ndecls; ++i) {
      if (keep[i])
        total += dt->decls[i].len;
    }
    buf = malloc_or_die(total + 1);
    size_t pos = 0;
    for (uint32_t i = 0; i < dt->ndecls; ) {
      if (!keep[i]) {
        ++i;
        continue;
      }
      // A run of kept declarations: one read
      uint32_t j = i, len = 0;
      for (; j < dt->ndecls && keep[j]; ++j)
        len += dt->decls[j].len;
      if (fseek(dt->fp, dt->text_pos + dt->decls[i].off, SEEK_SET) != 0 ||
          fread(buf + pos, 1, len, dt->fp) != len)
        error("pch: cannot read %s", cache_path);
      for (; i < j; ++i) {
        dt->decls[i].off = pos;
        pos += dt->decls[i].len;
      }
    }
    text = buf;
  }

  if (nkeep < (int)dt->ndecls)
    fprintf(ofp, "# pruned\n");
  const char *curfile = NULL;
  uint32_t curline = 0;
  for (uint32_t i = 0; i < dt->ndecls; ++i) {
    if (!keep[i])
      continue;
    const Decl *d = &dt->decls[i];
    const char *file = dt->files[d->file];
    if (file != curfile || d->line < curline || d->line - curline > 8) {
      fprintf(ofp, "# %u \"%s\"\n", d->line, file);
    } else {
      for (; curline < d->line; ++curline)
        fputc('\n', ofp);
    }
    const char *s = text + d->off;
    fwrite(s, 1, d->len, ofp);
    fputc('\n', ofp);
    uint32_t nl = 0;
    for (const char *q = s; (q = memchr(q, '\n', s + d->len - q)) != NULL; ++q)
      ++nl;
    curfile = file;
    curline = d->line + nl + 1;
  }
  fwrite(body, 1, body_size, ofp);
  stamp("output");
  if (pch_stats)
    fprintf(stderr, "pch: %d of %d declarations\n", nkeep, dt->ndecls);
  free(keep);
  free(buf);
  if (dt->fp != NULL)
    fclose(dt->fp);
}
