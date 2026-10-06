// FPC: xcc's linker for ARM32: NuttX loadable modules.
//
// The objects (and the archive members they need) are linked into one
// relocatable ELF, as NuttX's ELF loader takes it (a "module": what
// `ld -r -e _start -T gnu-elf.ld` makes of gcc's objects): sections .text,
// .rodata, .data, .init_array, .fini_array, .bss, each placed by the loader
// where it likes, and the relocations it is to do: against the sections
// for what the module defines, against undefined symbols for what the
// firmware exports (its symbol table).  Calls are resolved here; a call to
// the firmware goes through a veneer (movw/movt ip, bx ip) at the end of
// .text, since the firmware is beyond a BL's reach.  e_entry is _start's
// offset in .text (Thumb).  nx_stacksize and nx_priority (--stack,
// --priority) are what the loader starts the task with.

#include "../../config.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "archive.h"
#include "table.h"
#include "util.h"

enum {
  SHT_PROGBITS_ = 1, SHT_SYMTAB_ = 2, SHT_STRTAB_ = 3, SHT_NOBITS_ = 8, SHT_REL_ = 9,
  SHT_INIT_ARRAY_ = 14, SHT_FINI_ARRAY_ = 15,
  SHF_WRITE_ = 1, SHF_ALLOC_ = 2, SHF_EXECINSTR_ = 4, SHF_INFO_LINK_ = 0x40,
  STB_LOCAL_ = 0, STB_GLOBAL_ = 1, STB_WEAK_ = 2,
  STT_NOTYPE_ = 0, STT_OBJECT_ = 1, STT_FUNC_ = 2, STT_SECTION_ = 3,
  SHN_UNDEF_ = 0, SHN_ABS_ = 0xfff1, SHN_COMMON_ = 0xfff2,
  R_ARM_NONE_ = 0, R_ARM_ABS32_ = 2, R_ARM_THM_CALL_ = 10, R_ARM_THM_JUMP24_ = 30,
  R_ARM_THM_MOVW_ABS_NC_ = 47, R_ARM_THM_MOVT_ABS_ = 48, R_ARM_V4BX_ = 40,
};

// Output sections
enum { OS_TEXT, OS_RODATA, OS_DATA, OS_INIT, OS_FINI, OS_BSS, OS_COUNT };
static const char *kOutNames[OS_COUNT] = {
  ".text", ".rodata", ".data", ".init_array", ".fini_array", ".bss",
};

typedef struct {
  unsigned char *buf;
  size_t len, cap;
} Buf;

static void buf_add(Buf *b, const void *p, size_t n) {
  if (b->len + n > b->cap) {
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < b->len + n)
      cap *= 2;
    b->buf = realloc_or_die(b->buf, cap);
    b->cap = cap;
  }
  if (p != NULL)
    memcpy(b->buf + b->len, p, n);
  else
    memset(b->buf + b->len, 0, n);
  b->len += n;
}
static void buf_u8(Buf *b, uint32_t v) { unsigned char c = v; buf_add(b, &c, 1); }
static void buf_u16(Buf *b, uint32_t v) { buf_u8(b, v); buf_u8(b, v >> 8); }
static void buf_u32(Buf *b, uint32_t v) { buf_u16(b, v); buf_u16(b, v >> 16); }
static uint32_t buf_str(Buf *b, const char *s, size_t n) {
  uint32_t ofs = b->len;
  buf_add(b, s, n);
  buf_u8(b, 0);
  return ofs;
}

static uint32_t rd16(const unsigned char *p) { return p[0] | (p[1] << 8); }
static uint32_t rd32(const unsigned char *p) { return rd16(p) | (rd16(p + 2) << 16); }
static void wr16(unsigned char *p, uint32_t v) { p[0] = v; p[1] = v >> 8; }
static void wr32(unsigned char *p, uint32_t v) { wr16(p, v); wr16(p + 2, v >> 16); }

typedef struct {
  uint32_t name, type, flags, addr, offset, size, link, info, align, entsize;
} Shdr;

typedef struct {
  uint32_t name, value, size;
  unsigned char info, other;
  uint16_t shndx;
} Sym;

typedef struct Obj {
  const char *name;
  unsigned char *data;
  size_t size;
  Shdr *shdrs;
  int shnum;
  Sym *syms;
  int nsyms;
  const char *strtab;
  int *out_sec;      // [shnum] output section of an input section, or -1
  uint32_t *out_ofs; // [shnum] its offset there
} Obj;

// A global symbol: defined (obj/section/value) or not
typedef struct {
  const Name *name;
  Obj *obj;
  int shndx;
  uint32_t value;
  int type;
  int bind;
  bool common;
  uint32_t common_size, common_align;
  // Linker-defined (section boundaries): os >= 0
  int os;
  uint32_t os_ofs;
} Global;

static Table g_globals;    // <Global*>
static Vector *g_objs;     // <Obj*>
static Buf g_out[OS_COUNT];
static size_t g_bss_size;
static uint32_t g_out_align[OS_COUNT];

static Obj *read_obj(const char *name, unsigned char *data, size_t size) {
  if (size < 52 || memcmp(data, "\x7f" "ELF\x01\x01", 6) != 0 || rd16(data + 16) != 1 ||
      rd16(data + 18) != 40) {
    fprintf(stderr, "%s: not an ARM relocatable ELF32 object\n", name);
    return NULL;
  }
  Obj *obj = calloc_or_die(sizeof(*obj));
  obj->name = name;
  obj->data = data;
  obj->size = size;
  uint32_t shoff = rd32(data + 32);
  obj->shnum = rd16(data + 48);
  obj->shdrs = calloc_or_die(sizeof(Shdr) * obj->shnum);
  for (int i = 0; i < obj->shnum; ++i) {
    const unsigned char *p = data + shoff + 40 * i;
    Shdr *s = &obj->shdrs[i];
    s->name = rd32(p); s->type = rd32(p + 4); s->flags = rd32(p + 8); s->addr = rd32(p + 12);
    s->offset = rd32(p + 16); s->size = rd32(p + 20); s->link = rd32(p + 24);
    s->info = rd32(p + 28); s->align = rd32(p + 32); s->entsize = rd32(p + 36);
  }
  for (int i = 0; i < obj->shnum; ++i) {
    Shdr *s = &obj->shdrs[i];
    if (s->type == SHT_SYMTAB_) {
      obj->nsyms = s->size / 16;
      obj->syms = calloc_or_die(sizeof(Sym) * obj->nsyms);
      for (int k = 0; k < obj->nsyms; ++k) {
        const unsigned char *p = data + s->offset + 16 * k;
        Sym *y = &obj->syms[k];
        y->name = rd32(p); y->value = rd32(p + 4); y->size = rd32(p + 8);
        y->info = p[12]; y->other = p[13]; y->shndx = rd16(p + 14);
      }
      obj->strtab = (const char*)data + obj->shdrs[s->link].offset;
    }
  }
  obj->out_sec = calloc_or_die(sizeof(int) * obj->shnum);
  obj->out_ofs = calloc_or_die(sizeof(uint32_t) * obj->shnum);
  for (int i = 0; i < obj->shnum; ++i)
    obj->out_sec[i] = -1;
  return obj;
}

static const char *sym_name(Obj *obj, const Sym *s) {
  return obj->strtab + s->name;
}

static int out_section_of(Obj *obj, int shndx) {
  Shdr *s = &obj->shdrs[shndx];
  if (!(s->flags & SHF_ALLOC_))
    return -1;
  const char *shstr = (const char*)obj->data + obj->shdrs[rd16(obj->data + 50)].offset;
  const char *name = shstr + s->name;
  if (s->type == SHT_NOBITS_)
    return OS_BSS;
  if (s->type == SHT_INIT_ARRAY_ || strncmp(name, ".init_array", 11) == 0)
    return OS_INIT;
  if (s->type == SHT_FINI_ARRAY_ || strncmp(name, ".fini_array", 11) == 0)
    return OS_FINI;
  if (s->flags & SHF_EXECINSTR_)
    return OS_TEXT;
  if (s->flags & SHF_WRITE_)
    return OS_DATA;
  if (strncmp(name, ".ARM.exidx", 10) == 0 || strncmp(name, ".ARM.extab", 10) == 0)
    return -1;  // Unwinding tables: not used
  return OS_RODATA;
}

// Add an object's global definitions; false on a duplicate.
static bool define_globals(Obj *obj) {
  bool ok = true;
  for (int k = 1; k < obj->nsyms; ++k) {
    Sym *s = &obj->syms[k];
    int bind = s->info >> 4;
    if (bind == STB_LOCAL_ || s->shndx == SHN_UNDEF_)
      continue;
    const Name *name = alloc_name(sym_name(obj, s), NULL, false);
    Global *g = table_get(&g_globals, name);
    if (g != NULL && g->obj != NULL && !g->common) {
      if (bind == STB_WEAK_)
        continue;
      if (g->bind != STB_WEAK_ && !(s->shndx == SHN_COMMON_)) {
        fprintf(stderr, "duplicate symbol: %.*s (%s, %s)\n", NAMES(name), g->obj->name, obj->name);
        ok = false;
        continue;
      }
    }
    if (g == NULL) {
      g = calloc_or_die(sizeof(*g));
      g->name = name;
      g->os = -1;
      table_put(&g_globals, name, g);
    }
    if (s->shndx == SHN_COMMON_) {
      if (g->obj != NULL && !g->common)
        continue;  // A definition wins over a common
      g->common = true;
      if (s->size > g->common_size)
        g->common_size = s->size;
      if (s->value > g->common_align)
        g->common_align = s->value;
    } else {
      g->common = false;
    }
    g->obj = obj;
    g->shndx = s->shndx;
    g->value = s->value;
    g->type = s->info & 15;
    g->bind = bind;
  }
  return ok;
}

static bool is_undefined(const Name *name) {
  Global *g = table_get(&g_globals, name);
  return g == NULL || (g->obj == NULL && g->os < 0);
}

static void *load_member(FILE *fp, const char *name, size_t size) {
  unsigned char *data = malloc_or_die(size);
  if (fread(data, 1, size, fp) != size) {
    free(data);
    return NULL;
  }
  return read_obj(strdup(name), data, size);
}

// Undefined names referenced by an object
static void collect_undefined(Obj *obj, Vector *undef) {
  for (int k = 1; k < obj->nsyms; ++k) {
    Sym *s = &obj->syms[k];
    if ((s->info >> 4) != STB_LOCAL_ && s->shndx == SHN_UNDEF_ && s->name != 0)
      vec_push(undef, alloc_name(sym_name(obj, s), NULL, false));
  }
}

static bool link_archives(Vector *archives) {
  for (;;) {
    bool added = false;
    Vector *undef = new_vector();
    for (int i = 0; i < g_objs->len; ++i)
      collect_undefined(g_objs->data[i], undef);
    for (int i = 0; i < undef->len; ++i) {
      const Name *name = undef->data[i];
      if (!is_undefined(name))
        continue;
      for (int a = 0; a < archives->len; ++a) {
        Archive *ar = archives->data[a];
        ArSymbol *arsym;
        if (!table_try_get(&ar->symbol_table, name, (void**)&arsym))
          continue;
        if (arsym->content->obj != NULL)
          continue;
        Obj *obj = load_archive_content(ar, arsym, load_member);
        if (obj == NULL)
          return false;
        vec_push(g_objs, obj);
        if (!define_globals(obj))
          return false;
        added = true;
        break;
      }
    }
    if (!added)
      return true;
  }
}

// Place every allocated input section in its output section.
static void layout(void) {
  for (int os = 0; os < OS_COUNT; ++os)
    g_out_align[os] = 4;
  for (int i = 0; i < g_objs->len; ++i) {
    Obj *obj = g_objs->data[i];
    for (int k = 1; k < obj->shnum; ++k) {
      int os = out_section_of(obj, k);
      if (os < 0)
        continue;
      Shdr *s = &obj->shdrs[k];
      uint32_t align = s->align > 0 ? s->align : 1;
      if (align > g_out_align[os])
        g_out_align[os] = align;
      if (os == OS_BSS) {
        g_bss_size = ALIGN(g_bss_size, align);
        obj->out_ofs[k] = g_bss_size;
        g_bss_size += s->size;
      } else {
        Buf *b = &g_out[os];
        size_t ofs = ALIGN(b->len, align);
        buf_add(b, NULL, ofs - b->len);
        obj->out_ofs[k] = ofs;
        buf_add(b, obj->data + s->offset, s->size);
      }
      obj->out_sec[k] = os;
    }
  }
  // Commons in .bss
  const Name *name;
  Global *g;
  for (int it = 0; (it = table_iterate(&g_globals, it, &name, (void**)&g)) != -1; ) {
    if (!g->common)
      continue;
    uint32_t align = g->common_align > 0 ? g->common_align : 4;
    g_bss_size = ALIGN(g_bss_size, align);
    g->os = OS_BSS;
    g->os_ofs = g_bss_size;
    g->obj = NULL;
    g_bss_size += g->common_size;
  }
}

// Linker-defined section boundaries, if referenced and undefined
static void define_boundaries(void) {
  static const struct { const char *name; int os; bool end; } kDefs[] = {
    {"__init_array_start", OS_INIT, false}, {"__init_array_end", OS_INIT, true},
    {"__fini_array_start", OS_FINI, false}, {"__fini_array_end", OS_FINI, true},
    {"_sinit", OS_INIT, false}, {"_einit", OS_INIT, true},
    {"_stext", OS_TEXT, false}, {"_etext", OS_TEXT, true},
    {"_sbss", OS_BSS, false}, {"_ebss", OS_BSS, true},
  };
  for (size_t i = 0; i < ARRAY_SIZE(kDefs); ++i) {
    const Name *name = alloc_cname(kDefs[i].name);
    if (!is_undefined(name))
      continue;
    Global *g = table_get(&g_globals, name);
    if (g == NULL) {
      g = calloc_or_die(sizeof(*g));
      g->name = name;
      table_put(&g_globals, name, g);
    }
    g->os = kDefs[i].os;
    g->os_ofs = kDefs[i].end ? (kDefs[i].os == OS_BSS ? g_bss_size : g_out[kDefs[i].os].len) : 0;
    g->bind = STB_LOCAL_;
  }
}

// Where a relocation's symbol is: (output section, offset) or undefined (os -1, name)
typedef struct {
  int os;
  uint32_t ofs;      // with the Thumb bit for a function
  const Name *name;  // undefined
  int type;
} Target;

static Target resolve_target(Obj *obj, int symidx) {
  Target t = {.os = -1, .ofs = 0, .name = NULL, .type = STT_NOTYPE_};
  Sym *s = &obj->syms[symidx];
  int bind = s->info >> 4;
  t.type = s->info & 15;
  if (bind == STB_LOCAL_ || (s->shndx != SHN_UNDEF_ && s->shndx != SHN_COMMON_ &&
                             table_get(&g_globals, alloc_name(sym_name(obj, s), NULL, false)) ==
                                 NULL)) {
    if (s->shndx == SHN_ABS_) {
      t.os = -2;
      t.ofs = s->value;
      return t;
    }
    t.os = obj->out_sec[s->shndx];
    t.ofs = obj->out_ofs[s->shndx] + s->value;
    return t;
  }
  const Name *name = alloc_name(sym_name(obj, s), NULL, false);
  Global *g = table_get(&g_globals, name);
  if (g != NULL && g->os >= 0) {
    t.os = g->os;
    t.ofs = g->os_ofs;
    return t;
  }
  if (g != NULL && g->obj != NULL) {
    if (g->shndx == SHN_ABS_) {
      t.os = -2;
      t.ofs = g->value;
      return t;
    }
    t.os = g->obj->out_sec[g->shndx];
    t.ofs = g->obj->out_ofs[g->shndx] + g->value;
    t.type = g->type;
    return t;
  }
  t.name = name;
  return t;
}

// Output relocations and the symbols they need
typedef struct {
  uint32_t offset;
  int type;
  int sym;  // output symbol index
} OutRel;

static Vector *g_outrels[OS_COUNT];  // <OutRel*>
static Buf g_syms, g_strtab;
static int g_nsyms;
static Table g_undef_syms;  // name -> index
static Table g_local_syms;  // "os:ofs" -> index
static int g_secsym[OS_COUNT];

static int add_sym(uint32_t name, uint32_t value, uint32_t size, int bind, int type, int shndx) {
  buf_u32(&g_syms, name);
  buf_u32(&g_syms, value);
  buf_u32(&g_syms, size);
  buf_u8(&g_syms, (bind << 4) | type);
  buf_u8(&g_syms, 0);
  buf_u16(&g_syms, shndx);
  return g_nsyms++;
}

static int os_shndx(int os);

// Locals first: the section symbols, then symbols for movw/movt targets
// (their addends are only 16 bits), made as needed; the globals (undefined)
// are added after.
static Vector *g_pending_globals;  // <const Name*>

static int local_sym(int os, uint32_t ofs, int type) {
  char key[32];
  snprintf(key, sizeof(key), "%d:%u", os, (unsigned)ofs);
  const Name *name = alloc_name(key, NULL, true);
  void *p;
  if (table_try_get(&g_local_syms, name, &p))
    return (int)(intptr_t)p;
  char lname[32];
  snprintf(lname, sizeof(lname), ".L%s_%x", kOutNames[os] + 1, (unsigned)ofs);
  int idx = add_sym(buf_str(&g_strtab, lname, strlen(lname)), ofs, 0, STB_LOCAL_,
                    type == STT_FUNC_ ? STT_FUNC_ : STT_NOTYPE_, os_shndx(os));
  table_put(&g_local_syms, name, (void*)(intptr_t)idx);
  return idx;
}

static void add_outrel(int os, uint32_t offset, int type, int sym) {
  OutRel *r = calloc_or_die(sizeof(*r));
  r->offset = offset;
  r->type = type;
  r->sym = sym;
  if (g_outrels[os] == NULL)
    g_outrels[os] = new_vector();
  vec_push(g_outrels[os], r);
}

// An undefined symbol's index: global symbols must come after the locals,
// so they are given negative placeholders, fixed when written.
static int undef_sym(const Name *name) {
  void *p;
  if (table_try_get(&g_undef_syms, name, &p))
    return (int)(intptr_t)p;
  if (g_pending_globals == NULL)
    g_pending_globals = new_vector();
  vec_push(g_pending_globals, name);
  int idx = -g_pending_globals->len;  // fixed later
  table_put(&g_undef_syms, name, (void*)(intptr_t)idx);
  return idx;
}

// Thumb BL/B.W offset field
static int32_t decode_t4(const unsigned char *p) {
  uint32_t hw1 = rd16(p), hw2 = rd16(p + 2);
  uint32_t s = (hw1 >> 10) & 1, j1 = (hw2 >> 13) & 1, j2 = (hw2 >> 11) & 1;
  uint32_t i1 = (~(j1 ^ s)) & 1, i2 = (~(j2 ^ s)) & 1;
  int32_t off = (s << 24) | (i1 << 23) | (i2 << 22) | ((hw1 & 0x3ff) << 12) | ((hw2 & 0x7ff) << 1);
  if (off & 0x01000000)
    off -= 0x02000000;
  return off;
}

static void encode_t4(unsigned char *p, int32_t off) {
  uint32_t hw1 = rd16(p), hw2 = rd16(p + 2);
  uint32_t s = (off >> 24) & 1;
  uint32_t i1 = (off >> 23) & 1, i2 = (off >> 22) & 1;
  uint32_t j1 = (~(i1 ^ s)) & 1, j2 = (~(i2 ^ s)) & 1;
  wr16(p, (hw1 & 0xf800) | (s << 10) | ((off >> 12) & 0x3ff));
  wr16(p + 2, (hw2 & 0xd000) | (j1 << 13) | (j2 << 11) | ((off >> 1) & 0x7ff));
}

static uint32_t get_imm16(const unsigned char *p) {
  uint32_t hw1 = rd16(p), hw2 = rd16(p + 2);
  return ((hw1 & 0xf) << 12) | (((hw1 >> 10) & 1) << 11) | (((hw2 >> 12) & 7) << 8) | (hw2 & 0xff);
}

static void put_imm16(unsigned char *p, uint32_t u) {
  uint32_t hw1 = rd16(p), hw2 = rd16(p + 2);
  wr16(p, (hw1 & 0xfbf0) | ((u >> 12) & 0xf) | (((u >> 11) & 1) << 10));
  wr16(p + 2, (hw2 & 0x8f00) | (((u >> 8) & 7) << 12) | (u & 0xff));
}

// Veneers to the firmware: name -> offset in .text
static Table g_veneers;

static uint32_t veneer_for(const Name *name) {
  void *p;
  if (table_try_get(&g_veneers, name, &p))
    return (uint32_t)(intptr_t)p;
  Buf *t = &g_out[OS_TEXT];
  buf_add(t, NULL, ALIGN(t->len, 4) - t->len);
  uint32_t ofs = t->len;
  // movw ip, #:lower16:name; movt ip, #:upper16:name; bx ip; nop
  static const unsigned char kCode[] = {
    0x40, 0xf2, 0x00, 0x0c,  // movw ip, #0
    0xc0, 0xf2, 0x00, 0x0c,  // movt ip, #0
    0x60, 0x47,              // bx ip
    0x00, 0xbf,              // nop
  };
  buf_add(t, kCode, sizeof(kCode));
  int sym = undef_sym(name);
  add_outrel(OS_TEXT, ofs, R_ARM_THM_MOVW_ABS_NC_, sym);
  add_outrel(OS_TEXT, ofs + 4, R_ARM_THM_MOVT_ABS_, sym);
  table_put(&g_veneers, name, (void*)(intptr_t)ofs);
  return ofs;
}

static bool relocate(void) {
  bool ok = true;
  for (int i = 0; i < g_objs->len; ++i) {
    Obj *obj = g_objs->data[i];
    for (int k = 1; k < obj->shnum; ++k) {
      Shdr *rs = &obj->shdrs[k];
      if (rs->type != SHT_REL_)
        continue;
      int target_sec = rs->info;
      int os = obj->out_sec[target_sec];
      if (os < 0 || os == OS_BSS)
        continue;
      unsigned char *base = g_out[os].buf + obj->out_ofs[target_sec];
      for (uint32_t r = 0; r < rs->size / 8; ++r) {
        const unsigned char *rp = obj->data + rs->offset + 8 * r;
        uint32_t offset = rd32(rp), info = rd32(rp + 4);
        int type = info & 0xff, symidx = info >> 8;
        unsigned char *p = base + offset;
        uint32_t pos = obj->out_ofs[target_sec] + offset;  // in the output section
        Target t = resolve_target(obj, symidx);
        switch (type) {
        case R_ARM_NONE_: case R_ARM_V4BX_:
          break;
        case R_ARM_ABS32_:
          if (t.os == -2) {
            wr32(p, rd32(p) + t.ofs);
          } else if (t.os >= 0) {
            wr32(p, rd32(p) + t.ofs);
            add_outrel(os, pos, R_ARM_ABS32_, g_secsym[t.os]);
          } else {
            add_outrel(os, pos, R_ARM_ABS32_, undef_sym(t.name));
          }
          break;
        case R_ARM_THM_MOVW_ABS_NC_: case R_ARM_THM_MOVT_ABS_:
          if (t.os == -2) {
            uint32_t v = (int16_t)get_imm16(p) + t.ofs;
            put_imm16(p, type == R_ARM_THM_MOVT_ABS_ ? v >> 16 : v & 0xffff);
          } else if (t.os >= 0) {
            add_outrel(os, pos, type, local_sym(t.os, t.ofs, t.type));
          } else {
            add_outrel(os, pos, type, undef_sym(t.name));
          }
          break;
        case R_ARM_THM_CALL_: case R_ARM_THM_JUMP24_:
          {
            int32_t addend = decode_t4(p);
            if (t.os == OS_TEXT && os == OS_TEXT) {
              encode_t4(p, (int32_t)(t.ofs & ~1U) + addend - (int32_t)pos);
            } else if (t.os < 0 && t.name != NULL) {
              uint32_t v = veneer_for(t.name);
              base = g_out[os].buf + obj->out_ofs[target_sec];  // the buffer may have moved
              p = base + offset;
              encode_t4(p, (int32_t)v + addend - (int32_t)pos);
            } else {
              fprintf(stderr, "%s: call to a non-code section\n", obj->name);
              ok = false;
            }
          }
          break;
        default:
          fprintf(stderr, "%s: relocation type %d not supported\n", obj->name, type);
          ok = false;
          break;
        }
      }
    }
  }
  return ok;
}

static int os_shndx(int os) {
  // The output section indices: those with contents, in order, from 1
  int idx = 0;
  for (int i = 0; i <= os; ++i) {
    bool present = i == OS_BSS ? g_bss_size > 0 : (g_out[i].len > 0 || i == OS_TEXT);
    if (present)
      ++idx;
  }
  return idx;
}

static void usage(FILE *fp) {
  fprintf(fp,
      "Usage: ld [options] file...\n"
      "  -o FILE          the module (default a.out)\n"
      "  -e SYMBOL        the entry (default _start)\n"
      "  -L DIR, -lNAME   libraries (libNAME.a)\n"
      "  --stack=BYTES    the task's stack (nx_stacksize)\n"
      "  --priority=N     the task's priority (nx_priority)\n");
}

int main(int argc, char *argv[]) {
  const char *ofn = "a.out";
  const char *entry = "_start";
  long stacksize = 0, priority = 0;
  Vector *libdirs = new_vector();
  Vector *inputs = new_vector();
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (strcmp(a, "-o") == 0 && i + 1 < argc) ofn = argv[++i];
    else if (strncmp(a, "-o", 2) == 0 && a[2] != '\0') ofn = a + 2;
    else if (strcmp(a, "-e") == 0 && i + 1 < argc) entry = argv[++i];
    else if (strcmp(a, "-L") == 0 && i + 1 < argc) vec_push(libdirs, argv[++i]);
    else if (strncmp(a, "-L", 2) == 0) vec_push(libdirs, a + 2);
    else if (strncmp(a, "--stack=", 8) == 0) stacksize = strtol(a + 8, NULL, 0);
    else if (strncmp(a, "--priority=", 11) == 0) priority = strtol(a + 11, NULL, 0);
    else if (strncmp(a, "-l", 2) == 0) {
      bool found = false;
      for (int k = 0; k < libdirs->len && !found; ++k) {
        const char *dir = libdirs->data[k];
        char *path = malloc_or_die(strlen(dir) + strlen(a) + 8);
        sprintf(path, "%s/lib%s.a", dir, a + 2);
        if (is_file(path)) {
          vec_push(inputs, path);
          found = true;
        }
      }
      if (!found) {
        fprintf(stderr, "ld: library not found: %s\n", a);
        return 1;
      }
    } else if (strcmp(a, "-r") == 0 || strcmp(a, "--help") == 0) {
      if (a[2] == 'h') { usage(stdout); return 0; }
    } else if (a[0] == '-') {
      fprintf(stderr, "ld: unknown option %s\n", a);
    } else {
      vec_push(inputs, (void*)a);
    }
  }
  if (inputs->len == 0) {
    usage(stderr);
    return 1;
  }

  table_init(&g_globals);
  table_init(&g_undef_syms);
  table_init(&g_local_syms);
  table_init(&g_veneers);
  g_objs = new_vector();
  Vector *archives = new_vector();
  bool ok = true;
  for (int i = 0; i < inputs->len; ++i) {
    const char *fn = inputs->data[i];
    size_t len = strlen(fn);
    if (len > 2 && strcmp(fn + len - 2, ".a") == 0) {
      Archive *ar = load_archive(fn);
      if (ar == NULL) {
        fprintf(stderr, "ld: cannot open %s\n", fn);
        return 1;
      }
      vec_push(archives, ar);
      continue;
    }
    FILE *fp = fopen(fn, "rb");
    if (fp == NULL) {
      fprintf(stderr, "ld: cannot open %s\n", fn);
      return 1;
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char *data = malloc_or_die(size);
    if (fread(data, 1, size, fp) != size) {
      fprintf(stderr, "ld: cannot read %s\n", fn);
      return 1;
    }
    fclose(fp);
    Obj *obj = read_obj(fn, data, size);
    if (obj == NULL)
      return 1;
    vec_push(g_objs, obj);
    ok &= define_globals(obj);
  }
  if (!ok || !link_archives(archives))
    return 1;

  layout();
  define_boundaries();

  // Output symbols: null, section symbols
  buf_u8(&g_strtab, 0);
  add_sym(0, 0, 0, STB_LOCAL_, STT_NOTYPE_, SHN_UNDEF_);
  for (int os = 0; os < OS_COUNT; ++os) {
    bool present = os == OS_BSS ? g_bss_size > 0 : (g_out[os].len > 0 || os == OS_TEXT);
    g_secsym[os] = present ? add_sym(0, 0, 0, STB_LOCAL_, STT_SECTION_, os_shndx(os)) : 0;
  }
  if (!relocate())
    return 1;
  // .text may have grown (veneers): the bss/section indices hold

  // The entry
  Global *eg = table_get(&g_globals, alloc_cname(entry));
  uint32_t e_entry = 0;
  if (eg == NULL || eg->obj == NULL) {
    fprintf(stderr, "ld: entry %s not found\n", entry);
    return 1;
  }
  e_entry = eg->obj->out_ofs[eg->shndx] + eg->value;
  if (eg->obj->out_sec[eg->shndx] != OS_TEXT) {
    fprintf(stderr, "ld: entry %s not in .text\n", entry);
    return 1;
  }

  // Globals after the locals: the entry, the task attributes, the undefined
  int first_global = g_nsyms;
  add_sym(buf_str(&g_strtab, entry, strlen(entry)), e_entry, 0, STB_GLOBAL_, STT_FUNC_,
          os_shndx(OS_TEXT));
  if (stacksize > 0)
    add_sym(buf_str(&g_strtab, "nx_stacksize", 12), stacksize, 0, STB_GLOBAL_, STT_NOTYPE_, SHN_ABS_);
  if (priority > 0)
    add_sym(buf_str(&g_strtab, "nx_priority", 11), priority, 0, STB_GLOBAL_, STT_NOTYPE_, SHN_ABS_);
  int undef_base = g_nsyms;
  if (g_pending_globals != NULL) {
    for (int i = 0; i < g_pending_globals->len; ++i) {
      const Name *name = g_pending_globals->data[i];
      add_sym(buf_str(&g_strtab, name->chars, name->bytes), 0, 0, STB_GLOBAL_, STT_NOTYPE_,
              SHN_UNDEF_);
    }
  }

  // Sections: [0], the output sections, their .rel, .symtab, .strtab, .shstrtab
  Buf shstr = {0};
  buf_u8(&shstr, 0);
  typedef struct { uint32_t name, type, flags, size, link, info, align, entsize; Buf *data; } OSh;
  OSh sh[OS_COUNT * 2 + 4];
  int nsh = 0;
  int relcount = 0;
  for (int os = 0; os < OS_COUNT; ++os) {
    bool present = os == OS_BSS ? g_bss_size > 0 : (g_out[os].len > 0 || os == OS_TEXT);
    if (!present)
      continue;
    OSh *h = &sh[nsh++];
    memset(h, 0, sizeof(*h));
    h->name = buf_str(&shstr, kOutNames[os], strlen(kOutNames[os]));
    h->type = os == OS_BSS ? SHT_NOBITS_ : os == OS_INIT ? SHT_INIT_ARRAY_ :
              os == OS_FINI ? SHT_FINI_ARRAY_ : SHT_PROGBITS_;
    h->flags = SHF_ALLOC_ | (os == OS_TEXT ? SHF_EXECINSTR_ : 0) |
               (os >= OS_DATA ? SHF_WRITE_ : 0);
    h->size = os == OS_BSS ? g_bss_size : g_out[os].len;
    h->align = g_out_align[os];
    h->data = os == OS_BSS ? NULL : &g_out[os];
  }
  int nalloc = nsh;
  Buf relbufs[OS_COUNT];
  memset(relbufs, 0, sizeof(relbufs));
  for (int os = 0; os < OS_COUNT; ++os) {
    if (g_outrels[os] == NULL || g_outrels[os]->len == 0)
      continue;
    for (int i = 0; i < g_outrels[os]->len; ++i) {
      OutRel *r = g_outrels[os]->data[i];
      int sym = r->sym < 0 ? undef_base + (-r->sym - 1) : r->sym;
      buf_u32(&relbufs[os], r->offset);
      buf_u32(&relbufs[os], ((uint32_t)sym << 8) | r->type);
    }
    OSh *h = &sh[nsh++];
    memset(h, 0, sizeof(*h));
    char name[24];
    snprintf(name, sizeof(name), ".rel%s", kOutNames[os]);
    h->name = buf_str(&shstr, name, strlen(name));
    h->type = SHT_REL_;
    h->flags = SHF_INFO_LINK_;
    h->size = relbufs[os].len;
    h->info = os_shndx(os);
    h->align = 4;
    h->entsize = 8;
    h->data = &relbufs[os];
    ++relcount;
  }
  int symtab_idx = nsh + 1;
  for (int i = nalloc; i < nsh; ++i)
    sh[i].link = symtab_idx;
  {
    OSh *h = &sh[nsh++];
    memset(h, 0, sizeof(*h));
    h->name = buf_str(&shstr, ".symtab", 7);
    h->type = SHT_SYMTAB_;
    h->size = g_syms.len;
    h->link = symtab_idx + 1;
    h->info = first_global;
    h->align = 4;
    h->entsize = 16;
    h->data = &g_syms;
    h = &sh[nsh++];
    memset(h, 0, sizeof(*h));
    h->name = buf_str(&shstr, ".strtab", 7);
    h->type = SHT_STRTAB_;
    h->size = g_strtab.len;
    h->align = 1;
    h->data = &g_strtab;
    h = &sh[nsh++];
    memset(h, 0, sizeof(*h));
    h->name = buf_str(&shstr, ".shstrtab", 9);
    h->type = SHT_STRTAB_;
    h->align = 1;
    h->data = &shstr;
  }
  sh[nsh - 1].size = shstr.len;

  // Write: header, contents, section headers
  Buf out = {0};
  buf_add(&out, NULL, 52);
  uint32_t *offsets = ALLOCA(sizeof(uint32_t) * nsh);
  for (int i = 0; i < nsh; ++i) {
    OSh *h = &sh[i];
    uint32_t align = h->align >= 4 ? h->align : (h->type == SHT_STRTAB_ ? 1 : 4);
    buf_add(&out, NULL, ALIGN(out.len, align) - out.len);
    offsets[i] = out.len;
    if (h->type != SHT_NOBITS_ && h->data != NULL)
      buf_add(&out, h->data->buf, h->size);
  }
  buf_add(&out, NULL, ALIGN(out.len, 4) - out.len);
  uint32_t shoff = out.len;
  buf_add(&out, NULL, 40);
  for (int i = 0; i < nsh; ++i) {
    OSh *h = &sh[i];
    buf_u32(&out, h->name);
    buf_u32(&out, h->type);
    buf_u32(&out, h->flags);
    buf_u32(&out, 0);
    buf_u32(&out, offsets[i]);
    buf_u32(&out, h->size);
    buf_u32(&out, h->link);
    buf_u32(&out, h->info);
    buf_u32(&out, h->align);
    buf_u32(&out, h->entsize);
  }
  static const unsigned char ident[16] = {0x7f, 'E', 'L', 'F', 1, 1, 1, 0};
  memcpy(out.buf, ident, 16);
  unsigned char *hp = out.buf + 16;
  wr16(hp, 1);              // ET_REL
  wr16(hp + 2, 40);         // EM_ARM
  wr32(hp + 4, 1);
  wr32(hp + 8, e_entry);
  wr32(hp + 12, 0);
  wr32(hp + 16, shoff);
  wr32(hp + 20, 0x05000400);
  wr16(hp + 24, 52);
  wr16(hp + 26, 0);
  wr16(hp + 28, 0);
  wr16(hp + 30, 40);
  wr16(hp + 32, nsh + 1);
  wr16(hp + 34, nsh);       // .shstrtab

  FILE *ofp = fopen(ofn, "wb");
  if (ofp == NULL) {
    fprintf(stderr, "ld: cannot write %s\n", ofn);
    return 1;
  }
  fwrite(out.buf, out.len, 1, ofp);
  fclose(ofp);
  UNUSED(relcount);
  return 0;
}
