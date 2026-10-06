// FPC: ARM32 assembler: the object file (ELF32, little endian, REL
// relocations), as arm-none-eabi-as writes them: Thumb function symbols
// with bit 0 set, mapping symbols ($t, $d), the AAPCS attributes of
// .eabi_attribute in .ARM.attributes.

#include "../../../config.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ir_asm.h"
#include "parse_asm.h"
#include "table.h"
#include "util.h"

// ELF32 (our own definitions: no <elf.h> needed on the board)
enum {
  ET_REL_ = 1, EM_ARM_ = 40,
  SHT_PROGBITS_ = 1, SHT_SYMTAB_ = 2, SHT_STRTAB_ = 3, SHT_NOBITS_ = 8, SHT_REL_ = 9,
  SHT_INIT_ARRAY_ = 14, SHT_FINI_ARRAY_ = 15, SHT_ARM_ATTRIBUTES_ = 0x70000003,
  SHF_WRITE_ = 1, SHF_ALLOC_ = 2, SHF_EXECINSTR_ = 4, SHF_INFO_LINK_ = 0x40,
  STB_LOCAL_ = 0, STB_GLOBAL_ = 1, STB_WEAK_ = 2,
  STT_NOTYPE_ = 0, STT_OBJECT_ = 1, STT_FUNC_ = 2, STT_SECTION_ = 3,
  SHN_UNDEF_ = 0, SHN_COMMON_ = 0xfff2,
  R_ARM_ABS32_ = 2, R_ARM_THM_CALL_ = 10, R_ARM_THM_JUMP24_ = 30,
  R_ARM_THM_MOVW_ABS_NC_ = 47, R_ARM_THM_MOVT_ABS_ = 48,
};

// .eabi_attribute TAG, VALUE (parse_asm.c)
Vector *arm32_eabi_attributes;  // <intptr_t> pairs

typedef struct {
  uint32_t name, value, size;
  unsigned char info, other;
  uint16_t shndx;
} Sym32;

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

static void buf_uleb(Buf *b, uint32_t v) {
  do {
    unsigned char c = v & 0x7f;
    v >>= 7;
    if (v != 0)
      c |= 0x80;
    buf_u8(b, c);
  } while (v != 0);
}

typedef struct {
  const char *name;
  uint32_t type, flags, link, info, align, entsize;
  const unsigned char *data;
  uint32_t size;
  uint32_t offset;  // in the file
  bool nobits;
} Shdr;

static size_t section_size(const SectionInfo *s) {
  return s->flag & SF_BSS ? s->bss_size : s->ds->len;
}

int emit_elf_obj(const char *ofn, Vector *sections, Table *label_table, Vector *unresolved) {
  FILE *ofp = ofn != NULL ? fopen(ofn, "wb") : stdout;
  if (ofp == NULL) {
    fprintf(stderr, "Failed to open output file: %s\n", ofn);
    return 1;
  }

  // Sections with something in them (.text always)
  Vector *secs = new_vector();
  for (int i = 0; i < sections->len; ++i) {
    SectionInfo *s = sections->data[i];
    s->index = 0;
    if (section_size(s) > 0 || (s->name != NULL && equal_name(s->name, alloc_cname(".text"))))
      vec_push(secs, s);
  }
  for (int i = 0; i < secs->len; ++i)
    ((SectionInfo*)secs->data[i])->index = i + 1;

  // Symbols: null, sections, mapping symbols, locals; then globals
  Buf strtab = {0}, syms = {0};
  buf_u8(&strtab, 0);
  Table symidx;
  table_init(&symidx);
  int nsyms = 0;
#define ADD_SYM(nm, val, sz, bind, type, sh)  do { \
    Sym32 s_ = {.name = (nm), .value = (val), .size = (sz), \
                .info = (unsigned char)(((bind) << 4) | (type)), .other = 0, .shndx = (sh)}; \
    buf_add(&syms, &s_, sizeof(s_)); ++nsyms; } while (0)
  ADD_SYM(0, 0, 0, STB_LOCAL_, STT_NOTYPE_, SHN_UNDEF_);
  int *secsym = ALLOCA(sizeof(int) * (secs->len + 1));
  for (int i = 0; i < secs->len; ++i) {
    secsym[i + 1] = nsyms;
    ADD_SYM(0, 0, 0, STB_LOCAL_, STT_SECTION_, i + 1);
  }
  uint32_t name_t = buf_str(&strtab, "$t", 2), name_d = buf_str(&strtab, "$d", 2);
  for (int i = 0; i < secs->len; ++i) {
    SectionInfo *s = secs->data[i];
    if (section_size(s) > 0 && !(s->flag & SF_BSS))
      ADD_SYM(s->flag & SF_EXECUTABLE ? name_t : name_d, 0, 0, STB_LOCAL_, STT_NOTYPE_, i + 1);
  }

  for (int pass = 0; pass < 2; ++pass) {
    const Name *name;
    LabelInfo *label;
    for (int it = 0; (it = table_iterate(label_table, it, &name, (void**)&label)) != -1; ) {
      bool global = (label->flag & (LF_GLOBAL | LF_WEAK)) != 0;
      if (global != (pass == 1))
        continue;
      if (!global && !(label->flag & LF_REFERRED) && label->kind != LK_FUNC)
        continue;
      LabelInfo *target = label;
      if (label->alias != NULL) {
        target = table_get(label_table, label->alias);
        assert(target != NULL);
      }
      int type = target->kind == LK_FUNC ? STT_FUNC_ : target->kind == LK_OBJECT ? STT_OBJECT_
                                                                                  : STT_NOTYPE_;
      int bind = label->flag & LF_WEAK ? STB_WEAK_ : global ? STB_GLOBAL_ : STB_LOCAL_;
      uint32_t value = 0, size = target->size;
      int shndx = SHN_UNDEF_;
      if (target->flag & LF_DEFINED && target->section != NULL) {
        if ((target->flag & (LF_COMM | LF_GLOBAL)) == (LF_COMM | LF_GLOBAL)) {
          shndx = SHN_COMMON_;
          value = target->align;
        } else {
          shndx = target->section->index;
          value = target->address - target->section->start_address;
          if (type == STT_FUNC_)
            value |= 1;  // Thumb
        }
      }
      table_put(&symidx, name, (void*)(intptr_t)nsyms);
      ADD_SYM(buf_str(&strtab, name->chars, name->bytes), value, size, bind, type, shndx);
    }
    if (pass == 0)
      table_put(&symidx, alloc_cname("  first global"), (void*)(intptr_t)nsyms);
  }
  int first_global = (int)(intptr_t)table_get(&symidx, alloc_cname("  first global"));

  // Relocations, by section
  Buf *rels = calloc_or_die(sizeof(Buf) * (secs->len + 1));
  for (int i = 0; i < unresolved->len; ++i) {
    UnresolvedInfo *u = unresolved->data[i];
    int type;
    switch (u->kind) {
    case UNRES_ARM_ABS32:       type = R_ARM_ABS32_; break;
    case UNRES_ARM_THM_CALL:    type = R_ARM_THM_CALL_; break;
    case UNRES_ARM_THM_JUMP24:  type = R_ARM_THM_JUMP24_; break;
    case UNRES_ARM_THM_MOVW:    type = R_ARM_THM_MOVW_ABS_NC_; break;
    case UNRES_ARM_THM_MOVT:    type = R_ARM_THM_MOVT_ABS_; break;
    default: assert(false); type = 0; break;
    }
    int sym;
    LabelInfo *label = table_get(label_table, u->label);
    if (u->kind == UNRES_ARM_ABS32 && label != NULL && (label->flag & LF_DEFINED) &&
        !(label->flag & (LF_GLOBAL | LF_WEAK))) {
      sym = secsym[label->section->index];  // the addend has the label's offset
    } else {
      void *p;
      if (!table_try_get(&symidx, u->label, &p)) {
        fprintf(stderr, "symbol not found: %.*s\n", NAMES(u->label));
        return 1;
      }
      sym = (int)(intptr_t)p;
    }
    Buf *b = &rels[u->src_section->index];
    buf_u32(b, u->offset);
    buf_u32(b, ((uint32_t)sym << 8) | type);
  }

  // .ARM.attributes
  Buf attrs = {0};
  if (arm32_eabi_attributes != NULL && arm32_eabi_attributes->len > 0) {
    Buf sub = {0};
    for (int i = 0; i + 1 < arm32_eabi_attributes->len; i += 2) {
      buf_uleb(&sub, (uint32_t)(intptr_t)arm32_eabi_attributes->data[i]);
      buf_uleb(&sub, (uint32_t)(intptr_t)arm32_eabi_attributes->data[i + 1]);
    }
    buf_u8(&attrs, 'A');
    buf_u32(&attrs, 4 + 6 + 1 + 4 + sub.len);  // the vendor subsection
    buf_add(&attrs, "aeabi", 6);
    buf_u8(&attrs, 1);  // Tag_File
    buf_u32(&attrs, 1 + 4 + sub.len);
    buf_add(&attrs, sub.buf, sub.len);
  }

  // Section headers
  Vector *shdrs = new_vector();
  Buf shstrtab = {0};
  buf_u8(&shstrtab, 0);
  for (int i = 0; i < secs->len; ++i) {
    SectionInfo *s = secs->data[i];
    Shdr *h = calloc_or_die(sizeof(*h));
    h->name = s->name != NULL ? strndup(s->name->chars, s->name->bytes) : ".text";
    h->flags = SHF_ALLOC_ | (s->flag & SF_EXECUTABLE ? SHF_EXECINSTR_ : 0) |
               (s->flag & (SF_WRITABLE | SF_BSS) ? SHF_WRITE_ : 0);
    h->type = s->flag & SF_BSS ? SHT_NOBITS_ : SHT_PROGBITS_;
    if (strcmp(h->name, ".init_array") == 0) {
      h->type = SHT_INIT_ARRAY_;
      h->flags = SHF_ALLOC_ | SHF_WRITE_;
    } else if (strcmp(h->name, ".fini_array") == 0) {
      h->type = SHT_FINI_ARRAY_;
      h->flags = SHF_ALLOC_ | SHF_WRITE_;
    }
    h->align = s->align > 0 ? s->align : 1;
    h->nobits = (s->flag & SF_BSS) != 0;
    h->data = h->nobits ? NULL : s->ds->buf;
    h->size = section_size(s);
    vec_push(shdrs, h);
  }
  int nsecs = secs->len;
  int symtab_index = nsecs + 1;
  for (int i = 0; i < nsecs; ++i) {
    if (rels[i + 1].len == 0)
      continue;
    Shdr *h = calloc_or_die(sizeof(*h));
    SectionInfo *s = secs->data[i];
    char *name = malloc_or_die(s->name->bytes + 5);
    sprintf(name, ".rel%.*s", NAMES(s->name));
    h->name = name;
    h->type = SHT_REL_;
    h->flags = SHF_INFO_LINK_;
    h->info = i + 1;
    h->align = 4;
    h->entsize = 8;
    h->data = rels[i + 1].buf;
    h->size = rels[i + 1].len;
    vec_push(shdrs, h);
    ++symtab_index;
  }
  if (attrs.len > 0) {
    Shdr *h = calloc_or_die(sizeof(*h));
    h->name = ".ARM.attributes";
    h->type = SHT_ARM_ATTRIBUTES_;
    h->align = 1;
    h->data = attrs.buf;
    h->size = attrs.len;
    vec_push(shdrs, h);
    ++symtab_index;
  }
  {
    Shdr *h = calloc_or_die(sizeof(*h));
    h->name = ".symtab";
    h->type = SHT_SYMTAB_;
    h->link = symtab_index + 1;  // .strtab
    h->info = first_global;
    h->align = 4;
    h->entsize = 16;
    h->data = syms.buf;
    h->size = syms.len;
    vec_push(shdrs, h);

    h = calloc_or_die(sizeof(*h));
    h->name = ".strtab";
    h->type = SHT_STRTAB_;
    h->align = 1;
    h->data = strtab.buf;
    h->size = strtab.len;
    vec_push(shdrs, h);
  }
  for (int i = 0; i < shdrs->len; ++i) {
    Shdr *h = shdrs->data[i];
    if (h->type == SHT_REL_)
      h->link = symtab_index;
  }
  Shdr *shstr = calloc_or_die(sizeof(*shstr));
  shstr->name = ".shstrtab";
  shstr->type = SHT_STRTAB_;
  shstr->align = 1;
  vec_push(shdrs, shstr);
  uint32_t *shname = ALLOCA(sizeof(uint32_t) * shdrs->len);
  for (int i = 0; i < shdrs->len; ++i) {
    Shdr *h = shdrs->data[i];
    shname[i] = buf_str(&shstrtab, h->name, strlen(h->name));
  }
  shstr->data = shstrtab.buf;
  shstr->size = shstrtab.len;

  // Layout: header, section contents, section headers
  uint32_t offset = 52;
  for (int i = 0; i < shdrs->len; ++i) {
    Shdr *h = shdrs->data[i];
    uint32_t align = h->align < 4 && (h->type == SHT_PROGBITS_ || h->type == SHT_NOBITS_) ? h->align
                                                                                         : 4;
    if (h->align > align)
      align = h->align;
    offset = (offset + align - 1) & ~(align - 1);
    h->offset = offset;
    if (!h->nobits)
      offset += h->size;
  }
  uint32_t shoff = (offset + 3) & ~3U;

  Buf out = {0};
  // ELF header
  static const unsigned char ident[16] = {0x7f, 'E', 'L', 'F', 1, 1, 1, 0};
  buf_add(&out, ident, 16);
  buf_u16(&out, ET_REL_);
  buf_u16(&out, EM_ARM_);
  buf_u32(&out, 1);           // e_version
  buf_u32(&out, 0);           // e_entry
  buf_u32(&out, 0);           // e_phoff
  buf_u32(&out, shoff);
  buf_u32(&out, 0x05000400);  // EABI v5, hard float
  buf_u16(&out, 52);          // e_ehsize
  buf_u16(&out, 0);           // e_phentsize
  buf_u16(&out, 0);           // e_phnum
  buf_u16(&out, 40);          // e_shentsize
  buf_u16(&out, shdrs->len + 1);
  buf_u16(&out, shdrs->len);  // e_shstrndx: the last
  for (int i = 0; i < shdrs->len; ++i) {
    Shdr *h = shdrs->data[i];
    if (h->nobits)
      continue;
    if (out.len < h->offset)
      buf_add(&out, NULL, h->offset - out.len);
    buf_add(&out, h->data, h->size);
  }
  if (out.len < shoff)
    buf_add(&out, NULL, shoff - out.len);
  buf_add(&out, NULL, 40);  // [0]
  for (int i = 0; i < shdrs->len; ++i) {
    Shdr *h = shdrs->data[i];
    buf_u32(&out, shname[i]);
    buf_u32(&out, h->type);
    buf_u32(&out, h->flags);
    buf_u32(&out, 0);  // sh_addr
    buf_u32(&out, h->offset);
    buf_u32(&out, h->size);
    buf_u32(&out, h->link);
    buf_u32(&out, h->info);
    buf_u32(&out, h->align);
    buf_u32(&out, h->entsize);
  }
  fwrite(out.buf, out.len, 1, ofp);
  if (ofp != stdout)
    fclose(ofp);
  return 0;
}
