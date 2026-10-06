// FPC: ARM32 assembler: addresses, branch relaxation, relocations.
//
// Branches are short (16-bit) until their target is too far, then wide
// (32-bit); the passes repeat until nothing grows.  A branch to a label of
// the same section is resolved here; a call or a jump to anything else
// gets a relocation (R_ARM_THM_CALL, R_ARM_THM_JUMP24), as do the
// addresses of labels (movw/movt: R_ARM_THM_MOVW_ABS_NC/MOVT_ABS against
// the label's own symbol; .long: R_ARM_ABS32).  The addends are put in the
// instructions and words (REL), as GNU as does.

#include "../../../config.h"
#include "ir_asm.h"

#include <assert.h>
#include <string.h>

#include "inst.h"
#include "parse_asm.h"
#include "table.h"
#include "util.h"

static void sec_add_data(SectionInfo *section, const void *data, size_t bytes) {
  assert(!(section->flag & SF_BSS));
  data_append(section->ds, data, bytes);
}

static void sec_add_bss(SectionInfo *section, size_t size) {
  assert(section->flag & SF_BSS);
  section->bss_size += size;
}

static void sec_align_size(SectionInfo *section, size_t align) {
  if (align > section->align)
    section->align = align;

  if (section->flag & SF_BSS)
    section->bss_size = ALIGN(section->bss_size, align);
  else
    data_align(section->ds, align);
}

bool calc_label_address(uint64_t start_address, Vector *sections, Table *label_table) {
  bool settle = true;
  uint64_t address = start_address;
  for (int sec = 0; sec < sections->len; ++sec) {
    SectionInfo *section = sections->data[sec];
    address = ALIGN(address, section->align);
    section->start_address = address;

    Vector *irs = section->irs;
    for (int i = 0, len = irs->len; i < len; ++i) {
      IR *ir = irs->data[i];
      ir->address = address;
      switch (ir->kind) {
      case IR_LABEL:
        {
          LabelInfo *label;
          if (!table_try_get(label_table, ir->label, (void**)&label)) {
            fprintf(stderr, "[%.*s] not found\n", NAMES(ir->label));
            assert(!"Unexpected");
          } else {
            label->address = address;
          }
        }
        break;
      case IR_CODE:
        address += ir->code.len;
        break;
      case IR_DATA:
        address += ir->data.len;
        break;
      case IR_BSS:
        address += ir->bss;
        break;
      case IR_ZERO:
        address += ir->zero;
        break;
      case IR_ALIGN:
        ir->address = address = ALIGN(address, ir->align);
        if ((size_t)ir->align > section->align) {
          section->align = ir->align;
          settle = false;
        }
        break;
      case IR_EXPR_BYTE:
        address += BYTE_SIZE;
        break;
      case IR_EXPR_SHORT:
        address += SHORT_SIZE;
        break;
      case IR_EXPR_LONG:
        address += LONG_SIZE;
        break;
      case IR_EXPR_QUAD:
        address += QUAD_SIZE;
        break;
      }
    }
  }
  return settle;
}

static void put_hw(unsigned char *p, uint32_t hw) {
  p[0] = hw;
  p[1] = hw >> 8;
}

static uint32_t get_hw(const unsigned char *p) {
  return p[0] | (p[1] << 8);
}

// B.W / BL (T4 offset): offset relative to the instruction + 4
static void encode_t4(unsigned char *p, int64_t offset, bool link) {
  uint32_t s = (offset >> 24) & 1;
  uint32_t i1 = (offset >> 23) & 1, i2 = (offset >> 22) & 1;
  uint32_t j1 = (~(i1 ^ s)) & 1, j2 = (~(i2 ^ s)) & 1;
  put_hw(p, 0xf000 | (s << 10) | ((offset >> 12) & 0x3ff));
  put_hw(p + 2, (link ? 0xd000 : 0x9000) | (j1 << 13) | (j2 << 11) | ((offset >> 1) & 0x7ff));
}

// B<c>.W (T3)
static void encode_t3(unsigned char *p, int cond, int64_t offset) {
  uint32_t s = (offset >> 20) & 1;
  uint32_t j2 = (offset >> 19) & 1, j1 = (offset >> 18) & 1;
  put_hw(p, 0xf000 | (s << 10) | (cond << 6) | ((offset >> 12) & 0x3f));
  put_hw(p + 2, 0x8000 | (j1 << 13) | (j2 << 11) | ((offset >> 1) & 0x7ff));
}

static void add_unresolved(Vector *unresolved, enum UnresolvedKind kind, const Name *label,
                           SectionInfo *section, uintptr_t offset, int add) {
  UnresolvedInfo *unres = calloc_or_die(sizeof(*unres));
  unres->kind = kind;
  unres->label = label;
  unres->src_section = section;
  unres->offset = offset;
  unres->add = add;
  vec_push(unresolved, unres);
}

// Is the label defined in this section (resolvable here)?
static LabelInfo *local_label(Table *label_table, const Name *name, SectionInfo *section) {
  LabelInfo *label = table_get(label_table, name);
  if (label == NULL || !(label->flag & LF_DEFINED) || label->section != section)
    return NULL;
  if (label->flag & (LF_GLOBAL | LF_WEAK))
    return NULL;  // May be interposed: through a relocation
  return label;
}

static bool resolve_branch(IR *ir, SectionInfo *section, Table *label_table, Vector *unresolved,
                           uint64_t start_address, bool *upgraded) {
  Inst *inst = ir->code.inst;
  Code *code = &ir->code;
  uint64_t address = ir->address;
  Value value = calc_expr(label_table, inst->opr[0].direct.expr);
  if (value.label == NULL)
    error("branch target expected");
  bool link = inst->op == BL;
  bool cond = inst->op == B && inst->cond != NOCOND && inst->cond != AL;
  LabelInfo *label = local_label(label_table, value.label, section);
  if (label == NULL) {
    // Elsewhere: a relocation; the addend is the offset - 4 (the pc).
    if (cond)
      error("conditional branch to another section");
    if (code->len != 4) {
      code->len = 4;
      *upgraded = true;
    }
    encode_t4(code->buf, value.offset - 4, link);
    add_unresolved(unresolved, link ? UNRES_ARM_THM_CALL : UNRES_ARM_THM_JUMP24, value.label,
                   section, address - start_address, value.offset);
    return true;
  }

  int64_t offset = (int64_t)(label->address + value.offset) - (int64_t)(address + 4);
  if (link) {
    if (offset < -(1 << 24) || offset >= (1 << 24))
      error("call too far");
    encode_t4(code->buf, offset, true);
    return true;
  }
  if (code->len == 2) {
    if (!cond && offset >= -2048 && offset <= 2046) {
      put_hw(code->buf, 0xe000 | ((offset >> 1) & 0x7ff));
      return true;
    }
    if (cond && offset >= -256 && offset <= 254) {
      put_hw(code->buf, 0xd000 | (inst->cond << 8) | ((offset >> 1) & 0xff));
      return true;
    }
    code->len = 4;  // Too far: wide, the next pass.
    *upgraded = true;
  }
  if (cond) {
    if (offset < -(1 << 20) || offset >= (1 << 20))
      error("conditional branch too far");
    encode_t3(code->buf, inst->cond, offset);
  } else {
    if (offset < -(1 << 24) || offset >= (1 << 24))
      error("branch too far");
    encode_t4(code->buf, offset, false);
  }
  return true;
}

static void put_imm16(unsigned char *p, uint32_t u) {
  uint32_t hw1 = get_hw(p), hw2 = get_hw(p + 2);
  hw1 = (hw1 & 0xfbf0) | ((u >> 12) & 0xf) | (((u >> 11) & 1) << 10);
  hw2 = (hw2 & 0x8f00) | (((u >> 8) & 7) << 12) | (u & 0xff);
  put_hw(p, hw1);
  put_hw(p + 2, hw2);
}

bool resolve_relative_address(Vector *sections, Table *label_table, Vector *unresolved) {
  assert(unresolved != NULL);
  vec_clear(unresolved);
  bool upgraded = false;
  for (int sec = 0; sec < sections->len; ++sec) {
    SectionInfo *section = sections->data[sec];
    Vector *irs = section->irs;
    uint64_t start_address = section->start_address;
    for (int i = 0, len = irs->len; i < len; ++i) {
      IR *ir = irs->data[i];
      uint64_t address = ir->address;
      switch (ir->kind) {
      case IR_CODE:
        {
          Inst *inst = ir->code.inst;
          switch (inst->op) {
          case B: case BL:
            if (inst->opr[0].type == DIRECT)
              resolve_branch(ir, section, label_table, unresolved, start_address, &upgraded);
            break;
          case MOVW: case MOVT:
            if (inst->opr[1].type == LABEL16) {
              Value value = calc_expr(label_table, inst->opr[1].direct.expr);
              if (value.label == NULL) {
                // A constant after all
                uint32_t u = (uint32_t)value.offset;
                put_imm16(ir->code.buf, inst->op == MOVT ? u >> 16 : u & 0xffff);
                break;
              }
              put_imm16(ir->code.buf, (uint32_t)value.offset & 0xffff);
              add_unresolved(unresolved, inst->op == MOVW ? UNRES_ARM_THM_MOVW : UNRES_ARM_THM_MOVT,
                             value.label, section, address - start_address, value.offset);
            }
            break;
          default:
            break;
          }
        }
        break;
      case IR_EXPR_LONG:
        {
          Value value = calc_expr(label_table, ir->expr.expr);
          assert(value.label != NULL);
          int64_t addend = value.offset;
          LabelInfo *label = table_get(label_table, value.label);
          if (label != NULL && (label->flag & LF_DEFINED) &&
              !(label->flag & (LF_GLOBAL | LF_WEAK))) {
            // Against the section: the label's offset in the addend.  A
            // function's thumb bit comes with it.
            addend += label->address - label->section->start_address;
            if (label->kind == LK_FUNC)
              addend |= 1;
          }
          ir->expr.addend = addend;
          add_unresolved(unresolved, UNRES_ARM_ABS32, value.label, section,
                         address - start_address, value.offset);
        }
        break;
      case IR_EXPR_BYTE:
      case IR_EXPR_SHORT:
      case IR_EXPR_QUAD:
        {
          // A difference of labels (or a constant) only.
          Value value = calc_expr(label_table, ir->expr.expr);
          if (value.label != NULL)
            error("relocation of this size not supported");
          ir->expr.addend = value.offset;
        }
        break;
      case IR_LABEL:
      case IR_DATA:
      case IR_BSS:
      case IR_ZERO:
      case IR_ALIGN:
        break;
      }
    }
  }

  return !upgraded;
}

void emit_irs(Vector *sections) {
  for (int sec = 0; sec < sections->len; ++sec) {
    SectionInfo *section = sections->data[sec];
    Vector *irs = section->irs;
    for (int i = 0, len = irs->len; i < len; ++i) {
      IR *ir = irs->data[i];
      switch (ir->kind) {
      case IR_LABEL:
        break;
      case IR_CODE:
        sec_add_data(section, ir->code.buf, ir->code.len);
        break;
      case IR_DATA:
        sec_add_data(section, ir->data.buf, ir->data.len);
        break;
      case IR_BSS:
        sec_add_bss(section, ir->bss);
        break;
      case IR_ZERO:
        if (section->flag & SF_BSS) {
          sec_add_bss(section, ir->zero);
        } else {
          void *buf = calloc_or_die(ir->zero);
          sec_add_data(section, buf, ir->zero);
        }
        break;
      case IR_ALIGN:
        sec_align_size(section, ir->align);
        break;
      case IR_EXPR_BYTE:
      case IR_EXPR_SHORT:
      case IR_EXPR_LONG:
      case IR_EXPR_QUAD:
        {
          int64_t value = ir->expr.addend;
          int size = 1 << (ir->kind - IR_EXPR_BYTE);
          unsigned char buf[8];
          for (int k = 0; k < size; ++k)
            buf[k] = (uint64_t)value >> (8 * k);
          sec_add_data(section, buf, size);
        }
        break;
      }
    }
  }
}
