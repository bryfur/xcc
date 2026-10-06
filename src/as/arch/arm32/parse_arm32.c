// FPC: ARM32 assembler: instruction parsing (unified syntax)

#include "../../../config.h"
#include "inst.h"

#include <assert.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "parse_asm.h"
#include "util.h"

// The generic parser's opcode table (unused: arm32_parse_inst parses the
// mnemonics), for the linker when the dead code it is in stays
const char *kRawOpTable[] = {NULL};

typedef struct {
  const char *name;
  enum Opcode op;
  bool s;     // takes an S suffix
} OpName;

// Longer names before their prefixes.
static const OpName kOpNames[] = {
  {"vldmia", VLDM}, {"vstmia", VSTM},
  {"vpush", VPUSH}, {"vpop", VPOP},
  {"vldr", VLDR}, {"vstr", VSTR}, {"vmov", VMOV},
  {"vadd", VADD}, {"vsub", VSUB}, {"vmul", VMUL}, {"vdiv", VDIV},
  {"vneg", VNEG}, {"vabs", VABS}, {"vsqrt", VSQRT},
  {"vcmpe", VCMPE}, {"vcmp", VCMP}, {"vmrs", VMRS}, {"vcvt", VCVT},
  {"ldmia", LDM}, {"stmia", STM}, {"ldm", LDM}, {"stm", STM},
  {"ldrsb", LDRSB}, {"ldrsh", LDRSH}, {"ldrb", LDRB}, {"ldrh", LDRH}, {"ldrd", LDRD},
  {"ldr", LDR},
  {"strb", STRB}, {"strh", STRH}, {"strd", STRD}, {"str", STR},
  {"push", PUSH}, {"pop", POP},
  {"movw", MOVW}, {"movt", MOVT}, {"mov", MOV, true}, {"mvn", MVN, true},
  {"and", AND, true}, {"bic", BIC, true}, {"orr", ORR, true}, {"orn", ORN, true},
  {"eor", EOR, true}, {"add", ADD, true}, {"adc", ADC, true}, {"sbc", SBC, true},
  {"sub", SUB, true}, {"rsb", RSB, true},
  {"cmp", CMP}, {"cmn", CMN}, {"tst", TST}, {"teq", TEQ},
  {"lsl", LSL, true}, {"lsr", LSR, true}, {"asr", ASR, true}, {"ror", ROR, true},
  {"umull", UMULL}, {"smull", SMULL}, {"mul", MUL, true}, {"mla", MLA}, {"mls", MLS},
  {"sdiv", SDIV}, {"udiv", UDIV},
  {"sxtb", SXTB}, {"sxth", SXTH}, {"uxtb", UXTB}, {"uxth", UXTH},
  {"clz", CLZ}, {"rbit", RBIT}, {"ubfx", UBFX}, {"sbfx", SBFX},
  {"cbnz", CBNZ}, {"cbz", CBZ},
  {"blx", BLX}, {"bx", BX}, {"bl", BL}, {"b", B},
  {"svc", SVC}, {"nop", NOP},
};

static const char *kCondNames[] = {
  "eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc",
  "hi", "ls", "ge", "lt", "gt", "le", "al",
};

static int find_cond(const char *p, size_t n) {
  if (n != 2)
    return -2;
  if (strncasecmp(p, "cs", 2) == 0)
    return HS;
  if (strncasecmp(p, "cc", 2) == 0)
    return LO;
  for (int i = 0; i < (int)ARRAY_SIZE(kCondNames); ++i)
    if (strncasecmp(p, kCondNames[i], 2) == 0)
      return i;
  return -2;
}

// Mnemonic: name, [s], [cond], [.w/.n/.f32...]
static bool parse_mnemonic(const char *p, size_t n, Inst *inst) {
  // Split off the dotted suffixes.
  size_t base = 0;
  while (base < n && p[base] != '.')
    ++base;
  int nvt = 0;
  for (size_t i = base; i < n; ) {
    size_t j = i + 1;
    while (j < n && p[j] != '.')
      ++j;
    const char *s = p + i + 1;
    size_t len = j - i - 1;
    if (len == 1 && (*s == 'w' || *s == 'W'))
      inst->width = 4;
    else if (len == 1 && (*s == 'n' || *s == 'N'))
      inst->width = 2;
    else if (nvt < 2) {
      enum VType vt = VT_NONE;
      if (len == 3 && strncasecmp(s, "f32", 3) == 0)  vt = VT_F32;
      else if (len == 3 && strncasecmp(s, "f64", 3) == 0)  vt = VT_F64;
      else if (len == 3 && strncasecmp(s, "s32", 3) == 0)  vt = VT_S32;
      else if (len == 3 && strncasecmp(s, "u32", 3) == 0)  vt = VT_U32;
      else if (len == 2 && strncasecmp(s, "32", 2) == 0)  vt = VT_F32;  // vmov.32
      else return false;
      inst->vt[nvt++] = vt;
    } else {
      return false;
    }
    i = j;
  }

  // IT blocks: it, itt, ite, itte...
  if (base >= 2 && base <= 5 && strncasecmp(p, "it", 2) == 0) {
    bool ok = true;
    for (size_t i = 2; i < base; ++i)
      if (tolower(p[i]) != 't' && tolower(p[i]) != 'e')
        ok = false;
    if (ok) {
      inst->op = IT;
      memset(inst->itpattern, 0, sizeof(inst->itpattern));
      for (size_t i = 2; i < base; ++i)
        inst->itpattern[i - 2] = tolower(p[i]);
      return true;
    }
  }

  for (size_t k = 0; k < ARRAY_SIZE(kOpNames); ++k) {
    const OpName *o = &kOpNames[k];
    size_t len = strlen(o->name);
    if (len > base || strncasecmp(p, o->name, len) != 0)
      continue;
    const char *r = p + len;
    size_t rn = base - len;
    bool s = false;
    if (rn > 0 && o->s && (*r == 's' || *r == 'S') && (rn == 1 || rn == 3)) {
      s = true;
      ++r;
      --rn;
    }
    int cond = NOCOND;
    if (rn > 0) {
      cond = find_cond(r, rn);
      if (cond < 0)
        continue;
    }
    inst->op = o->op;
    inst->setflags = s;
    inst->cond = cond;
    return true;
  }
  return false;
}

// Registers

static int parse_core_reg(const char **pp) {
  const char *p = *pp;
  static const struct { const char *name; int no; } kAlias[] = {
    {"sp", 13}, {"lr", 14}, {"pc", 15}, {"ip", 12}, {"fp", 11}, {"sl", 10}, {"sb", 9},
  };
  for (size_t i = 0; i < ARRAY_SIZE(kAlias); ++i) {
    size_t n = strlen(kAlias[i].name);
    if (strncasecmp(p, kAlias[i].name, n) == 0 && !is_label_chr(p[n])) {
      *pp = p + n;
      return kAlias[i].no;
    }
  }
  if ((*p == 'r' || *p == 'R') && isdigit(p[1])) {
    int no = p[1] - '0';
    const char *q = p + 2;
    if (isdigit(*q))
      no = no * 10 + (*q++ - '0');
    if (no <= 15 && !is_label_chr(*q)) {
      *pp = q;
      return no;
    }
  }
  return -1;
}

// s0-s31 (kind 's') or d0-d15 (kind 'd')
static int parse_fp_reg(const char **pp, char *kind) {
  const char *p = *pp;
  char k = tolower(*p);
  if ((k != 's' && k != 'd') || !isdigit(p[1]))
    return -1;
  int no = p[1] - '0';
  const char *q = p + 2;
  if (isdigit(*q))
    no = no * 10 + (*q++ - '0');
  if (is_label_chr(*q) || (k == 's' && no > 31) || (k == 'd' && no > 15))
    return -1;
  *kind = k;
  *pp = q;
  return no;
}

static bool parse_imm_value(ParseInfo *parser, int64_t *value) {
  // A plain number (maybe negative, hex), or a constant expression.
  const char *p = parser->p;
  if (immediate(&p, value) && (*p == '\0' || *p == ',' || *p == ']' || isspace(*p))) {
    parser->p = p;
    return true;
  }
  parse_set_p(parser, parser->p);
  Expr *expr = parse_expr(parser);
  if (expr == NULL || expr->kind != EX_FIXNUM)
    return false;
  *value = expr->fixnum;
  return true;
}

static bool parse_shift_name(const char **pp, enum ShiftType *shift) {
  static const char *kNames[] = {"lsl", "lsr", "asr", "ror"};
  for (int i = 0; i < 4; ++i) {
    if (strncasecmp(*pp, kNames[i], 3) == 0 && isspace((*pp)[3])) {
      *shift = (enum ShiftType)i;
      *pp = skip_whitespaces(*pp + 3);
      return true;
    }
  }
  return false;
}

static bool parse_indirect(ParseInfo *parser, Operand *opr) {
  const char *p = skip_whitespaces(parser->p + 1);  // '['
  int base = parse_core_reg(&p);
  if (base < 0)
    return parse_error(parser, "base register expected");
  opr->type = INDIRECT;
  opr->indirect.base = base;
  opr->indirect.index = -1;
  opr->indirect.shift = 0;
  opr->indirect.offset = 0;
  opr->indirect.prepost = 0;
  p = skip_whitespaces(p);
  if (*p == ',') {
    p = skip_whitespaces(p + 1);
    if (*p == '#') {
      parser->p = p + 1;
      if (!parse_imm_value(parser, &opr->indirect.offset))
        return parse_error(parser, "offset expected");
      p = skip_whitespaces(parser->p);
    } else {
      int index = parse_core_reg(&p);
      if (index < 0)
        return parse_error(parser, "index register expected");
      opr->indirect.index = index;
      p = skip_whitespaces(p);
      if (*p == ',') {
        p = skip_whitespaces(p + 1);
        enum ShiftType sh;
        if (!parse_shift_name(&p, &sh) || sh != SH_LSL || *p != '#')
          return parse_error(parser, "lsl expected");
        parser->p = p + 1;
        int64_t amount;
        if (!parse_imm_value(parser, &amount))
          return parse_error(parser, "shift expected");
        opr->indirect.shift = (int)amount;
        p = skip_whitespaces(parser->p);
      }
    }
  }
  if (*p != ']')
    return parse_error(parser, "`]' expected");
  p = skip_whitespaces(p + 1);
  if (*p == '!') {
    opr->indirect.prepost = 1;
    ++p;
  } else if (*p == ',') {
    // Post-indexed: [r], #nn
    const char *q = skip_whitespaces(p + 1);
    if (*q == '#') {
      parser->p = q + 1;
      if (!parse_imm_value(parser, &opr->indirect.offset))
        return parse_error(parser, "offset expected");
      opr->indirect.prepost = 2;
      p = parser->p;
    }
  }
  parser->p = p;
  return true;
}

static bool parse_reglist(ParseInfo *parser, Operand *opr) {
  const char *p = skip_whitespaces(parser->p + 1);  // '{'
  uint32_t mask = 0;
  int first = -1, count = 0;
  char fkind = 0;
  for (;;) {
    char kind = 0;
    int r = parse_core_reg(&p);
    if (r < 0) {
      r = parse_fp_reg(&p, &kind);
      if (r < 0)
        return parse_error(parser, "register expected");
    }
    int last = r;
    p = skip_whitespaces(p);
    if (*p == '-') {
      p = skip_whitespaces(p + 1);
      char kind2 = 0;
      last = kind ? parse_fp_reg(&p, &kind2) : parse_core_reg(&p);
      if (last < r || kind2 != kind)
        return parse_error(parser, "register range");
      p = skip_whitespaces(p);
    }
    if (kind) {
      if (fkind && fkind != kind)
        return parse_error(parser, "register list");
      fkind = kind;
      if (first < 0)
        first = r;
      else if (r != first + count)
        return parse_error(parser, "consecutive registers expected");
      count += last - r + 1;
    } else {
      for (int i = r; i <= last; ++i)
        mask |= 1U << i;
    }
    if (*p == ',') {
      p = skip_whitespaces(p + 1);
      continue;
    }
    if (*p != '}')
      return parse_error(parser, "`}' expected");
    ++p;
    break;
  }
  if (fkind) {
    opr->type = FREGLIST;
    opr->list.mask = first;
    opr->list.count = count;
    opr->list.dbl = fkind == 'd';
  } else {
    opr->type = REGLIST;
    opr->list.mask = mask;
  }
  parser->p = p;
  return true;
}

static bool parse_operand1(ParseInfo *parser, Operand *opr) {
  const char *p = parser->p;
  if (*p == '[')
    return parse_indirect(parser, opr);
  if (*p == '{')
    return parse_reglist(parser, opr);
  if (*p == '#') {
    p = skip_whitespaces(p + 1);
    if (*p == ':') {
      int upper;
      if (strncasecmp(p, ":lower16:", 9) == 0)
        upper = 0;
      else if (strncasecmp(p, ":upper16:", 9) == 0)
        upper = 1;
      else
        return parse_error(parser, ":lower16: or :upper16: expected");
      parse_set_p(parser, p + 9);
      Expr *expr = parse_expr(parser);
      if (expr == NULL)
        return parse_error(parser, "expression expected");
      opr->type = LABEL16;
      opr->direct.expr = expr;
      opr->direct.upper = upper;
      return true;
    }
    parser->p = p;
    int64_t value;
    if (!parse_imm_value(parser, &value))
      return parse_error(parser, "immediate expected");
    opr->type = IMMEDIATE;
    opr->immediate = value;
    return true;
  }
  if (strncasecmp(p, "APSR_nzcv", 9) == 0 && !is_label_chr(p[9])) {
    opr->type = SPECIAL;
    opr->reg = 0;
    parser->p = p + 9;
    return true;
  }
  if (strncasecmp(p, "fpscr", 5) == 0 && !is_label_chr(p[5])) {
    opr->type = SPECIAL;
    opr->reg = 1;
    parser->p = p + 5;
    return true;
  }
  int r = parse_core_reg(&p);
  if (r >= 0) {
    opr->type = REG;
    opr->reg = r;
    parser->p = p;
    return true;
  }
  char kind;
  r = parse_fp_reg(&p, &kind);
  if (r >= 0) {
    opr->type = kind == 's' ? SREG : DREG;
    opr->reg = r;
    parser->p = p;
    return true;
  }
  // A label expression (branch targets).
  parse_set_p(parser, p);
  Expr *expr = parse_expr(parser);
  if (expr == NULL)
    return parse_error(parser, "operand expected");
  opr->type = DIRECT;
  opr->direct.expr = expr;
  return true;
}

bool arm32_parse_inst(ParseInfo *parser, Line *line) {
  const char *p = parser->p;
  const char *q = p;
  while (isalnum(*q) || *q == '.' || *q == '_')
    ++q;
  Inst inst;
  memset(&inst, 0, sizeof(inst));
  inst.cond = NOCOND;
  if (!parse_mnemonic(p, q - p, &inst))
    return parse_error(parser, "unknown instruction");
  parse_set_p(parser, skip_whitespaces(q));

  int n = 0;
  while (*parser->p != '\0') {
    if (n > 0) {
      if (*parser->p != ',')
        return parse_error(parser, "`,' expected");
      parse_set_p(parser, skip_whitespaces(parser->p + 1));
    }
    // A shift of the previous register: operand 2.
    enum ShiftType sh;
    const char *r = parser->p;
    if (n > 0 && inst.opr[n - 1].type == REG && parse_shift_name(&r, &sh)) {
      if (*r != '#')
        return parse_error(parser, "shift amount expected");
      parser->p = r + 1;
      int64_t amount;
      if (!parse_imm_value(parser, &amount))
        return parse_error(parser, "shift amount expected");
      Operand *prev = &inst.opr[n - 1];
      int reg = prev->reg;
      prev->type = SHIFTREG;
      prev->shiftreg.reg = reg;
      prev->shiftreg.shift = sh;
      prev->shiftreg.amount = (int)amount;
      parse_set_p(parser, skip_whitespaces(parser->p));
      continue;
    }
    if (n >= (int)ARRAY_SIZE(inst.opr))
      return parse_error(parser, "too many operands");
    if (!parse_operand1(parser, &inst.opr[n]))
      return false;
    if (inst.opr[n].type == REG && *parser->p == '!') {  // ldm/stm rn!
      inst.writeback = true;
      parse_set_p(parser, parser->p + 1);
    }
    ++n;
    parse_set_p(parser, skip_whitespaces(parser->p));
  }

  Inst *cloned = calloc_or_die(sizeof(inst));
  *cloned = inst;
  line->inst = cloned;
  return true;
}
