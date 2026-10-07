// FPC: between wcc (gen_wasm.c, with wcc.h) and wasm_ir.c (xcc's IR
// backend's headers): what each needs of the other, without either's
// headers (their types collide).

#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef struct DataStorage DataStorage;
typedef struct Function Function;
typedef struct Name Name;
typedef struct Type Type;
typedef struct Vector Vector;

// wasm_ir.c: the function's code through xcc's IR and optimiser into code
// (its locals' declarations and body; relocations into relocs).  false:
// not done (code left empty), wcc's own code generator does it.
bool wir_gen_defun(Function *func, DataStorage *code, Vector *relocs);

// gen_wasm.c: instructions that need wcc's tables and relocations, added
// to code (their relocations to func's, at offsets in code; func->extra is
// wcc's then).  false: not known.
bool wir_global_addr(Function *func, DataStorage *code, const Name *label);  // i32.const <address or table index>
bool wir_call(Function *func, DataStorage *code, const Name *label);  // call <function>
bool wir_call_indirect(Function *func, DataStorage *code, const Type *functype);  // call_indirect <type> <table>
bool wir_sp(Function *func, DataStorage *code, bool set);  // global.get/set __stack_pointer
void wir_relocs_shift(Vector *relocs, int from, uint32_t delta);  // relocs[from..]'s offsets += delta
void wir_relocs_drop(Vector *relocs, int from);  // relocs[from..] removed

// wasm_opt.c
void wasm_opt_func(DataStorage *code, Vector *relocs, const unsigned char *ptypes,
                   uint32_t nparams);
