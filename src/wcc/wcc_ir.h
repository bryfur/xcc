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
// to code (their relocations into relocs, at offsets in code; nothing kept
// is allocated: wasm_ir.c's arena window).  false: not known.
bool wir_global_addr(Vector *relocs, DataStorage *code, const Name *label);  // i32.const <address or table index>
bool wir_call(Vector *relocs, DataStorage *code, const Name *label);  // call <function>
bool wir_call_indirect(Vector *relocs, DataStorage *code, const Type *functype);  // call_indirect <type> <table>
bool wir_sp(Vector *relocs, DataStorage *code, bool set);  // global.get/set __stack_pointer
// Relocations through memory that outlives an arena window: their size
// packed, packed (offsets + delta), and made again into relocs
size_t wir_relocs_size(Vector *relocs);
void wir_relocs_save(Vector *relocs, void *dst, uint32_t delta);
void wir_relocs_restore(const void *src, size_t size, Vector *relocs);

// wasm_opt.c
void wasm_opt_func(DataStorage *code, Vector *relocs, const unsigned char *ptypes,
                   uint32_t nparams);
