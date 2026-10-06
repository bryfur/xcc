// FPC: precompiled prefixes (pch.c)
#pragma once

#include <stdbool.h>
#include <stdio.h>

typedef struct Macro Macro;
typedef struct Vector Vector;

extern const char *pch_dir;  // The cache's directory (NULL: no cache)
extern bool pch_prune;       // Only the system declarations the file uses
extern bool pch_trust_system;  // The system headers as they were (--pch-key says)
extern int pch_stats;        // Print what was done (--pch-stats)

void pch_set_args(int argc, char *argv[]);

// The preprocessor's hooks
int pch_begin(const char *filename, bool *phit);  // the prefix's lines (0: none)
void pch_add_dep(const char *path);
void pch_snapshot(const char *text, size_t size);
void pch_end(FILE *ofp, const char *prefix, size_t prefix_size, const char *body,
             size_t body_size);
void pch_materialize(Macro *macro);

// What pch.c needs of the preprocessor
Vector *pp_pragma_once_files(void);
Vector *pp_guarded_files(void);
Vector *pp_inc_paths(int order);
void pp_add_pragma_once(const char *filename);
void pp_add_guard(const char *filename, const char *name, size_t len);
