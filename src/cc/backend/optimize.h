#pragma once

typedef struct Vector BBContainer;
typedef struct RegAlloc RegAlloc;

void optimize(RegAlloc *ra, BBContainer *bbcon);

#include <stdbool.h>
bool remove_dead_defs(RegAlloc *ra, BBContainer *bbcon);  // FPC
