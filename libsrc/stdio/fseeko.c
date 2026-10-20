#include "stdio.h"

// FPC: fseeko() and ftello(): off_t is long here, as fseek()'s offset

int fseeko(FILE *fp, off_t offset, int origin) {
  return fseek(fp, offset, origin);
}

off_t ftello(FILE *fp) {
  return ftell(fp);
}
