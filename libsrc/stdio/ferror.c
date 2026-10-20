#include "stdio.h"

#include "_file.h"

// FPC: the error flag (a read or write that failed), and clearerr()

int ferror(FILE *fp) {
  return (fp->flag & FF_ERROR) ? 1 : 0;
}

void clearerr(FILE *fp) {
  fp->flag &= ~(FF_ERROR | FF_EOF);
}
