#include "stdio.h"

// FPC: freopen(): the file closed and the new one opened.  The FILE it
// returns may not be the one it was given (the result is what to use, as
// miniz does; stdin, stdout and stderr cannot be reopened this way), and
// a NULL file name (a new mode for the same file) is not supported.

FILE *freopen(const char *fileName, const char *mode, FILE *fp) {
  if (fp != NULL)
    fclose(fp);
  if (fileName == NULL)
    return NULL;
  return fopen(fileName, mode);
}
