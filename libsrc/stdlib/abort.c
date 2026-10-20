#include "stdlib.h"
#include "signal.h"
#include "unistd.h"

#if defined(__wasm)
#include "../_wasm/wasi.h"

// FPC: WASI has no signals: the program ends as one killed by SIGABRT
// would (status 134), without its atexit handlers
void abort(void) {
  proc_exit(134);
}
#else
void abort(void) {
  raise(SIGABRT);
}
#endif
