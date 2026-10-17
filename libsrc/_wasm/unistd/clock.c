// FPC: clock(), the process's time in microseconds (WASI's process clock)
#include "time.h"
#include "stdint.h"
#include "../wasi.h"

clock_t clock(void) {
  uint64_t time;
  if (clock_time_get(CLOCK_PROCESS_CPUTIME_ID, 1000, &time) != 0)
    return (clock_t)-1;
  return (clock_t)(time / 1000);
}
