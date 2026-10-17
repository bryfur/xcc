#pragma once

typedef long time_t;

struct timespec {
  time_t tv_sec;
  long tv_nsec;
};

typedef enum {
  CLOCK_REALTIME = 0,
  CLOCK_MONOTONIC = 1,  // FPC: (WASI's numbers)
  CLOCK_PROCESS_CPUTIME_ID = 2,
  CLOCK_REALTIME_COARSE = 5,
} clockid_t;

// FPC: clock()
typedef long clock_t;
#define CLOCKS_PER_SEC  (1000000L)
clock_t clock(void);

time_t time(time_t *timer);
int clock_gettime(clockid_t clk_id, struct timespec *tp);
