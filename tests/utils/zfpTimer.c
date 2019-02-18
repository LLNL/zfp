#include "zfpTimer.h"
#include <stdlib.h>

struct zfp_timer {
#if defined(__unix__) || defined(_WIN32)
  clock_t timeStart, timeEnd;
#elif defined(__MACH__)
  uint64_t timeStart, timeEnd;
#endif
};

zfp_timer*
zfp_timer_alloc()
{
  return malloc(sizeof(zfp_timer));
}

void
zfp_timer_free(zfp_timer* timer) {
  free(timer);
}

int
zfp_timer_start(zfp_timer* timer)
{
#if defined(__unix__) || defined(_WIN32)
  timer->timeStart = clock();
#elif defined(__MACH__)
  timer->timeStart = mach_absolute_time();
#else
  return 1;
#endif
  return 0;
}

double
zfp_timer_stop(zfp_timer* timer)
{
  double time;

  // stop timer, compute elapsed time
#if defined(__unix__) || defined(_WIN32)
  timer->timeEnd = clock();
  time = (double)((timer->timeEnd) - (timer->timeStart)) / CLOCKS_PER_SEC;
#elif defined(__MACH__)
  timer->timeEnd = mach_absolute_time();

  mach_timebase_info_data_t tb = {0};
  mach_timebase_info(&tb);
  double timebase = tb.numer / tb.denom;
  time = ((timer->timeEnd) - (timer->timeStart)) * timebase * (1E-9);
#endif

  return time;
}

