#ifndef ZFP_TIMER_H
#define ZFP_TIMER_H

#if defined(__unix__) || defined(_WIN32)
  #include <time.h>
#elif defined(__MACH__)
  #include <mach/mach_time.h>
#endif

typedef struct zfp_timer zfp_timer;

zfp_timer*
zfp_timer_alloc();

void
zfp_timer_free(zfp_timer* timer);

int
zfp_timer_start(zfp_timer* timer);

double
zfp_timer_stop(zfp_timer* timer);

#endif
