#ifndef ZFP_SYSTEM_H
#define ZFP_SYSTEM_H

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
  /* C99: use restrict */
  #define restrict_ restrict
#else
  /* C89: no restrict keyword */
  #define restrict_
#endif

/* macros for exporting and importing symbols */
#if defined(_MSC_VER) && defined(ZFP_SHARED_LIBS)
  /* export (import) symbols when ZFP_SOURCE is (is not) defined */
  #ifdef ZFP_SOURCE
    #ifdef __cplusplus
      #define extern_ extern "C" __declspec(dllexport)
    #else
      #define extern_ extern     __declspec(dllexport)
    #endif
  #else
    #ifdef __cplusplus
      #define extern_ extern "C" __declspec(dllimport)
    #else
      #define extern_ extern     __declspec(dllimport)
    #endif
  #endif
#else /* !(_MSC_VER && ZFP_SHARED_LIBS) */
  #ifdef __cplusplus
    #define extern_ extern "C"
  #else
    #define extern_ extern
  #endif
#endif

/* L1 cache line size for alignment purposes */
#ifndef ZFP_CACHE_LINE_SIZE
  #define ZFP_CACHE_LINE_SIZE 0x100
#endif
/* ZFP_CACHE_LINE_SIZE=0 disables alignment */
#if defined(__GNUC__) && ZFP_CACHE_LINE_SIZE
  #define cache_align_(x) x __attribute__((aligned(ZFP_CACHE_LINE_SIZE)))
#else
  #define cache_align_(x) x
#endif

#endif
