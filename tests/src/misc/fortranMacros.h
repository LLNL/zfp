// flexibility to match name mangling gfort vs ifort

#ifdef __INTEL_COMPILER
  #define _prefixFortran(x) zforp_module_mp_ ## x ## _
#else
  #define _prefixFortran(x) __zforp_module_MOD_ ## x
#endif

#define _prefixFOR(x) _prefixFortran(x)
