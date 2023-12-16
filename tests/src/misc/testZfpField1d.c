#include "zfp.h"

#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

#define DIMS 1
#define ZFP_TYPE zfp_type_double
#define SCALAR double

#define NX 20
#define SX 2

#include "zfpFieldBase.c"

#undef DIMS
#undef ZFP_TYPE
#undef SCALAR
#undef NX
#undef SX
