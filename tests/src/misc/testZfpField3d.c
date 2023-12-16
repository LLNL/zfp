#include "zfp.h"

#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

#define DIMS 3
#define ZFP_TYPE zfp_type_double
#define SCALAR double

#define NX 20
#define NY 21
#define NZ 12
#define SX 2
#define SY 3
#define SZ 4

#include "zfpFieldBase.c"

#undef DIMS
#undef ZFP_TYPE
#undef SCALAR
#undef NX
#undef NY
#undef NZ
#undef SX
#undef SY
#undef SZ
