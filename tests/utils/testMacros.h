// generate test function names containing macros
#define _catFuncStr2(x, y) x ## y
#define _catFunc2(x, y) _catFuncStr2(x, y)

#define _catFuncStr3(x, y, z) x ## y ## z
#define _catFunc3(x, y, z) _catFuncStr3(x, y, z)

#define _cat_cmocka_unit_test(x) cmocka_unit_test(x)
#define _cmocka_unit_test(x) _cat_cmocka_unit_test(x)

#define _cat_cmocka_unit_test_setup_teardown(x, y, z) cmocka_unit_test_setup_teardown(x, y, z)
#define _cmocka_unit_test_setup_teardown(x, y, z) _cat_cmocka_unit_test_setup_teardown(x, y, z)

#ifdef PRINT_CHECKSUMS
  #include <stdio.h>
  #include "checksumKeyGen.h"

  // for both, x is freshly computed checksum from current compression-lib implementation
  // where-as y is the stored constant checksum

  // a pair (key, value) is printed
  // key: identifies what kind of compression occurred, on what input, etc
  // value: checksum
  // (macro substitutes printf(); 0; because we want conditional to fail after executing printf)
  #define ASSERT_EQ_CHECKSUM(testType, currSubject, mode, miscParam, x, y) printf("{0x%"PRIx64", 0x%"PRIx64"},\n", computeKey(testType, currSubject, mode, miscParam), x)
  #define COMPARE_NEQ_CHECKSUM(testType, currSubject, mode, miscParam, x, y) printf("{0x%"PRIx64", 0x%"PRIx64"},\n", computeKey(testType, currSubject, mode, miscParam), x) && 0
#else
  #define ASSERT_EQ_CHECKSUM(testType, currSubject, mode, miscParam, x, y) assert_int_equal(x, y)
  #define COMPARE_NEQ_CHECKSUM(testType, currSubject, mode, miscParam, x, y) (x != y)
#endif
