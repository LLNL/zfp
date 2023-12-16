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
  #include "zfpChecksums.h"

  // for both, x is freshly computed checksum from current compression-lib implementation
  // where-as y is the stored constant checksum

  // a triplet (key1, key2, value) is printed
  // key1: identifies what kind of compression occurred, on what input, etc
  // key2: identifies array dimensions
  // value: checksum
  // (macro substitutes "printf() && 0" because we want conditional to fail after executing printf)
  #define ASSERT_EQ_CHECKSUM(dims, zfpType, computedChecksum, key1, key2) printf("{UINT64C(0x%" PRIx64 "), UINT64C(0x%" PRIx64 "), UINT64C(0x%" PRIx64 ")},\n", key1, key2, computedChecksum)
  #define COMPARE_NEQ_CHECKSUM(dims, zfpType, computedChecksum, key1, key2) printf("{UINT64C(0x%" PRIx64 "), UINT64C(0x%" PRIx64 "), UINT64C(0x%" PRIx64 ")},\n", key1, key2, computedChecksum) && 0
#else
  #define ASSERT_EQ_CHECKSUM(dims, zfpType, computedChecksum, key1, key2) assert_int_equal(computedChecksum, getChecksumByKey(dims, zfpType, key1, key2))
  #define COMPARE_NEQ_CHECKSUM(dims, zfpType, computedChecksum, key1, key2) (computedChecksum != getChecksumByKey(dims, zfpType, key1, key2))
#endif

// for condensing repeat tests across different dimensionalities into singular tests
#define _repeat_arg(x, n)  _repeatN(x, n)
#define _repeatN(x, n) _repeat ## n ( x )
#define _repeat1(x) x
#define _repeat2(x) x, x
#define _repeat3(x) x, x, x
#define _repeat4(x) x, x, x, x
