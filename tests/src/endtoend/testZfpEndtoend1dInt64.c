#include "src/encode1l.c"

#include "constants/1dInt64.h"
#include "utils/testMacros.h"
#include "utils/genSmoothRandInts.c"
#include "utils/hash64.c"
#include "zfpEndtoendBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt64Array_when_ZfpCompress_expect_BitstreamChecksumMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt64Array_when_ZfpDecompress_expect_ArrayChecksumMatches, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
