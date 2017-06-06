#include "src/encode1i.c"

#include "constants/1dInt32.h"
#include "utils/testMacros.h"
#include "utils/genSmoothRandInts.c"
#include "utils/hash32.c"
#include "zfpEndtoendBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches, setupFixedPrec, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Array_when_ZfpCompress_expect_BitstreamChecksumMatches, setupFixedPrec, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Array_when_ZfpDecompress_expect_ArrayChecksumMatches, setupFixedPrec, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
