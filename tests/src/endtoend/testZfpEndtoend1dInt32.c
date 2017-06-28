#include "src/encode1i.c"

#include "constants/1dInt32.h"
#include "utils/testMacros.h"
#include "utils/genSmoothRandNums.h"
#include "utils/hash32.c"
#include "zfpEndtoendBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches, setupFixedPrec, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches, setupFixedPrec, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches, setupFixedPrec, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate, setupFixedRate, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
