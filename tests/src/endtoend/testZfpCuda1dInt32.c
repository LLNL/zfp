#include "src/encode1i.c"

#include "constants/1dInt32.h"
#include "utils/hash32.h"
#include "cudaExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

    /* strided */
    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32ReversedArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupReversed, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32ReversedArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupReversed, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32InterleavedArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupInterleaved, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32InterleavedArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupInterleaved, teardown),

    /* fixed-rate */
    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param, teardown),

    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param, teardown),

    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param, teardown),

    /* non fixed-rate */
    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32Array_when_ZfpCompressNonFixedRate_expect_BitstreamUntouchedAndReturnsZero, setupFixedPrec1Param, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dInt32Array_when_ZfpDecompressNonFixedRate_expect_BitstreamUntouchedAndReturnsZero, setupFixedPrec1Param, teardown),
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
