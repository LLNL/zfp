#include "src/encode2d.c"

#include "constants/2dDouble.h"
#include "utils/hash64.h"
#include "cudaExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

    /* strided */
    cmocka_unit_test_setup_teardown(given_Cuda_2dDoubleReversedArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupReversed, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_2dDoubleReversedArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupReversed, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_2dDoubleInterleavedArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupInterleaved, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_2dDoubleInterleavedArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupInterleaved, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_2dDoublePermutedArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupPermuted, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_2dDoublePermutedArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupPermuted, teardown),

    /* fixed-rate */
    cmocka_unit_test_setup_teardown(given_Cuda_2dDoubleArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_2dDoubleArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param, teardown),

    cmocka_unit_test_setup_teardown(given_Cuda_2dDoubleArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_2dDoubleArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param, teardown),

    cmocka_unit_test_setup_teardown(given_Cuda_2dDoubleArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_2dDoubleArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param, teardown),
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
