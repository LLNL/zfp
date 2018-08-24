#include "src/encode1f.c"

#include "constants/1dFloat.h"
#include "utils/hash32.h"
#include "cudaExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

    /* strided */
    cmocka_unit_test_setup_teardown(given_Cuda_1dFloatReversedArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupReversed, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dFloatReversedArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupReversed, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dFloatInterleavedArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupInterleaved, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dFloatInterleavedArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupInterleaved, teardown),

    /* fixed-rate */
    cmocka_unit_test_setup_teardown(given_Cuda_1dFloatArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate0Param, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dFloatArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate0Param, teardown),

    cmocka_unit_test_setup_teardown(given_Cuda_1dFloatArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate1Param, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dFloatArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate1Param, teardown),

    cmocka_unit_test_setup_teardown(given_Cuda_1dFloatArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches, setupFixedRate2Param, teardown),
    cmocka_unit_test_setup_teardown(given_Cuda_1dFloatArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches, setupFixedRate2Param, teardown),
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
