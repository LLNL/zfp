// requires #include "utils/testMacros.h", do outside of main()

_cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

// OpenMP decompression not yet supported

/* strided tests */
_cmocka_unit_test_setup_teardown(_catFunc3(given_OpenMP_, DIM_INT_STR, ReversedArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumsMatch), setupReversed, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_OpenMP_, DIM_INT_STR, InterleavedArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumsMatch), setupInterleaved, teardown),
#if DIMS >= 2
_cmocka_unit_test_setup_teardown(_catFunc3(given_OpenMP_, DIM_INT_STR, PermutedArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumsMatch), setupPermuted, teardown),
#endif

/* non-strided tests */
_cmocka_unit_test_setup_teardown(_catFunc3(given_OpenMP_, DIM_INT_STR, Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumsMatch), setupDefaultStride, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_OpenMP_, DIM_INT_STR, Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumsMatch), setupDefaultStride, teardown),
#ifdef FL_PT_DATA
_cmocka_unit_test_setup_teardown(_catFunc3(given_OpenMP_, DIM_INT_STR, Array_when_ZfpCompressFixedAccuracy_expect_BitstreamChecksumsMatch), setupDefaultStride, teardown),
#endif
_cmocka_unit_test_setup_teardown(_catFunc3(given_OpenMP_, DIM_INT_STR, Array_when_ZfpCompressReversible_expect_BitstreamChecksumsMatch), setupDefaultStride, teardown),
