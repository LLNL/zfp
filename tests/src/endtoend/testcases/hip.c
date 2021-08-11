// requires #include "utils/testMacros.h", do outside of main()

_cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

/* fixed-rate compression/decompression */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupDefaultStride, teardown),

/* fixed-precision compression */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, Array_when_ZfpCompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupDefaultStride, teardown),

/* fixed-accuracy compression */
#ifdef FL_PT_DATA
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, Array_when_ZfpCompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch), setupDefaultStride, teardown),
#endif

/* reversed layout */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, ReversedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupReversed, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, ReversedArray_when_ZfpCompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupReversed, teardown),
#ifdef FL_PT_DATA
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, ReversedArray_when_ZfpCompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch), setupReversed, teardown),
#endif

#if DIMS >= 2
/* permuted layout */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, PermutedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupPermuted, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, PermutedArray_when_ZfpCompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupPermuted, teardown),
#ifdef FL_PT_DATA
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, PermutedArray_when_ZfpCompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch), setupPermuted, teardown),
#endif
#endif

/* unsupported: non-contiguous layout */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, InterleavedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamUntouchedAndReturnsZero), setupInterleaved, teardown),

/* unsupported: reversible mode */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, Array_when_ZfpCompressDecompressReversible_expect_BitstreamUntouchedAndReturnsZero), setupDefaultStride, teardown),
/* unsupported: fixed precision/accuracy decompression */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Hip_, DIM_INT_STR, Array_when_ZfpDecompressFixedPrecisionOrAccuracy_expect_BitstreamUntouchedAndReturnsZero), setupDefaultStride, teardown),

/* unsupported: 4D arrays (TODO) */
