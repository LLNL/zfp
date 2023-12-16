// requires #include "utils/testMacros.h", do outside of main()

#if DIMS < 4
_cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

/* strided */
/* contiguous layout supported */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, ReversedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupReversed, teardown),
#if DIMS >= 2
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, PermutedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupPermuted, teardown),
#endif

/* non-contiguous unsupported */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, InterleavedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamUntouchedAndReturnsZero), setupInterleaved, teardown),

/* fixed-rate */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupDefaultStride, teardown),

/* non fixed-rate modes unsupported */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompressNonFixedRate_expect_BitstreamUntouchedAndReturnsZero), setupDefaultStride, teardown),
#else
/* 4d compression unsupported */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompress_expect_BitstreamUntouchedAndReturnsZero), setupDefaultStride, teardown),
#endif
