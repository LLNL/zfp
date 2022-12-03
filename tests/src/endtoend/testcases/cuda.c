// requires #include "utils/testMacros.h", do outside of main()

#if DIMS < 4
_cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

/* fixed-rate compression/decompression */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupDefaultStride, teardown),

/* fixed-precision compression/decompression */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupDefaultIndexed, teardown),

/* fixed-accuracy compression/decompression */
#ifdef FL_PT_DATA
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch), setupDefaultIndexed, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressFixedAccuracy_expect_CompressedValuesWithinAccuracy), setupDefaultIndexed, teardown),
#endif

/* reversed layout */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, ReversedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupReversed, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, ReversedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupReversedIndexed, teardown),
#ifdef FL_PT_DATA
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, ReversedArray_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch), setupReversedIndexed, teardown),
#endif

#if DIMS >= 2
/* permuted layout */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, PermutedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupPermuted, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, PermutedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupPermutedIndexed, teardown),
#ifdef FL_PT_DATA
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, PermutedArray_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch), setupPermutedIndexed, teardown),
#endif
#endif

/* CUDA expects contiguous storage */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, InterleavedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamUntouchedAndReturnsZero), setupInterleaved, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, InterleavedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamUntouchedAndReturnsZero), setupInterleaved, teardown),
#ifdef FL_PT_DATA
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, InterleavedArray_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamUntouchedAndReturnsZero), setupInterleaved, teardown),
#endif

#else
/* 4d compression unsupported */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompress_expect_BitstreamUntouchedAndReturnsZero), setupDefaultStride, teardown),
#endif
