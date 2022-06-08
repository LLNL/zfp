// requires #include "utils/testMacros.h", do outside of main()

_cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

/* strided */
/* contiguous layout supported */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, ReversedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupReversed, teardown),
#if DIMS >= 2
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, PermutedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupPermuted, teardown),
#endif

/* fixed-rate */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupDefaultStride, teardown),

/* variable-rate */
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupDefaultIndexed, teardown),
#ifdef FL_PT_DATA
_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch), setupDefaultIndexed, teardown),
#endif
/*_cmocka_unit_test_setup_teardown(_catFunc3(given_Cuda_, DIM_INT_STR, Array_when_ZfpCompressDecompressReversible_expect_BitstreamAndArrayChecksumsMatch), setupDefaultIndexed, teardown),*/
