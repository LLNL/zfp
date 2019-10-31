// requires #include "utils/testMacros.h", do outside of main()

_cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

#ifndef PRINT_CHECKSUMS

/* strided tests */
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, ReversedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupReversed, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, InterleavedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupInterleaved, teardown),
#if DIMS >= 2
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, PermutedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupPermuted, teardown),
#endif

#endif

/* fixed-precision */
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch), setupDefaultStride, teardown),

/* fixed-rate */
_cmocka_unit_test(_catFunc3(given_, DIM_INT_STR, ZfpStream_when_SetRateWithWriteRandomAccess_expect_RateRoundedUpProperly)),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch), setupDefaultStride, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate), setupDefaultStride, teardown),

#ifdef FL_PT_DATA
/* fixed-accuracy */
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch), setupDefaultStride, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompressFixedAccuracy_expect_CompressedValuesWithinAccuracy), setupDefaultStride, teardown),
#endif

/* reversible */
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompressDecompressReversible_expect_BitstreamAndArrayChecksumsMatch), setupDefaultStride, teardown),
