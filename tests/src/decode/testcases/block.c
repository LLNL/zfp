// requires #include "utils/testMacros.h", do outside of main()

#ifndef PRINT_CHECKSUMS
_cmocka_unit_test_setup_teardown(when_seededRandomDataGenerated_expect_ChecksumMatches, setup, teardown),
#endif

_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlock_expect_ReturnValReflectsNumBitsReadFromBitstream), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlock_expect_ArrayChecksumMatches), setup, teardown),

#ifdef FL_PT_DATA
// reversible compression and decompression of blocks containing special floating-point values
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodeSpecialBlocks_expect_ArraysMatchBitForBit), setupSpecial, teardown),
#endif
