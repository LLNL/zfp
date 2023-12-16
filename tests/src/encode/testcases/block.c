// requires #include "utils/testMacros.h", do outside of main()

_cmocka_unit_test_setup_teardown(when_seededRandomDataGenerated_expect_ChecksumMatches, setup, teardown),

_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlock_expect_ReturnValReflectsNumBitsWrittenToBitstream), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlock_expect_BitstreamChecksumMatches), setup, teardown),

#ifdef FL_PT_DATA
// reversible compression of blocks containing special floating-point values
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_EncodeSpecialBlocks_expect_BitstreamChecksumMatches), setupSpecial, teardown),
#endif
