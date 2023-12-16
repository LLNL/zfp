// requires #include "utils/testMacros.h", do outside of main()

// remove redundant checksum tests already run in non-strided tests
#ifndef PRINT_CHECKSUMS

_cmocka_unit_test_setup_teardown(when_seededRandomDataGenerated_expect_ChecksumMatches, setup, teardown),

_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlockStrided_expect_ReturnValReflectsNumBitsReadFromBitstream), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlockStrided_expect_OnlyStridedEntriesChangedInDestinationArray), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlockStrided_expect_ArrayChecksumMatches), setup, teardown),

_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodePartialBlockStrided_expect_ReturnValReflectsNumBitsReadFromBitstream), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodePartialBlockStrided_expect_NonStridedEntriesUnchangedInDestinationArray), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodePartialBlockStrided_expect_EntriesOutsidePartialBlockBoundsUnchangedInDestinationArray), setup, teardown),

#endif

_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodePartialBlockStrided_expect_ArrayChecksumMatches), setup, teardown),
