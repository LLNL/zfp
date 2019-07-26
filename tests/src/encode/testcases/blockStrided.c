// requires #include "utils/testMacros.h", do outside of main()

// omit redundant checksums covered in non-strided block tests
#ifndef PRINT_CHECKSUMS

_cmocka_unit_test_setup_teardown(when_seededRandomDataGenerated_expect_ChecksumMatches, setup, teardown),

_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlockStrided_expect_ReturnValReflectsNumBitsWrittenToBitstream), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlockStrided_expect_OnlyStridedEntriesUsed), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlockStrided_expect_BitstreamChecksumMatches), setup, teardown),

_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_EncodePartialBlockStrided_expect_ReturnValReflectsNumBitsWrittenToBitstream), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_EncodePartialBlockStrided_expect_OnlyStridedEntriesUsed), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_EncodePartialBlockStrided_expect_OnlyEntriesWithinPartialBlockBoundsUsed), setup, teardown),

#endif

_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_EncodePartialBlockStrided_expect_BitstreamChecksumMatches), setup, teardown),
