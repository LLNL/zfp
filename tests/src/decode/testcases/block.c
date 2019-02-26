#include "utils/testMacros.h"

_cmocka_unit_test_setup_teardown(when_seededRandomDataGenerated_expect_ChecksumMatches, setup, teardown),

_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlock_expect_ReturnValReflectsNumBitsReadFromBitstream), setup, teardown),
_cmocka_unit_test_setup_teardown(_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlock_expect_ArrayChecksumMatches), setup, teardown),
