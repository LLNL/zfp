#include "src/decode2i.c"

#include "constants/2dInt32.h"
#include "utils/testMacros.h"
#include "utils/rand32.c"
#include "utils/hash32.c"
#include "zfpDecodeBlockBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(when_seededRandomDataGenerated_expect_ChecksumMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_2dInt32Block_when_DecodeBlock_expect_ReturnValReflectsNumBitsReadFromBitstream, setup, teardown),
    cmocka_unit_test_setup_teardown(given_2dInt32Block_when_DecodeBlock_expect_ArrayChecksumMatches, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
