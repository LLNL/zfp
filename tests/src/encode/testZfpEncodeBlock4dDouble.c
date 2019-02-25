#include "src/encode4d.c"

#include "constants/4dDouble.h"
#include "utils/rand64.h"
#include "zfpEncodeBlockBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(when_seededRandomDataGenerated_expect_ChecksumMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_4dDoubleBlock_when_EncodeBlock_expect_ReturnValReflectsNumBitsWrittenToBitstream, setup, teardown),
    cmocka_unit_test_setup_teardown(given_4dDoubleBlock_when_EncodeBlock_expect_BitstreamChecksumMatches, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
