#include "src/encode1i.c"

#include "constants/1dInt32.h"
#include "utils/rand32.h"
#include "utils/hash32.h"
#include "zfpEncodeBlockStridedBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(when_seededRandomDataGenerated_expect_ChecksumMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Block_when_EncodeBlockStrided_expect_ReturnValReflectsNumBitsWrittenToBitstream, setup, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Block_when_EncodeBlockStrided_expect_OnlyStridedEntriesUsed, setup, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Block_when_EncodeBlockStrided_expect_BitstreamChecksumMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Block_when_EncodePartialBlockStrided_expect_ReturnValReflectsNumBitsWrittenToBitstream, setup, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Block_when_EncodePartialBlockStrided_expect_OnlyStridedEntriesUsed, setup, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Block_when_EncodePartialBlockStrided_expect_OnlyEntriesWithinPartialBlockBoundsUsed, setup, teardown),
    cmocka_unit_test_setup_teardown(given_1dInt32Block_when_EncodePartialBlockStrided_expect_BitstreamChecksumMatches, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
