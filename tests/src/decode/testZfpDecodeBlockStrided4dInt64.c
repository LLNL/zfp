#include "src/decode4l.c"

#include "constants/4dInt64.h"
#include "utils/rand64.h"
#include "zfpDecodeBlockStridedBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/blockStrided.c"
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
