#include "src/encode3i.c"

#include "constants/3dInt32.h"
#include "utils/rand32.h"
#include "zfpEncodeBlockStridedBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/blockStrided.c"
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
