#include "src/encode1i.c"

#include "constants/1dInt32.h"
#include "hipExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/hip.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
