#include "src/encode1d.c"

#include "constants/1dDouble.h"
#include "hipExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/hip.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
