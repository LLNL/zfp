#include "src/encode3f.c"

#include "constants/3dFloat.h"
#include "hipExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/hip.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
