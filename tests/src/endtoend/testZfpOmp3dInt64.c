#include "src/encode3l.c"

#include "constants/3dInt64.h"
#include "ompExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/omp.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
