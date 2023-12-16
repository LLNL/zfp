#include "src/encode2i.c"

#include "constants/2dInt32.h"
#include "ompExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/omp.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
