#include "src/encode2f.c"

#include "constants/2dFloat.h"
#include "ompExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/omp.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
