#include "src/encode1l.c"

#include "constants/1dInt64.h"
#include "syclExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/sycl.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
