#include "src/encode2l.c"

#include "constants/2dInt64.h"
#include "syclExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/sycl.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
