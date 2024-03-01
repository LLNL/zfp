#include "src/encode2d.c"

#include "constants/2dDouble.h"
#include "syclExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/sycl.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
