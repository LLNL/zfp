#include "src/encode2i.c"

#include "constants/2dInt32.h"
#include "syclExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/sycl.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
