#include "src/encode4f.c"

#include "constants/4dFloat.h"
#include "cudaExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/cuda.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
