#include "src/encode3f.c"

#include "constants/3dFloat.h"
#include "cudaExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/cuda.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
