#include "src/encode1d.c"

#include "constants/1dDouble.h"
#include "serialExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/serial.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
