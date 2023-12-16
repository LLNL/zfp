#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "zfp/array.h"

/* only run this test when compiling with CFP_NAMESPACE=cfp2 */

/* test fails if compiler errors out */
static void
given_cfpCompiledWithNamespace_cfp2_when_linkToCfpLib_expect_namespacePersists(void** state)
{
  cfp_array1d arr = cfp2.array1d.ctor_default();
  assert_non_null(arr.object);

  cfp2.array1d.dtor(arr);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(given_cfpCompiledWithNamespace_cfp2_when_linkToCfpLib_expect_namespacePersists),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
