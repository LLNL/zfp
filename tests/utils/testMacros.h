// generate test function names containing macros
#define _catFuncStr2(x, y) x ## y
#define _catFunc2(x, y) _catFuncStr2(x, y)

#define _catFuncStr3(x, y, z) x ## y ## z
#define _catFunc3(x, y, z) _catFuncStr3(x, y, z)

#define _cat_cmocka_unit_test(x) cmocka_unit_test(x)
#define _cmocka_unit_test(x) _cat_cmocka_unit_test(x)

#define _cat_cmocka_unit_test_setup_teardown(x, y, z) cmocka_unit_test_setup_teardown(x, y, z)
#define _cmocka_unit_test_setup_teardown(x, y, z) _cat_cmocka_unit_test_setup_teardown(x, y, z)
