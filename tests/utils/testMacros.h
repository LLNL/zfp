// generate test function names containing macros
#define _catFuncStr2(x, y) x ## y
#define _catFunc2(x, y) _catFuncStr2(x, y)

#define _catFuncStr3(x, y, z) x ## y ## z
#define _catFunc3(x, y, z) _catFuncStr3(x, y, z)
