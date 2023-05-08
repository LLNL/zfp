set(CMAKE_SHARED_MODULE_PREFIX "")
include(zfpy)

set(TEST_PYTHON_PATH "${PYLIB_BUILD_DIR}")
set(ENV{PYTHONPATH} "${ZFP_SOURCE_DIR}/python")

zfpy_add_test_module(test_utils ${CMAKE_CURRENT_LIST_DIR})
zfpy_add_test(test_numpy)
