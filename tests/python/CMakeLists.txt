set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/python/scikit-build-cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/python/eyescale-cmake)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${ZFP_SOURCE_DIR}/include)
include_directories(${ZFP_SOURCE_DIR}/python)
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

include_directories(${ZFP_SOURCE_DIR}/tests/python)
include_directories(${ZFP_SOURCE_DIR}/tests/utils)
include_directories(${ZFP_SOURCE_DIR})
add_cython_target(test_utils test_utils.pyx C)
add_library(test_utils MODULE ${test_utils})
target_link_libraries(test_utils zfp genSmoothRandNumsLib stridedOperationsLib zfpCompressionParamsLib zfpChecksumsLib zfpHashLib)
python_extension_module(test_utils)
set_target_properties(test_utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})

add_test(NAME test_numpy
  COMMAND ${PYTHON_EXECUTABLE} test_numpy.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set_tests_properties(test_numpy PROPERTIES
  ENVIRONMENT PYTHONPATH=${PYLIB_BUILD_DIR}:$ENV{PYTHONPATH})
