if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.27.0)
    cmake_policy(SET CMP0148 OLD)
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/python/scikit-build-cmake)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${ZFP_SOURCE_DIR}/include)
include_directories(${ZFP_SOURCE_DIR}/python)
include_directories(${NumPy_INCLUDE_DIR})

include_directories(${ZFP_SOURCE_DIR}/tests/python)
include_directories(${ZFP_SOURCE_DIR}/tests/utils)
include_directories(${ZFP_SOURCE_DIR})
add_cython_target(test_utils test_utils.pyx C PY3)
add_library(test_utils MODULE ${test_utils})
target_link_libraries(test_utils zfp genSmoothRandNumsLib stridedOperationsLib zfpCompressionParamsLib zfpChecksumsLib zfpHashLib)
python_extension_module(test_utils)
set_target_properties(test_utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})

set(TEST_PYTHON_PATH "${PYLIB_BUILD_DIR}")
if(MSVC)
  set(TEST_PYTHON_PATH "${TEST_PYTHON_PATH}/${CMAKE_BUILD_TYPE}")
endif()

if(DEFINED ENV{PYTHONPATH})
  set(TEST_PYTHON_PATH "${TEST_PYTHON_PATH}:$ENV{PYTHONPATH}")
endif()

add_test(NAME test_numpy
  COMMAND ${PYTHON_EXECUTABLE} test_numpy.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set_tests_properties(test_numpy PROPERTIES
  ENVIRONMENT PYTHONPATH=${TEST_PYTHON_PATH})
