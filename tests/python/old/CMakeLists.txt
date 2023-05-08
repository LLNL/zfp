list(APPEND CMAKE_MODULE_PATH "${zfpy_SOURCE_DIR}/cmake")

find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package(PythonExtensions)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
include_directories(${NumPy_INCLUDE_DIR})

set(CMAKE_SHARED_MODULE_PREFIX "")

add_cython_target(test_utils test_utils.pyx CXX PY3)
add_library(test_utils MODULE ${test_utils})
target_include_directories(test_utils PRIVATE 
    ${ZFP_SOURCE_DIR}/include 
    ${ZFP_SOURCE_DIR}/tests/utils
    ${ZFP_SOURCE_DIR}/python
)
#target_link_libraries(${name} zfp)
#set_property(TARGET ${name} APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:zfp>")
#python_extension_module(${name})
#set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})
#
#set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
#install(TARGETS ${name} LIBRARY DESTINATION ${python_install_lib_dir})



#find_package(Python3 COMPONENTS Interpreter Development NumPy)
#find_package(PythonExtensions)
#find_package(Cython REQUIRED)
#find_package(NumPy REQUIRED)
#include_directories(${NumPy_INCLUDE_DIR})
#
#set(CMAKE_SHARED_MODULE_PREFIX "")
#
## Cythonize
#add_cython_target(test_utils ${CMAKE_CURRENT_LIST_DIR}/test_utils.pyx CXX PY3)
#add_library(test_utils MODULE ${test_utils})
#target_include_directories(test_utils PRIVATE ${ZFP_SOURCE_DIR}/include ${ZFP_SOURCE_DIR}/python ${ZFP_SOURCE_DIR}/tests/utils ${ZFP_SOURCE_DIR}/tests/python)
#target_link_libraries(test_utils zfp genSmoothRandNumsLib stridedOperationsLib zfpCompressionParamsLib zfpChecksumsLib zfpHashLib)
#set_property(TARGET test_utils APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:zfp>")
#python_extension_module(test_utils)
#set_target_properties(test_utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})
#
## Install to the typical python module directory
#set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
#install(TARGETS test_utils LIBRARY DESTINATION ${python_install_lib_dir})

#add_cython_target(test_utils test_utils.pyx CXX PY3)
#add_library(test_utils MODULE ${test_utils})
#target_include_directories(test_utils PRIVATE 
#    ${ZFP_SOURCE_DIR}/include
#    ${ZFP_SOURCE_DIR}/tests/python
#    ${ZFP_SOURCE_DIR}/tests/utils
#    ${ZFP_SOURCE_DIR}/python)
#target_link_libraries(test_utils zfp genSmoothRandNumsLib stridedOperationsLib zfpCompressionParamsLib zfpChecksumsLib zfpHashLib)
#set_property(TARGET test_utils APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:zfp>")
#python_extension_module(test_utils)
#set_target_properties(test_utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})

# Install to the typical python module directory
#set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
#install(TARGETS test_utils LIBRARY DESTINATION ${python_install_lib_dir})
#
#
#set(TEST_PYTHON_PATH "${PYLIB_BUILD_DIR}")
#if(MSVC)
#  set(TEST_PYTHON_PATH "${TEST_PYTHON_PATH}/${CMAKE_BUILD_TYPE}")
#endif()
#
#if(DEFINED ENV{PYTHONPATH})
#  set(TEST_PYTHON_PATH "${TEST_PYTHON_PATH}:$ENV{PYTHONPATH}")
#endif()
#
#add_test(NAME test_numpy
#  COMMAND ${PYTHON_EXECUTABLE} test_numpy.py
#  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#
#set_tests_properties(test_numpy PROPERTIES
#  ENVIRONMENT PYTHONPATH=${TEST_PYTHON_PATH})
