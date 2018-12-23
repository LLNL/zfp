set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/scikit-build-cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/eyescale-cmake)
include(UseCython)
include(FindPythonExtensions)
include(FindNumPy)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${ZFP_SOURCE_DIR}/include)
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

# cannot reuse the zfp target, use _zfp instead
add_cython_target(_zfp zfp.pyx C PY2)
add_library(_zfp MODULE ${_zfp})
target_link_libraries(_zfp zfp)
python_extension_module(_zfp)

# Build to the currrent binary dir to avoid conflicts with other libraries named zfp
set(pylib_build_dir "${CMAKE_CURRENT_BINARY_DIR}/lib")
set_target_properties(_zfp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${pylib_build_dir})
# Make sure the final library is called zfp (rather than _zfp)
set_target_properties(_zfp PROPERTIES LIBRARY_OUTPUT_NAME zfp)
# Install to the typical python module directory
set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
install(TARGETS _zfp LIBRARY DESTINATION ${python_install_lib_dir})

if(BUILD_TESTING)
  add_test(NAME test_numpy
    COMMAND ${PYTHON_EXECUTABLE} test_numpy.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  set_tests_properties(test_numpy PROPERTIES
    ENVIRONMENT PYTHONPATH=${pylib_build_dir}:$ENV{PYTHONPATH})
endif()
