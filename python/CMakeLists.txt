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

add_cython_target(zfpyarray ${CMAKE_CURRENT_LIST_DIR}/zfpyarray.pyx CXX)
add_cython_target(zfpynumpy ${CMAKE_CURRENT_LIST_DIR}/zfpynumpy.pyx CXX)
add_library(zfpyarray MODULE ${zfpyarray})
add_library(zfpynumpy MODULE ${zfpynumpy})
target_link_libraries(zfpyarray zfp)
target_link_libraries(zfpynumpy zfp)
python_extension_module(zfpyarray)
python_extension_module(zfpynumpy)

# Build to the current binary dir to avoid conflicts with other libraries named zfp
set(PYLIB_BUILD_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Directory where zfp python library will be built")
set_target_properties(zfpyarray PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})
set_target_properties(zfpynumpy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})

# Install to the typical python module directory
set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
install(TARGETS zfpyarray zfpynumpy LIBRARY DESTINATION ${python_install_lib_dir})
