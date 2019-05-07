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

add_cython_target(zfpy zfpy.pyx C)
add_library(zfpy MODULE ${zfpy})
target_link_libraries(zfpy zfp)
python_extension_module(zfpy)

# Build to the currrent binary dir to avoid conflicts with other libraries named zfp
set(PYLIB_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib" CACHE PATH "Directory where zfp python library will be built")
set_target_properties(zfpy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})
# Install to the typical python module directory
set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
install(TARGETS zfpy LIBRARY DESTINATION ${python_install_lib_dir})
