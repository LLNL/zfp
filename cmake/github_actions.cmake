cmake_minimum_required(VERSION 3.9)

set(CTEST_SOURCE_DIRECTORY "$ENV{GITHUB_WORKSPACE}")
set(CTEST_BINARY_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/build")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 4. Construct Configure Options
set(CONFIGURE_OPTIONS
  "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
  "-DCMAKE_C_COMPILER=${C_COMPILER}"
  "-DBUILD_TESTING_FULL=ON"
  "-DBUILD_ZFPY=ON"
  "-DZFP_WITH_OPENMP=${OMP_SETTING}"
  "-DPYTHON_INCLUDE_DIR=${Python_INCLUDE_DIRS}"
  "-DPYTHON_LIBRARY=${Python_LIBRARIES}"
)

ctest_start(Experimental GROUP "${BUILD_GROUP}")

ctest_configure(
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE configure_result
)
ctest_submit(PARTS Configure)
if(configure_result)
  message(FATAL_ERROR "Failed to configure.")
endif()

ctest_build(
    RETURN_VALUE build_result
)
ctest_submit(PARTS Build)
if(build_result)
  message(FATAL_ERROR "Failed to build.")
endif()

ctest_test(RETURN_VALUE test_result)
if(test_result)
  message(FATAL_ERROR "Tests failed.")
endif()
ctest_submit(PARTS Test Done)