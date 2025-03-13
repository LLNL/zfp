# This script is used to configure the build and submit the results to CDash.
message("CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")
set(cmake_args
  ${CTEST_CMAKE_EXTRA_OPTIONS}
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
  -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
  -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
  -DBUILD_TESTING_FULL:STRING=${BUILD_TESTING_FULL}
  -DZFP_WITH_OPENMP:STRING=${ZFP_WITH_OPENMP}
  -DBUILD_ZFPY:STRING=${BUILD_ZFPY}
)

message("cmake_args:${cmake_args}")

# Create an entry in CDash.
ctest_start(Experimental)

# Gather update information.
find_package(Git)
set(CTEST_UPDATE_VERSION_ONLY ON)
set(CTEST_UPDATE_COMMAND "${GIT_EXECUTABLE}")

ctest_update()
ctest_configure(APPEND
  OPTIONS "${cmake_args}"
  RETURN_VALUE configure_result)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
  ctest_submit(PARTS Update BUILD_ID build_id)
  message(STATUS "Update submission build_id: ${build_id}")
  ctest_submit(PARTS Configure BUILD_ID build_id)
  message(STATUS "Configure submission build_id: ${build_id}")
else()
  ctest_submit(PARTS Update)
  ctest_submit(PARTS Configure)
endif()

if (configure_result)
  message(FATAL_ERROR "Failed to configure")
endif ()
