
set(CTEST_SOURCE_DIRECTORY "$ENV{TRAVIS_BUILD_DIR}")
set(CTEST_BINARY_DIRECTORY "$ENV{TRAVIS_BUILD_DIR}/build")

# We need to set this otherwise we yet 255 as our return code!
set(CTEST_COMMAND ctest)
include(${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake)
set(CTEST_SITE "travis")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_NAME "$ENV{TRAVIS_BRANCH}-#$ENV{TRAVIS_JOB_NUMBER}")
set(cfg_options
  -DCMAKE_C_STANDARD=${C_STANDARD}
  -DCMAKE_CXX_STANDARD=${CXX_STANDARD}
  -DBUILD_CFP=${BUILD_CFP}
  -DZFP_WITH_OPENMP=${BUILD_OPENMP}
  -DZFP_WITH_CUDA=${BUILD_CUDA}
  )

if(CFP_NAMESPACE)
  list(APPEND cfg_options
    -DCFP_NAMESPACE=${CFP_NAMESPACE}
  )
endif()

if(WITH_COVERAGE)
  list(APPEND cfg_options
    -DCMAKE_C_FLAGS=-coverage
    -DCMAKE_CXX_FLAGS=-coverage
    )
endif()

ctest_start(Experimental TRACK Travis)
ctest_configure(OPTIONS "${cfg_options}")
ctest_submit(PARTS Update Notes Configure)
ctest_build(FLAGS -j1)
ctest_submit(PARTS Build)
ctest_test(RETURN_VALUE rv)
ctest_submit(PARTS Test)

if(WITH_COVERAGE)
  ctest_coverage()
  ctest_submit(PARTS Coverage)
endif()

if(NOT rv EQUAL 0)
  message(FATAL_ERROR "Test failures occurred.")
endif()
