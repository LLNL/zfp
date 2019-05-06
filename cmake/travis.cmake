
set(CTEST_SOURCE_DIRECTORY "$ENV{TRAVIS_BUILD_DIR}")
set(CTEST_BINARY_DIRECTORY "$ENV{TRAVIS_BUILD_DIR}/build")

set(CTEST_COMMAND ctest)
include(${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake)
set(CTEST_SITE "travis")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_NAME "$ENV{TRAVIS_BRANCH}-#$ENV{TRAVIS_JOB_NUMBER}")
set(cfg_options
  -DCMAKE_C_STANDARD=${C_STANDARD}
  -DCMAKE_CXX_STANDARD=${CXX_STANDARD}
  -DBUILD_CFP=${BUILD_CFP}
  -DBUILD_ZFPY=${BUILD_ZFPY}
  -DBUILD_ZFORP=${BUILD_ZFORP}
  -DZFP_WITH_OPENMP=${BUILD_OPENMP}
  -DZFP_WITH_CUDA=${BUILD_CUDA}
  )

# Add the variants to the testers name so that we can report multiple
# times from the same CI builder
if(BUILD_OPENMP)
  set(CTEST_SITE "${CTEST_SITE}_openmp")
endif()

if(BUILD_CUDA)
  set(CTEST_SITE "${CTEST_SITE}_cuda")
endif()

if(BUILD_CFP)
  set(CTEST_SITE "${CTEST_SITE}_cfp")

  if(CFP_NAMESPACE)
    list(APPEND cfg_options
      -DCFP_NAMESPACE=${CFP_NAMESPACE}
      )
    set(CTEST_SITE "${CTEST_SITE}namespace")
  endif()
endif()

if(BUILD_ZFPY)
  set(CTEST_SITE "${CTEST_SITE}_zfpy$ENV{PYTHON_VERSION}")
  list(APPEND cfg_options
    -DPYTHON_INCLUDE_DIR=$ENV{PYTHON_INCLUDE_DIR}
    -DPYTHON_LIBRARY=$ENV{PYTHON_LIBRARY}
    -DPYTHON_EXECUTABLE=$ENV{PYTHON_EXECUTABLE}
    )
endif()

if(BUILD_ZFORP)
  set(CTEST_SITE "${CTEST_SITE}_zforp$ENV{FORTRAN_STANDARD}")
  list(APPEND cfg_options
    -DCMAKE_FORTRAN_FLAGS='-std=f$ENV{FORTRAN_STANDARD}'
    )
endif()

if(WITH_COVERAGE)
  list(APPEND cfg_options
    -DCMAKE_C_FLAGS=-coverage
    -DCMAKE_CXX_FLAGS=-coverage
    -DCMAKE_Fortran_FLAGS=-coverage
    )
  set(CTEST_SITE "${CTEST_SITE}_coverage")
endif()

if(OMP_TESTS_ONLY)
  list(APPEND cfg_options
    -DZFP_OMP_TESTS_ONLY=1
    )
endif()

ctest_start(Experimental TRACK Travis)
ctest_configure(OPTIONS "${cfg_options}")
ctest_submit(PARTS Update Notes Configure)
ctest_build(FLAGS -j1)
ctest_submit(PARTS Build)
ctest_test(PARALLEL_LEVEL 6 RETURN_VALUE rv)
ctest_submit(PARTS Test)

if(WITH_COVERAGE)
  ctest_coverage()
  ctest_submit(PARTS Coverage)
endif()

if(NOT rv EQUAL 0)
  message(FATAL_ERROR "Test failures occurred.")
endif()
