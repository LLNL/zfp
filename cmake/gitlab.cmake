set(CTEST_SOURCE_DIRECTORY "$ENV{ZFP_PROJECT_DIR}")
set(CTEST_BINARY_DIRECTORY "$ENV{ZFP_PROJECT_DIR}/build")

set(CTEST_COMMAND ctest)
include(${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake)
set(CTEST_SITE "gitlab")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_NAME "$ENV{ZFP_COMMIT_BRANCH}_$ENV{ZFP_RUNNER_DESCRIPTION}")
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

if(BUILD_HIP)
  set(CTEST_SITE "${CTEST_SITE}_hip")
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

ctest_start(Experimental TRACK Gitlab GROUP Gitlab)
ctest_configure(OPTIONS "${cfg_options}")
ctest_submit(PARTS Update Notes Configure 
             RETRY_COUNT 3
            )

ctest_build(RETURN_VALUE rv_build)
ctest_submit(PARTS Build 
             RETRY_COUNT 3
            )

if(NOT rv_build EQUAL 0)
  message(FATAL_ERROR "Build failures occurred.")
endif()

ctest_test(PARALLEL_LEVEL 6 RETURN_VALUE rv_test)
ctest_submit(PARTS Test 
             RETRY_COUNT 3
            )

if(NOT rv_test EQUAL 0)
  message(FATAL_ERROR "Test failures occurred.")
endif()

if(WITH_COVERAGE)
  ctest_coverage()
  ctest_submit(PARTS Coverage
               RETRY_COUNT 3
              )
endif()
