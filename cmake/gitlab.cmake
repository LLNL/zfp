set(CTEST_SOURCE_DIRECTORY "$ENV{ZFP_PROJECT_DIR}")
set(CTEST_BINARY_DIRECTORY "$ENV{ZFP_PROJECT_DIR}/build")

set(CTEST_COMMAND ctest)
include(${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake)
set(CTEST_SITE "gitlab")
set(CTEST_BUILD_NAME "$ENV{ZFP_COMMIT_BRANCH}-$ENV{ZFP_RUNNER_DESCRIPTION}")
set(cfg_options
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_CFP=${BUILD_CFP}
  -DBUILD_ZFPY=${BUILD_ZFPY}
  -DZFP_WITH_OPENMP=${BUILD_OPENMP}
  -DZFP_WITH_CUDA=${BUILD_CUDA}
  )

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
    -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}
    -DPYTHON_LIBRARY=$ENV{PYTHON_LIB_PATH}
    )
endif()

ctest_start(Experimental TRACK GitLab)
ctest_configure(OPTIONS "${cfg_options}")
ctest_submit(PARTS Update Notes Configure)
ctest_build()
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
