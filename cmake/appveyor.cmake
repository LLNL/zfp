
set(CTEST_SOURCE_DIRECTORY "$ENV{APPVEYOR_BUILD_FOLDER}")
set(CTEST_BINARY_DIRECTORY "$ENV{APPVEYOR_BUILD_FOLDER}/build")

#make the appveyor job name have a nicer form for CDash
string(REPLACE ", " "-" job_details "$ENV{APPVEYOR_JOB_NAME}")

set(CTEST_COMMAND ctest)
include(${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake)
set(CTEST_SITE "appveyor")
set(CTEST_CMAKE_GENERATOR "${GENERATOR}")
set(CTEST_BUILD_NAME "$ENV{APPVEYOR_REPO_BRANCH}-${job_details}")
set(cfg_options
  -DCMAKE_BUILD_TYPE=$ENV{BUILD_TYPE}
  -DBUILD_TESTING_FULL=ON
  -DBUILD_CFP=${BUILD_CFP}
  -DBUILD_ZFPY=${BUILD_ZFPY}
  -DZFP_WITH_OPENMP=${BUILD_OPENMP}
  -DZFP_WITH_CUDA=${BUILD_CUDA}
  )

# Work-around the fact that sh.exe is on the path
# for appveyor mingw builds which CMake considers
# to be an error. This is sorta-hacky but works for us
if(NOT ${GENERATOR} MATCHES "Visual Studio")
  list(APPEND cfg_options
    -DCMAKE_SH=CMAKE_SH-NOTFOUND
    )
endif()

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

  # sanitize python include dir path (ex. windows vs linux slashes)
  set(Python_INCLUDE_DIR "")
  file(TO_CMAKE_PATH "${CTEST_SOURCE_DIRECTORY}\\$ENV{VIRTUALENV_NAME}\\Include" Python_INCLUDE_DIR)

  list(APPEND cfg_options
    -DPython_INCLUDE_DIR=${Python_INCLUDE_DIR}
    -DPython_LIBRARY=$ENV{PYTHON_LIB_PATH}
    )
endif()

if(OMP_TESTS_ONLY)
  list(APPEND cfg_options
    -DZFP_OMP_TESTS_ONLY=1
    )
endif()

ctest_start(Experimental TRACK AppVeyor)
ctest_configure(OPTIONS "${cfg_options}")
ctest_submit(PARTS Update Notes Configure)
ctest_build()
ctest_submit(PARTS Build)

if(BUILD_OPENMP)
  # only run tests not run in previous build, due to appveyor time limit (1 hour)
  ctest_test(PARALLEL_LEVEL 6 RETURN_VALUE rv INCLUDE ".*Omp.*")
else()
  ctest_test(PARALLEL_LEVEL 6 RETURN_VALUE rv)
endif()
ctest_submit(PARTS Test)

if(WITH_COVERAGE)
  ctest_coverage()
  ctest_submit(PARTS Coverage)
endif()

if(NOT rv EQUAL 0)
  message(FATAL_ERROR "Test failures occurred.")
endif()
