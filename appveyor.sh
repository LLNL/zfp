#!/usr/bin/env sh
set -e

# pass additional args in $1 (starting with whitespace character)
run_all () {
  run_all_cmd="ctest -V -C $BUILD_TYPE -DGENERATOR=\"$GENERATOR\" -S \"$APPVEYOR_BUILD_FOLDER/cmake/appveyor.cmake\""
  eval "${run_all_cmd}$1"
}

# create build dir for out-of-source build
mkdir build
cd build

# config without OpenMP, with CFP (and custom namespace), with aligned allocations (compressed arrays)
run_all " -DBUILD_CFP=ON -DCFP_NAMESPACE=cfp2 -DZFP_WITH_ALIGNED_ALLOC=1 -DBUILD_OPENMP=OFF -DBUILD_CUDA=OFF"

# build empty project requiring OpenMP, in a temp directory that ZFP is oblivious to
mkdir tmpBuild
cd tmpBuild
# (CMAKE_SH satisfies mingw builds)
set +e
if [ $COMPILER != "msvc" ]; then
  cmake -G "$GENERATOR" "$APPVEYOR_BUILD_FOLDER/tests/ci-utils" -DCMAKE_SH=CMAKE_SH-NOTFOUND
else
  cmake -G "$GENERATOR" "$APPVEYOR_BUILD_FOLDER/tests/ci-utils"
fi

if [ $? -eq 0 ]; then
  echo "OpenMP found, starting 2nd zfp build"
  set -e
  cd ..
  # keep compiled testing frameworks, to speedup Appveyor
  rm CMakeCache.txt

  # only run tests not run in previous build, due to appveyor time limit (1 hour)
  run_all " -DBUILD_OPENMP=ON -DOMP_TESTS_ONLY=1"
else
  echo "OpenMP not found, build completed."
fi
