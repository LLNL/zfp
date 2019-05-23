#!/usr/bin/env sh
set -ex

# pass additional args in $1 (starting with whitespace character)
run_all () {
  FLAGS="$1"
  if [ $COMPILER != "msvc" ]; then
    FLAGS="$FLAGS -DCMAKE_SH=CMAKE_SH-NOTFOUND"
  fi

#  run_all_cmd="ctest -V -C $BUILD_TYPE -DGENERATOR=\"$GENERATOR\" -S \"$APPVEYOR_BUILD_FOLDER/cmake/appveyor.cmake\""
  run_all_cmd="cmake -G \"$GENERATOR\" -DCMAKE_BUILD_TYPE=\"$BUILD_TYPE\" .."
  eval "${run_all_cmd}$FLAGS"

  cmake --build . --config $BUILD_TYPE
  ctest -V -C $BUILD_TYPE
}

# create build dir for out-of-source build
mkdir build
cd build

# technically, flags are passed on to cmake/* and actually set there
# config without OpenMP, with CFP (and custom namespace), with aligned allocations (compressed arrays)
BUILD_FLAGS=""
BUILD_FLAGS="$BUILD_FLAGS -DBUILD_UTILITIES=ON"
BUILD_FLAGS="$BUILD_FLAGS -DBUILD_EXAMPLES=ON"
BUILD_FLAGS="$BUILD_FLAGS -DBUILD_CFP=ON"
BUILD_FLAGS="$BUILD_FLAGS -DCFP_NAMESPACE=cfp2"
BUILD_FLAGS="$BUILD_FLAGS -DZFP_WITH_ALIGNED_ALLOC=ON"
BUILD_FLAGS="$BUILD_FLAGS -DZFP_WITH_OPENMP=OFF"
BUILD_FLAGS="$BUILD_FLAGS -DZFP_WITH_CUDA=OFF"

run_all "$BUILD_FLAGS"

## build empty project requiring OpenMP, in a temp directory that ZFP is oblivious to
#mkdir tmpBuild
#cd tmpBuild
## (CMAKE_SH satisfies mingw builds)
#set +e
#if [ $COMPILER != "msvc" ]; then
#  cmake -G "$GENERATOR" "$APPVEYOR_BUILD_FOLDER/tests/ci-utils" -DCMAKE_SH=CMAKE_SH-NOTFOUND
#else
#  cmake -G "$GENERATOR" "$APPVEYOR_BUILD_FOLDER/tests/ci-utils"
#fi
#
#if [ $? -eq 0 ]; then
#  echo "OpenMP found, starting 2nd zfp build"
#  set -e
#  cd ..
#  # keep compiled testing frameworks, to speedup Appveyor
#  rm CMakeCache.txt
#
#  # only run tests not run in previous build, due to appveyor time limit (1 hour)
#  # but continue to build utilities & examples because some support OpenMP
#  BUILD_FLAGS=""
#  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_UTILITIES=ON"
#  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_EXAMPLES=ON"
#  BUILD_FLAGS="$BUILD_FLAGS -DBUILD_OPENMP=ON"
#  BUILD_FLAGS="$BUILD_FLAGS -DOMP_TESTS_ONLY=ON"
#
#  run_all "$BUILD_FLAGS"
#else
#  echo "OpenMP not found, build completed."
#fi
