#!/usr/bin/env sh
set -e

# copy/create "older codec" project at ../zfpV4
./copyScript.sh

# build/test plain lib
mkdir build
cd build
cmake .. -DCMAKE_C_STANDARD=${C_STANDARD:-99} -DCMAKE_CXX_STANDARD=${CXX_STANDARD:-98}
cmake --build .
ctest -V -C "Debug"

# build/test prefixed lib
cmake .. -DCMAKE_C_STANDARD=${C_STANDARD:-99} -DCMAKE_CXX_STANDARD=${CXX_STANDARD:-98} -DZFP_WITH_VERSION_PREFIX=ON
cmake --build .
ctest -V -C "Debug"

# build/test combined (2 versions) lib
cmake .. -DCMAKE_C_STANDARD=${C_STANDARD:-99} -DCMAKE_CXX_STANDARD=${CXX_STANDARD:-98} -DZFP_V4_DIR="../../zfpV4"
cmake --build .
ctest -V -C "Debug"
