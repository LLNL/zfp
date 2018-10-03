#!/usr/bin/env sh
set -e

mkdir build
cd build

if [ -n "${COVERAGE}" ]; then

  # build
  cmake .. -DCMAKE_C_STANDARD=${C_STANDARD:-99} -DCMAKE_CXX_STANDARD=${CXX_STANDARD:-98} -DBUILD_CFP=ON -DZFP_WITH_OPENMP=ON -DCMAKE_C_FLAGS=-coverage -DCMAKE_CXX_FLAGS=-coverage;
  cmake --build . ;
  ctest -V -C "Debug";

else

  # build/test without OpenMP, with CFP
  cmake .. -DCMAKE_C_STANDARD=${C_STANDARD:-99} -DCMAKE_CXX_STANDARD=${CXX_STANDARD:-98} -DZFP_WITH_OPENMP=OFF -DBUILD_CFP=ON;
  cmake --build . ;
  ctest -V -C "Debug";

  rm -rf ./* ;

  # build/test with OpenMP, with CFP custom namespace
  cmake .. -DCMAKE_C_STANDARD=${C_STANDARD:-99} -DCMAKE_CXX_STANDARD=${CXX_STANDARD:-98} -DBUILD_CFP=ON -DCFP_NAMESPACE=cfp2;
  cmake --build . ;
  ctest -V -C "Debug";

fi
