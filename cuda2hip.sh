#!/bin/bash
set -e

HIPIFY=/home/jieyang/dev/HIPIFY/build/hipify-clang

CUDA_SRC=./src/cuda_zfp
HIP_SRC=./src/hip_zfp
INCLUDE=./include

# rm -rf $HIP_SRC
# mkdir $HIP_SRC

for f in $CUDA_SRC/*.cu
do
	echo "Hipifying "$f
	$HIPIFY $f --o-dir=$HIP_SRC -I $INCLUDE -I $CUDA_SRC
	base=`basename "$f.hip" .cu.hip`
	mv $HIP_SRC/$base.cu.hip $HIP_SRC/$base.cpp
	sed -i 's/CUDA/HIP/g' $HIP_SRC/$base.cpp
	sed -i 's/cuda/hip/g' $HIP_SRC/$base.cpp
	sed -i 's/cuh/h/g' $HIP_SRC/$base.cpp
	sed -i 's/CU/HIP/g' $HIP_SRC/$base.cpp
	sed -i 's/cu/hip/g' $HIP_SRC/$base.cpp
	rename  's/cu/hip/' $HIP_SRC/$base.cpp
done

for f in $CUDA_SRC/*.h
do

	echo "Hipifying "$f
	$HIPIFY $f --o-dir=$HIP_SRC -I $INCLUDE -I $CUDA_SRC
	base=`basename "$f.hip" .h.hip`
	mv $HIP_SRC/$base.h.hip $HIP_SRC/$base.h
	sed -i 's/CUDA/HIP/g' $HIP_SRC/$base.h
	sed -i 's/cuda/hip/g' $HIP_SRC/$base.h
	sed -i 's/cuh/h/g' $HIP_SRC/$base.h
	sed -i 's/cu/hip/g' $HIP_SRC/$base.h
	sed -i 's/CU/HIP/g' $HIP_SRC/$base.h
	rename  's/cu/hip/' $HIP_SRC/$base.h
done

for f in $CUDA_SRC/*.cuh
do

	echo "Hipifying "$f
	$HIPIFY $f --o-dir=$HIP_SRC -I $INCLUDE -I $CUDA_SRC
	base=`basename "$f.hip" .cuh.hip`
	mv $HIP_SRC/$base.cuh.hip $HIP_SRC/$base.h
	sed -i 's/CUDA/HIP/g' $HIP_SRC/$base.h
	sed -i 's/cuda/hip/g' $HIP_SRC/$base.h
	sed -i 's/cuh/h/g' $HIP_SRC/$base.h
	sed -i 's/cu/hip/g' $HIP_SRC/$base.h
	sed -i 's/CU/HIP/g' $HIP_SRC/$base.h
	rename  's/cu/hip/' $HIP_SRC/$base.h
done