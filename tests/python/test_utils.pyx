# TODO: update zfpChecksums cython
import cython
from libc.stdlib cimport malloc, free
cimport libc.stdint as stdint
from libc.stddef cimport ptrdiff_t
from cython cimport view

import zfpy
cimport zfpy

import numpy as np
cimport numpy as np

ctypedef stdint.int32_t int32_t
ctypedef stdint.int64_t int64_t
ctypedef stdint.uint32_t uint32_t
ctypedef stdint.uint64_t uint64_t

cdef extern from "zfp.h":
    # Enums
    ctypedef enum zfp_type:
        zfp_type_none   = 0,
        zfp_type_int32  = 1,
        zfp_type_int64  = 2,
        zfp_type_float  = 3,
        zfp_type_double = 4

    ctypedef enum zfp_mode:
        zfp_mode_null            = 0,
        zfp_mode_expert          = 1,
        zfp_mode_fixed_rate      = 2,
        zfp_mode_fixed_precision = 3,
        zfp_mode_fixed_accuracy  = 4,
        zfp_mode_reversible      = 5 

type_none   = zfp_type_none
type_int32  = zfp_type_int32
type_int64  = zfp_type_int64
type_float  = zfp_type_float
type_double = zfp_type_double

mode_null            = zfp_mode_null
mode_expert          = zfp_mode_expert
mode_fixed_rate      = zfp_mode_fixed_rate
mode_fixed_precision = zfp_mode_fixed_precision
mode_fixed_accuracy  = zfp_mode_fixed_accuracy
mode_reversible      = zfp_mode_reversible


cdef extern from "genSmoothRandNums.h":
    cdef size_t intPow(size_t base, int exponent)
    cdef void generateSmoothRandInts64(size_t minTotalElements,
                                  int numDims,
                                  int amplitudeExp,
                                  int64_t** outputArr,
                                  size_t* outputSideLen,
                                  size_t* outputTotalLen)
    cdef void generateSmoothRandInts32(size_t minTotalElements,
                                  int numDims,
                                  int amplitudeExp,
                                  int32_t** outputArr32Ptr,
                                  size_t* outputSideLen,
                                  size_t* outputTotalLen)
    cdef void generateSmoothRandFloats(size_t minTotalElements,
                                  int numDims,
                                  float** outputArrPtr,
                                  size_t* outputSideLen,
                                  size_t* outputTotalLen)
    cdef void generateSmoothRandDoubles(size_t minTotalElements,
                                   int numDims,
                                   double** outputArrPtr,
                                   size_t* outputSideLen,
                                   size_t* outputTotalLen)

cdef extern from "stridedOperations.h":
    ctypedef enum stride_config:
        AS_IS = 0,
        PERMUTED = 1,
        INTERLEAVED = 2,
        REVERSED = 3

    cdef void reverseArray(void* inputArr,
                      void* outputArr,
                      size_t inputArrLen,
                      zfp_type zfpType)
    cdef void interleaveArray(void* inputArr,
                         void* outputArr,
                         size_t inputArrLen,
                         zfp_type zfpType)
    cdef int permuteSquareArray(void* inputArr,
                           void* outputArr,
                           size_t sideLen,
                           int dims,
                           zfp_type zfpType)
    cdef void getReversedStrides(int dims,
                            size_t n[4],
                            ptrdiff_t s[4])
    cdef void getInterleavedStrides(int dims,
                               size_t n[4],
                               ptrdiff_t s[4])
    cdef void getPermutedStrides(int dims,
                            size_t n[4],
                            ptrdiff_t s[4])

cdef extern from "zfpCompressionParams.h":
    cdef int computeFixedPrecisionParam(int param)
    cdef size_t computeFixedRateParam(int param)
    cdef double computeFixedAccuracyParam(int param)

cdef extern from "zfpChecksums.h":
    ctypedef enum test_type:
        BLOCK_FULL_TEST = 0,
        BLOCK_PARTIAL_TEST = 1,
        ARRAY_TEST = 2

    ctypedef enum subject:
        ORIGINAL_INPUT = 0,
        COMPRESSED_BITSTREAM = 1,
        DECOMPRESSED_ARRAY = 2,

    cdef void computeKeyOriginalInput(test_type tt,
                                 size_t n[4],
                                 uint64_t* key1,
                                 uint64_t* key2)
    cdef void computeKey(test_type tt,
                    subject sjt,
                    size_t n[4],
                    zfp_mode mode,
                    int miscParam,
                    uint64_t* key1,
                    uint64_t* key2)
    cdef uint64_t getChecksumByKey(int dims,
                              zfp_type _type,
                              uint64_t key1,
                              uint64_t key2)
    cdef uint64_t getChecksumOriginalDataBlock(int dims,
                                          zfp_type _type)
    cdef uint64_t getChecksumEncodedBlock(int dims,
                                     zfp_type _type)
    cdef uint64_t getChecksumEncodedPartialBlock(int dims,
                                            zfp_type _type)
    cdef uint64_t getChecksumDecodedBlock(int dims,
                                     zfp_type _type)
    cdef uint64_t getChecksumDecodedPartialBlock(int dims,
                                            zfp_type _type)
    cdef uint64_t getChecksumOriginalDataArray(int ndims,
                                          size_t[4] dims,
                                          zfp_type _type)
    cdef uint64_t getChecksumCompressedBitstream(int ndims,
                                            size_t[4] dims,
                                            zfp_type _type,
                                            zfp_mode mode,
                                            int compressParamNum)
    cdef uint64_t getChecksumDecompressedArray(int ndims,
                                          size_t[4] dims,
                                          zfp_type ztype,
                                          zfp_mode mode,
                                          int compressParamNum)

cdef extern from "zfpHash.h":
    cdef uint64_t hashBitstream(uint64_t* ptrStart,
                           size_t bufsizeBytes)
    cdef uint32_t hashArray32(const uint32_t* arr,
                         size_t nx,
                         ptrdiff_t sx)
    cdef uint32_t hashStridedArray32(const uint32_t* arr,
                                size_t n[4],
                                ptrdiff_t s[4])
    cdef uint64_t hashArray64(const uint64_t* arr,
                         size_t nx,
                         ptrdiff_t sx)
    cdef uint64_t hashStridedArray64(const uint64_t* arr,
                                size_t n[4],
                                ptrdiff_t s[4])

# enums
stride_as_is = AS_IS
stride_permuted = PERMUTED
stride_interleaved = INTERLEAVED
stride_reversed = REVERSED

# functions
cdef validate_num_dimensions(int dims):
    if dims > 4 or dims < 1:
        raise ValueError("Unsupported number of dimensions: {}".format(dims))

cdef validate_ztype(zfp_type ztype):
    if ztype not in [
            type_float,
            type_double,
            type_int32,
            type_int64
    ]:
        raise ValueError("Unsupported ztype: {}".format(ztype))

cdef validate_mode(zfp_mode mode):
    if mode not in [
            mode_fixed_rate,
            mode_fixed_precision,
            mode_fixed_accuracy,
    ]:
        raise ValueError("Unsupported mode: {}".format(mode))

cdef validate_compress_param(int comp_param):
    if comp_param not in range(3): # i.e., [0, 1, 2]
        raise ValueError(
            "Unsupported compression parameter number: {}".format(comp_param)
        )

cpdef getRandNumpyArray(
    int numDims,
    zfp_type ztype,
):
    validate_num_dimensions(numDims)
    validate_ztype(ztype)

    cdef size_t minTotalElements = 0
    cdef int amplitudeExp = 0

    if ztype in [type_float, type_double]:
        minTotalElements = 1000000
    elif ztype in [type_int32, type_int64]:
        minTotalElements = 4096

    cdef int64_t* outputArrInt64 = NULL
    cdef int32_t* outputArrInt32 = NULL
    cdef float* outputArrFloat = NULL
    cdef double* outputArrDouble = NULL
    cdef size_t outputSideLen = 0
    cdef size_t outputTotalLen = 0
    cdef view.array viewArr = None

    if ztype == type_int64:
        amplitudeExp = 64 - 2
        generateSmoothRandInts64(minTotalElements,
                                 numDims,
                                 amplitudeExp,
                                 &outputArrInt64,
                                 &outputSideLen,
                                 &outputTotalLen)
        if numDims == 1:
            viewArr = <int64_t[:outputSideLen]> outputArrInt64
        elif numDims == 2:
            viewArr = <int64_t[:outputSideLen, :outputSideLen]> outputArrInt64
        elif numDims == 3:
            viewArr = <int64_t[:outputSideLen, :outputSideLen, :outputSideLen]> outputArrInt64
        elif numDims == 4:
            viewArr = <int64_t[:outputSideLen, :outputSideLen, :outputSideLen, :outputSideLen]> outputArrInt64
    elif ztype == type_int32:
        amplitudeExp = 32 - 2
        generateSmoothRandInts32(minTotalElements,
                                 numDims,
                                 amplitudeExp,
                                 &outputArrInt32,
                                 &outputSideLen,
                                 &outputTotalLen)
        if numDims == 1:
            viewArr = <int32_t[:outputSideLen]> outputArrInt32
        elif numDims == 2:
            viewArr = <int32_t[:outputSideLen, :outputSideLen]> outputArrInt32
        elif numDims == 3:
            viewArr = <int32_t[:outputSideLen, :outputSideLen, :outputSideLen]> outputArrInt32
        elif numDims == 4:
            viewArr = <int32_t[:outputSideLen, :outputSideLen, :outputSideLen, :outputSideLen]> outputArrInt32
    elif ztype == type_float:
        generateSmoothRandFloats(minTotalElements,
                                 numDims,
                                 &outputArrFloat,
                                 &outputSideLen,
                                 &outputTotalLen)
        if numDims == 1:
            viewArr = <float[:outputSideLen]> outputArrFloat
        elif numDims == 2:
            viewArr = <float[:outputSideLen, :outputSideLen]> outputArrFloat
        elif numDims == 3:
            viewArr = <float[:outputSideLen, :outputSideLen, :outputSideLen]> outputArrFloat
        elif numDims == 4:
            viewArr = <float[:outputSideLen, :outputSideLen, :outputSideLen, :outputSideLen]> outputArrFloat
    elif ztype == type_double:
        generateSmoothRandDoubles(minTotalElements,
                                 numDims,
                                 &outputArrDouble,
                                 &outputSideLen,
                                 &outputTotalLen)
        if numDims == 1:
            viewArr = <double[:outputSideLen]> outputArrDouble
        elif numDims == 2:
            viewArr = <double[:outputSideLen, :outputSideLen]> outputArrDouble
        elif numDims == 3:
            viewArr = <double[:outputSideLen, :outputSideLen, :outputSideLen]> outputArrDouble
        elif numDims == 4:
            viewArr = <double[:outputSideLen, :outputSideLen, :outputSideLen, :outputSideLen]> outputArrDouble
    else:
        raise ValueError("Unknown zfp_type: {}".format(ztype))

    return np.asarray(viewArr)

# ======================================================
# TODO: examine best way to add python block level support
cdef uint64_t getChecksumOriginalDataBlock(
    int dims,
    zfp_type ztype
):
    return 0


cdef uint64_t getChecksumEncodedBlock(
    int dims,
    zfp_type ztype
):
    return 0


cdef uint64_t getChecksumEncodedPartialBlock(
    int dims,
    zfp_type ztype
):
    return 0


cdef uint64_t getChecksumDecodedBlock(
    int dims,
    zfp_type ztype
):
    return 0


cdef uint64_t getChecksumDecodedPartialBlock(
    int dims,
    zfp_type ztype
):
    return 0
# ======================================================

cdef uint64_t getChecksumOriginalDataArray(
    int ndims,
    size_t[4] dims,
    zfp_type ztype
):
    cdef uint64_t[1] key1, key2
    computeKeyOriginalInput(ARRAY_TEST, dims, key1, key2)
    return getChecksumByKey(ndims, ztype, key1[0], key2[0])

cdef  uint64_t getChecksumCompressedBitstream(
    int ndims,
    size_t[4] dims,
    zfp_type ztype,
    zfp_mode mode,
    int compressParamNum
):
    cdef uint64_t[1] key1, key2
    computeKey(ARRAY_TEST, COMPRESSED_BITSTREAM, dims, mode, compressParamNum, key1, key2)
    return getChecksumByKey(ndims, ztype, key1[0], key2[0])

cdef uint64_t getChecksumDecompressedArray(
    int ndims,
    size_t[4] dims,
    zfp_type ztype,
    zfp_mode mode,
    int compressParamNum
):
    cdef uint64_t[1] key1, key2
    computeKey(ARRAY_TEST, DECOMPRESSED_ARRAY, dims, mode, compressParamNum, key1, key2)
    return getChecksumByKey(ndims, ztype, key1[0], key2[0])


cpdef uint64_t getChecksumOrigArray(
    dims,
    zfp_type ztype
):
    cdef int ndims = 4-dims.count(0)
    validate_num_dimensions(ndims)
    validate_ztype(ztype)

    cdef size_t[4] d
    for i in range(len(dims)):
        d[i] = dims[i]
    return getChecksumOriginalDataArray(ndims, d, ztype)

cpdef uint64_t getChecksumCompArray(
    dims,
    zfp_type ztype,
    zfp_mode mode,
    int compressParamNum
):
    cdef int ndims = 4-dims.count(0)
    validate_num_dimensions(ndims)
    validate_ztype(ztype)
    validate_mode(mode)
    validate_compress_param(compressParamNum)

    cdef size_t[4] d
    for i in range(len(dims)):
        d[i] = dims[i]
    return getChecksumCompressedBitstream(ndims, d, ztype, mode, compressParamNum)

cpdef uint64_t getChecksumDecompArray(
    dims,
    zfp_type ztype,
    zfp_mode mode,
    int compressParamNum
):
    cdef int ndims = 4-dims.count(0)
    validate_num_dimensions(ndims)
    validate_ztype(ztype)
    validate_mode(mode)
    validate_compress_param(compressParamNum)

    cdef size_t[4] d
    for i in range(len(dims)):
        d[i] = dims[i]
    return getChecksumDecompressedArray(ndims, d, ztype, mode, compressParamNum)


cpdef computeParameterValue(zfp_mode mode, int param):
    validate_mode(mode)
    validate_compress_param(param)

    if mode == mode_fixed_accuracy:
        return computeFixedAccuracyParam(param)
    elif mode == mode_fixed_precision:
        return computeFixedPrecisionParam(param)
    elif mode == mode_fixed_rate:
        return computeFixedRateParam(param)

cpdef hashStridedArray(
    bytes inarray,
    zfp_type ztype,
    shape,
    strides,
):
    cdef char* array = inarray
    cdef size_t[4] padded_shape
    for i in range(4):
        padded_shape[i] = zfpy.gen_padded_int_list(shape)[i]
    cdef ptrdiff_t[4] padded_strides
    for i in range(4):
        padded_strides[i] = zfpy.gen_padded_int_list(strides)[i]

    if ztype == type_int32 or ztype == type_float:
        return hashStridedArray32(<uint32_t*>array, padded_shape, padded_strides)
    elif ztype == type_int64 or ztype == type_double:
        return hashStridedArray64(<uint64_t*>array, padded_shape, padded_strides)

cpdef hashNumpyArray(
    np.ndarray nparray,
    stride_config stride_conf = AS_IS,
):
    dtype = nparray.dtype
    if dtype not in [np.int32, np.float32, np.int64, np.float64]:
        raise ValueError("Unsupported numpy type: {}".format(dtype))
    if stride_conf not in [AS_IS, PERMUTED, INTERLEAVED, REVERSED]:
        raise ValueError("Unsupported stride config: {}".format(stride_conf))

    size = int(nparray.size)
    cdef ptrdiff_t[4] strides
    cdef size_t[4] shape
    if stride_conf in [AS_IS, INTERLEAVED]:
        stride_width = 1 if stride_conf is AS_IS else 2
        if dtype == np.int32 or dtype == np.float32:
            return hashArray32(<uint32_t*>nparray.data, size, stride_width)
        elif dtype == np.int64 or dtype == np.float64:
            return hashArray64(<uint64_t*>nparray.data, size, stride_width)
    elif stride_conf in [REVERSED, PERMUTED]:
        for i in range(4):
            strides[i] = zfpy.gen_padded_int_list(
                [x for x in nparray.strides[:nparray.ndim]][i]
        )
        for i in range(4):
            shape[i] = zfpy.gen_padded_int_list(
                [x for x in nparray.shape[:nparray.ndim]][i]
        )
        if dtype == np.int32 or dtype == np.float32:
            return hashStridedArray32(<uint32_t*>nparray.data, shape, strides)
        elif dtype == np.int64 or dtype == np.float64:
            return hashStridedArray64(<uint64_t*>nparray.data, shape, strides)


cpdef hashCompressedArray(
    bytes array,
):
    cdef const char* c_array = array
    return hashBitstream(<uint64_t*> c_array, len(array))


cpdef generateStridedRandomNumpyArray(
    stride_config stride,
    np.ndarray randomArray,
):
    cdef int ndim = randomArray.ndim
    shape = [int(x) for x in randomArray.shape[:ndim]]
    dtype = randomArray.dtype
    cdef zfp_type ztype = zfpy.dtype_to_ztype(dtype)
    cdef ptrdiff_t[4] strides
    for i in range(4):
        strides[i] = 0
    cdef size_t[4] dims
    for i in range(4):
        dims[i] = zfpy.gen_padded_int_list(shape)[i]
    cdef size_t inputLen = len(randomArray)
    cdef void* output_array_ptr = NULL
    cdef np.ndarray output_array = None
    cdef view.array output_array_view = None

    if stride == AS_IS:
        # return an unmodified copy
        return randomArray.copy(order='K')
    elif stride == PERMUTED:
        if ndim == 1:
            raise ValueError("Permutation not supported on 1D arrays")
        output_array = np.empty_like(randomArray, order='K')
        getPermutedStrides(ndim, dims, strides)
        for i in range(4):
            strides[i] = int(strides[i]) * (randomArray.itemsize)
        ret = permuteSquareArray(
            randomArray.data,
            output_array.data,
            dims[0],
            ndim,
            ztype
        )
        if ret != 0:
            raise RuntimeError("Error permuting square array")

        return np.lib.stride_tricks.as_strided(
            output_array,
            shape=[x for x in dims[:ndim]],
            strides=reversed([x for x in strides[:ndim]]),
        )

    elif stride == INTERLEAVED:
        num_elements = np.prod(shape)
        new_shape = [x for x in dims if x > 0]
        new_shape[-1] *= 2
        for i in range(4):
            dims[i] = zfpy.gen_padded_int_list(new_shape, pad=0, length=4)[i]

        output_array = np.empty(
            new_shape,
            dtype=dtype
        )
        interleaveArray(
            randomArray.data,
            output_array.data,
            num_elements,
            ztype
        )
        getInterleavedStrides(ndim, dims, strides)
        for i in range(4):
            strides[i] = int(strides[i]) * (randomArray.itemsize)
        return np.lib.stride_tricks.as_strided(
            output_array,
            shape=shape,
            strides=reversed([x for x in strides[:ndim]]),
        )
    else:
        raise ValueError("Unsupported_config: {|}".format(stride))
