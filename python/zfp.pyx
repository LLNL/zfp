import cython
from libc.stdlib cimport malloc, free
import operator
import functools
from cython cimport view
from cpython cimport array
import array

import numpy as np
cimport numpy as np

cdef extern from "bitstream.h":
    cdef struct bitstream:
        pass
    bitstream* stream_open(void* data, size_t bytes);
    void stream_close(bitstream* stream);

cdef extern from "zfp.h":
    # enums
    ctypedef enum zfp_type:
        zfp_type_none   = 0,
        zfp_type_int32  = 1,
        zfp_type_int64  = 2,
        zfp_type_float  = 3,
        zfp_type_double = 4

    # structs
    ctypedef struct zfp_field:
        zfp_type type
        cython.uint nx, ny, nz, nw
        int sx, sy, sz, sw
        void* data
    ctypedef struct zfp_stream:
        pass

    # include #define's
    cython.uint ZFP_HEADER_MAGIC
    cython.uint ZFP_HEADER_META
    cython.uint ZFP_HEADER_MODE
    cython.uint ZFP_HEADER_FULL

    # function definitions
    zfp_stream* zfp_stream_open(bitstream* stream);
    void zfp_stream_close(zfp_stream* stream);
    size_t zfp_stream_maximum_size(const zfp_stream* stream, const zfp_field* field);
    void zfp_stream_set_bit_stream(zfp_stream* stream, bitstream* bs);
    cython.uint zfp_stream_set_precision(zfp_stream* stream, cython.uint precision);
    double zfp_stream_set_accuracy(zfp_stream* stream, double tolerance);
    zfp_field* zfp_field_alloc();
    zfp_field* zfp_field_1d(void* pointer, zfp_type type, cython.uint nx);
    zfp_field* zfp_field_2d(void* pointer, zfp_type type, cython.uint nx, cython.uint ny);
    zfp_field* zfp_field_3d(void* pointer, zfp_type type, cython.uint nx, cython.uint ny, cython.uint nz);
    zfp_field* zfp_field_4d(void* pointer, zfp_type type, cython.uint nx, cython.uint ny, cython.uint nz, cython.uint nw);
    void zfp_field_free(zfp_field* field);
    size_t zfp_compress(zfp_stream* stream, const zfp_field* field);
    size_t zfp_decompress(zfp_stream* stream, zfp_field* field);
    size_t zfp_write_header(zfp_stream* stream, const zfp_field* field, cython.uint mask);
    size_t zfp_read_header(zfp_stream* stream, zfp_field* field, cython.uint mask);
    void zfp_stream_rewind(zfp_stream* stream);
    void zfp_field_set_pointer(zfp_field* field, void* pointer);

# export #define's
HEADER_MAGIC = ZFP_HEADER_MAGIC
HEADER_META = ZFP_HEADER_META
HEADER_MODE = ZFP_HEADER_MODE
HEADER_FULL = ZFP_HEADER_FULL

cpdef dtype_to_ztype(dtype):
    if dtype == np.int32:
        return zfp_type_int32
    elif dtype == np.int64:
        return zfp_type_int64
    elif dtype == np.float32:
        return zfp_type_float
    elif dtype == np.float64:
        return zfp_type_double
    else:
        raise TypeError("Unknown dtype: {}".format(dtype))

cpdef dtype_to_format(dtype):
    # format characters detailed here:
    # https://docs.python.org/2/library/array.html#module-array
    if dtype == np.int32:
        return 'i' # signed int
    elif dtype == np.int64:
        return 'l' # signed long
    elif dtype == np.float32:
        return 'f' # float
    elif dtype == np.float64:
        return 'd' # double
    else:
        raise TypeError("Unknown dtype: {}".format(dtype))

zfp_to_dtype_map = {
    zfp_type_int32: np.int32,
    zfp_type_int64: np.int64,
    zfp_type_float: np.float32,
    zfp_type_double: np.float64,
}
cpdef ztype_to_dtype(zfp_type ztype):
    try:
        return zfp_to_dtype_map[ztype]
    except KeyError:
        # TODO: is this the correct type?
        print "Using np.bytes_"
        return np.bytes_

cdef size_t sizeof_ztype(zfp_type ztype):
    if ztype == zfp_type_int32:
        return sizeof(signed int)
    elif ztype == zfp_type_int64:
        return sizeof(signed long long)
    elif ztype == zfp_type_float:
        return sizeof(float)
    elif ztype == zfp_type_double:
        return sizeof(double)
    else:
        return -1

cdef zfp_field* _init_field(np.ndarray arr):
    shape = arr.shape
    cdef int ndim = arr.ndim
    cdef zfp_type ztype = dtype_to_ztype(arr.dtype)
    cdef zfp_field* field
    cdef void* pointer = arr.data
    if ndim == 1:
        field = zfp_field_1d(pointer, ztype, shape[0])
    elif ndim == 2:
        field = zfp_field_2d(pointer, ztype, shape[0], shape[1])
    elif ndim == 3:
        field = zfp_field_3d(pointer, ztype, shape[0], shape[1], shape[2])
    elif ndim == 4:
        field = zfp_field_4d(pointer, ztype, shape[0], shape[1], shape[2], shape[3])
    else:
        raise RuntimeError("Greater than 4 dimensions not supported")
    return field

@cython.final
cdef class Memory:
    cdef void* data
    def __cinit__(self, size_t size):
        self.data = malloc(size)
        if self.data == NULL:
            raise MemoryError()
    cdef void* __enter__(self):
        return self.data
    def __exit__(self, exc_type, exc_value, exc_tb):
        free(self.data)

cpdef bytes compress_numpy(np.ndarray arr, double tolerance = 1e-3):
    # Input validation
    if arr is None:
        raise TypeError("Input array cannot be None")

    if not arr.flags['F_CONTIGUOUS']:
        # zfp requires a fortran ordered array for optimal compression
        # bonus side-effect: we get a contiguous chunk of memory
        arr = np.asfortranarray(arr)

    # Setup zfp structs to begin compression
    cdef zfp_field* field = _init_field(arr)
    stream = zfp_stream_open(NULL)
    zfp_stream_set_accuracy(stream, tolerance)

    # Allocate space based on the maximum size potentially required by zfp to
    # store the compressed array
    cdef bytes compress_str = None
    cdef size_t maxsize = zfp_stream_maximum_size(stream, field)
    with Memory(maxsize) as data:
        bstream = stream_open(data, maxsize)
        zfp_stream_set_bit_stream(stream, bstream)
        zfp_stream_rewind(stream)
        # write the full header so we can reconstruct the numpy array on
        # decompression
        zfp_write_header(stream, field, HEADER_FULL)
        compressed_size = zfp_compress(stream, field)
        # copy the compressed data into a perfectly sized bytes object
        compress_str = (<char *>data)[:compressed_size]

    zfp_field_free(field)
    zfp_stream_close(stream)
    stream_close(bstream)

    return compress_str

cdef np.ndarray _decompress_with_view(zfp_field* field, zfp_stream* stream):
    cdef zfp_type ztype = field[0].type
    dtype = ztype_to_dtype(ztype)
    format_type = dtype_to_format(dtype)

    shape = (field[0].nx, field[0].ny, field[0].nz, field[0].nw)
    shape = tuple([x for x in shape if x > 0])
    num_elements = functools.reduce(operator.mul, shape)

    cdef view.array decomp_arr = view.array(shape, itemsize=sizeof_ztype(ztype),
                                            format=format_type, mode="fortran",
                                            allocate_buffer=True)
    cdef void* pointer = <void *> decomp_arr.data
    zfp_field_set_pointer(field, pointer)
    if zfp_decompress(stream, field) == 0:
        raise RuntimeError("error during zfp decompression")

    return np.asarray(decomp_arr)

cpdef np.ndarray decompress_numpy(bytes compressed_data):
    if compressed_data is None:
        raise TypeError

    cdef char* comp_data_pointer = compressed_data
    cdef bitstream* bstream = stream_open(comp_data_pointer, len(compressed_data))
    cdef zfp_field* field = zfp_field_alloc()
    cdef zfp_stream* stream = zfp_stream_open(bstream)

    zfp_stream_rewind(stream)
    zfp_read_header(stream, field, HEADER_FULL)

    cdef zfp_type ztype = field[0].type
    if ztype == zfp_type_int32 or ztype == zfp_type_int64:
        raise NotImplementedError("Integer types not supported")

    output_arr = _decompress_with_view(field, stream)

    zfp_field_free(field)
    zfp_stream_close(stream)
    stream_close(bstream)

    return output_arr
