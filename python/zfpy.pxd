# cython: language_level=3

import cython
cimport libc.stdint as stdint
from libc.stddef cimport ptrdiff_t

cdef extern from "zfp/bitstream.h":
    cdef struct bitstream:
        pass
    bitstream* stream_open(void* data, size_t)
    void stream_close(bitstream* stream)

cdef extern from "zfp.h":
    # enums
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

    # structs
    ctypedef struct zfp_field:
        zfp_type _type "type"
        size_t nx, ny, nz, nw
        ptrdiff_t sx, sy, sz, sw
        void* data
    ctypedef struct zfp_stream:
        pass

    ctypedef int zfp_bool

    # include #define's
    cython.uint ZFP_HEADER_MAGIC
    cython.uint ZFP_HEADER_META
    cython.uint ZFP_HEADER_MODE
    cython.uint ZFP_HEADER_FULL

    # function declarations
    zfp_stream* zfp_stream_open(bitstream* stream)
    void zfp_stream_close(zfp_stream* stream)
    stdint.uint64_t zfp_stream_mode(const zfp_stream* zfp)
    size_t zfp_stream_maximum_size(const zfp_stream* stream, const zfp_field* field)
    void zfp_stream_rewind(zfp_stream* stream)
    void zfp_stream_set_bit_stream(zfp_stream* stream, bitstream* bs)
    void zfp_stream_set_reversible(zfp_stream* stream)
    double zfp_stream_set_rate(zfp_stream* stream, double rate, zfp_type type, cython.uint dims, zfp_bool align)
    cython.uint zfp_stream_set_precision(zfp_stream* stream, cython.uint precision)
    double zfp_stream_set_accuracy(zfp_stream* stream, double tolerance)
    zfp_mode zfp_stream_set_mode(zfp_stream* stream, stdint.uint64_t mode)
    zfp_mode zfp_stream_compression_mode(zfp_stream* stream)
    double zfp_stream_accuracy(zfp_stream* stream)
    double zfp_stream_rate(zfp_stream* stream, cython.uint dims)
    cython.uint zfp_stream_precision(const zfp_stream* stream)
    zfp_field* zfp_field_alloc()
    zfp_field* zfp_field_1d(void* pointer, zfp_type, size_t nx)
    zfp_field* zfp_field_2d(void* pointer, zfp_type, size_t nx, size_t ny)
    zfp_field* zfp_field_3d(void* pointer, zfp_type, size_t nx, size_t ny, size_t nz)
    zfp_field* zfp_field_4d(void* pointer, zfp_type, size_t nx, size_t ny, size_t nz, size_t nw)
    void zfp_field_free(zfp_field* field)
    zfp_bool zfp_field_stride(const zfp_field* field, ptrdiff_t* stride)
    void zfp_field_set_pointer(zfp_field* field, void* pointer) nogil
    zfp_type zfp_field_set_type(zfp_field* field, zfp_type type)
    void zfp_field_set_stride_1d(zfp_field* field, ptrdiff_t sx)
    void zfp_field_set_stride_2d(zfp_field* field, ptrdiff_t sx, ptrdiff_t sy)
    void zfp_field_set_stride_3d(zfp_field* field, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
    void zfp_field_set_stride_4d(zfp_field* field, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
    size_t zfp_compress(zfp_stream* stream, const zfp_field* field) nogil
    size_t zfp_decompress(zfp_stream* stream, zfp_field* field) nogil
    size_t zfp_write_header(zfp_stream* stream, const zfp_field* field, cython.uint mask)
    size_t zfp_read_header(zfp_stream* stream, zfp_field* field, cython.uint mask)
    void zfp_stream_params(zfp_stream* stream, cython.uint* minbits, cython.uint* maxbits, cython.uint* maxprec, int* minexp);
cdef gen_padded_int_list(orig_array, pad=*, length=*)
