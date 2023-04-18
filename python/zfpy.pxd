import cython

cimport libc.stdint as stdint
from libc.stddef cimport ptrdiff_t

cdef extern from "zfp/bitstream.h":
    # Structs
    cdef struct bitstream:
        pass

    # Functions
    cdef bitstream* stream_open(void* data, size_t)
    cdef void stream_close(bitstream* stream)

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

    ctypedef enum zfp_exec_policy:
        zfp_exec_serial = 0,
        zfp_exec_omp    = 1,
        zfp_exec_cuda   = 2 

    # Structs
    ctypedef struct zfp_field:
        zfp_type _type "type"
        size_t nx, ny, nz, nw
        ptrdiff_t sx, sy, sz, sw
        void* data

    ctypedef struct zfp_stream:
        pass

    # Types
    ctypedef int zfp_bool

    # Defines
    cython.uint ZFP_HEADER_MAGIC
    cython.uint ZFP_HEADER_META
    cython.uint ZFP_HEADER_MODE
    cython.uint ZFP_HEADER_FULL

    # Functions
    cdef zfp_stream* zfp_stream_open(bitstream* stream)
    cdef void zfp_stream_close(zfp_stream* stream)
    cdef stdint.uint64_t zfp_stream_mode(const zfp_stream* zfp)
    cdef size_t zfp_stream_maximum_size(const zfp_stream* stream, const zfp_field* field)
    cdef void zfp_stream_rewind(zfp_stream* stream)
    cdef void zfp_stream_set_bit_stream(zfp_stream* stream, bitstream* bs)
    cdef void zfp_stream_set_reversible(zfp_stream* stream)
    cdef double zfp_stream_set_rate(zfp_stream* stream, double rate, zfp_type type, cython.uint dims, zfp_bool align)
    cdef cython.uint zfp_stream_set_precision(zfp_stream* stream, cython.uint precision)
    cdef double zfp_stream_set_accuracy(zfp_stream* stream, double tolerance)
    cdef zfp_mode zfp_stream_set_mode(zfp_stream* stream, stdint.uint64_t mode)
    cdef zfp_mode zfp_stream_compression_mode(zfp_stream* stream)
    cdef double zfp_stream_accuracy(zfp_stream* stream)
    cdef double zfp_stream_rate(zfp_stream* stream, cython.uint dims)
    cdef cython.uint zfp_stream_precision(const zfp_stream* stream)
    cdef zfp_field* zfp_field_alloc()
    cdef zfp_field* zfp_field_1d(void* pointer, zfp_type, size_t nx)
    cdef zfp_field* zfp_field_2d(void* pointer, zfp_type, size_t nx, size_t ny)
    cdef zfp_field* zfp_field_3d(void* pointer, zfp_type, size_t nx, size_t ny, size_t nz)
    cdef zfp_field* zfp_field_4d(void* pointer, zfp_type, size_t nx, size_t ny, size_t nz, size_t nw)
    cdef void zfp_field_free(zfp_field* field)
    cdef zfp_bool zfp_field_stride(const zfp_field* field, ptrdiff_t* stride)
    cdef void zfp_field_set_pointer(zfp_field* field, void* pointer) nogil
    cdef zfp_type zfp_field_set_type(zfp_field* field, zfp_type type)
    cdef void zfp_field_set_stride_1d(zfp_field* field, ptrdiff_t sx)
    cdef void zfp_field_set_stride_2d(zfp_field* field, ptrdiff_t sx, ptrdiff_t sy)
    cdef void zfp_field_set_stride_3d(zfp_field* field, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
    cdef void zfp_field_set_stride_4d(zfp_field* field, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
    cdef size_t zfp_compress(zfp_stream* stream, const zfp_field* field) nogil
    cdef size_t zfp_decompress(zfp_stream* stream, zfp_field* field) nogil
    cdef size_t zfp_write_header(zfp_stream* stream, const zfp_field* field, cython.uint mask)
    cdef size_t zfp_read_header(zfp_stream* stream, zfp_field* field, cython.uint mask)
    cdef void zfp_stream_params(zfp_stream* stream, cython.uint* minbits, cython.uint* maxbits, cython.uint* maxprec, int* minexp)
    cdef zfp_bool zfp_stream_set_execution(zfp_stream* stream, zfp_exec_policy policy)
