import cython
cimport libc.stdint as stdint

cdef extern from "bitstream.h":
    cdef struct bitstream:
        pass
    bitstream* stream_open(void* data, size_t);
    void stream_close(bitstream* stream);

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
        zfp_mode_fixed_accuracy  = 4

    # structs
    ctypedef struct zfp_field:
        zfp_type _type "type"
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
    double zfp_stream_set_rate(zfp_stream* stream, double rate, zfp_type type, cython.uint dims, int wra);
    void zfp_stream_set_reversible(zfp_stream* stream);
    stdint.uint64_t zfp_stream_mode(const zfp_stream* zfp);
    zfp_mode zfp_stream_set_mode(zfp_stream* stream, stdint.uint64_t mode);
    zfp_field* zfp_field_alloc();
    zfp_field* zfp_field_1d(void* pointer, zfp_type, cython.uint nx);
    zfp_field* zfp_field_2d(void* pointer, zfp_type, cython.uint nx, cython.uint ny);
    zfp_field* zfp_field_3d(void* pointer, zfp_type, cython.uint nx, cython.uint ny, cython.uint nz);
    zfp_field* zfp_field_4d(void* pointer, zfp_type, cython.uint nx, cython.uint ny, cython.uint nz, cython.uint nw);
    void zfp_field_set_stride_1d(zfp_field* field, int sx);
    void zfp_field_set_stride_2d(zfp_field* field, int sx, int sy);
    void zfp_field_set_stride_3d(zfp_field* field, int sx, int sy, int sz);
    void zfp_field_set_stride_4d(zfp_field* field, int sx, int sy, int sz, int sw);
    int zfp_field_stride(const zfp_field* field, int* stride)
    void zfp_field_free(zfp_field* field);
    zfp_type zfp_field_set_type(zfp_field* field, zfp_type type);
    size_t zfp_compress(zfp_stream* stream, const zfp_field* field) nogil;
    size_t zfp_decompress(zfp_stream* stream, zfp_field* field) nogil;
    size_t zfp_write_header(zfp_stream* stream, const zfp_field* field, cython.uint mask);
    size_t zfp_read_header(zfp_stream* stream, zfp_field* field, cython.uint mask);
    void zfp_stream_rewind(zfp_stream* stream);
    void zfp_field_set_pointer(zfp_field* field, void* pointer) nogil;

cdef gen_padded_int_list(orig_array, pad=*, length=*)
