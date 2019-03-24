import six
import operator
import functools
import cython
from libc.stdlib cimport malloc, free
from cython cimport view
from cpython cimport array
import array

import itertools
if six.PY2:
    from itertools import izip_longest as zip_longest
elif six.PY3:
    from itertools import zip_longest

cimport zfp

import numpy as np
cimport numpy as np

# export #define's
HEADER_MAGIC = ZFP_HEADER_MAGIC
HEADER_META = ZFP_HEADER_META
HEADER_MODE = ZFP_HEADER_MODE
HEADER_FULL = ZFP_HEADER_FULL

# export enums
type_none = zfp_type_none
type_int32 = zfp_type_int32
type_int64 = zfp_type_int64
type_float = zfp_type_float
type_double = zfp_type_double
mode_null = zfp_mode_null
mode_expert = zfp_mode_expert
mode_fixed_rate = zfp_mode_fixed_rate
mode_fixed_precision = zfp_mode_fixed_precision
mode_fixed_accuracy = zfp_mode_fixed_accuracy


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

    strides = [int(x) / arr.itemsize for x in arr.strides[:ndim]]

    if ndim == 1:
        field = zfp_field_1d(pointer, ztype, shape[0])
        zfp_field_set_stride_1d(field, strides[0])
    elif ndim == 2:
        field = zfp_field_2d(pointer, ztype, shape[0], shape[1])
        zfp_field_set_stride_2d(field, strides[0], strides[1])
    elif ndim == 3:
        field = zfp_field_3d(pointer, ztype, shape[0], shape[1], shape[2])
        zfp_field_set_stride_3d(field, strides[0], strides[1], strides[2])
    elif ndim == 4:
        field = zfp_field_4d(pointer, ztype, shape[0], shape[1], shape[2], shape[3])
        zfp_field_set_stride_4d(field, strides[0], strides[1], strides[2], strides[3])
    else:
        raise RuntimeError("Greater than 4 dimensions not supported")

    return field

cdef gen_padded_int_list(orig_array, pad=0, length=4):
    return [int(x) for x in
            itertools.islice(
                itertools.chain(
                    orig_array,
                    itertools.repeat(pad)
                ),
                length
            )
    ]

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

cpdef bytes compress_numpy(np.ndarray arr, double tolerance = -1,
                           double rate = -1, int precision = -1,
                           write_header=True):
    # Input validation
    if arr is None:
        raise TypeError("Input array cannot be None")
    num_params_set = sum([1 for x in [tolerance, rate, precision] if x >= 0])
    if num_params_set == 0:
        raise ValueError("Either tolerance, rate, or precision must be set")
    elif num_params_set > 1:
        raise ValueError("Only one of tolerance, rate, or precision can be set")

    # Setup zfp structs to begin compression
    cdef zfp_field* field = _init_field(arr)
    cdef zfp_stream* stream = zfp_stream_open(NULL)

    cdef zfp_type ztype = zfp_type_none;
    cdef int ndim = arr.ndim;
    set_compression_mode(stream, ztype, ndim, tolerance, rate, precision)

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
        if write_header:
            zfp_write_header(stream, field, HEADER_FULL)
        compressed_size = zfp_compress(stream, field)
        # copy the compressed data into a perfectly sized bytes object
        compress_str = (<char *>data)[:compressed_size]

    zfp_field_free(field)
    zfp_stream_close(stream)
    stream_close(bstream)

    return compress_str

cdef np.ndarray _decompress_with_view(zfp_field* field, zfp_stream* stream):
    cdef zfp_type ztype = field[0]._type
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
    return decomp_arr

cdef _decompress_with_user_array(
    zfp_field* field,
    zfp_stream* stream,
    void* out,
):
    zfp_field_set_pointer(field, out)
    if zfp_decompress(stream, field) == 0:
        raise RuntimeError("error during zfp decompression")

cdef set_compression_mode(
    zfp_stream *stream,
    zfp_type ztype,
    int ndim,
    double tolerance = -1,
    double rate = -1,
    int precision = -1,
):
    if tolerance >= 0:
        zfp_stream_set_accuracy(stream, tolerance)
    elif rate >= 0:
        zfp_stream_set_rate(stream, rate, ztype, ndim, 0)
    elif precision >= 0:
        zfp_stream_set_precision(stream, precision)
    else:
        raise ValueError("Either tolerance, rate, or precision must be set")

cpdef np.ndarray decompress_numpy(
    bytes compressed_data,
    out=None,
    ztype=type_none,
    shape=None,
    strides=[0,0,0,0],
    double tolerance = -1,
    double rate = -1,
    int precision = -1,
):

    if compressed_data is None:
        raise TypeError("compressed_data cannot be None")
    if compressed_data is out:
        raise ValueError("Cannot decompress in-place")

    cdef char* comp_data_pointer = compressed_data
    cdef bitstream* bstream = stream_open(comp_data_pointer, len(compressed_data))
    cdef zfp_field* field = zfp_field_alloc()
    cdef zfp_stream* stream = zfp_stream_open(bstream)
    cdef np.ndarray output

    try:
        zfp_stream_rewind(stream)
        if zfp_read_header(stream, field, HEADER_FULL) == 0:
            zfp_stream_rewind(stream)
            if ztype == type_none or shape is None:
                raise ValueError(
                    "Failed to read zfp header and the ztype/shape/mode "
                    "were not provided"
                )
            try:
                if len(shape) > 4:
                    raise ValueError(
                        "User-provided shape has too many dimensions "
                        "(up to 4 supported)"
                    )
                elif len(shape) <= 0:
                    raise ValueError(
                        "User-provided shape needs at least one dimension"
                    )
                else: # pad the shape with zeros to reach len == 4
                    zshape = gen_padded_int_list(shape, pad=0, length=4)
            except TypeError:
                raise TypeError(
                    "User-provided shape is not an iterable"
                )
            # set the shape, type, and compression mode
            # strides are set further down
            field[0].nx, field[0].ny, field[0].nz, field[0].nw = zshape
            zfp_field_set_type(field, ztype)
            ndim = sum([1 for x in zshape if x > 0])
            set_compression_mode(stream, ztype, ndim, tolerance, rate, precision)
        else: # Check that user-inputs match the header
            if ztype != type_none and ztype is not field[0]._type:
                raise ValueError(
                    "User-provided zfp_type does not match zfp_type in header"
                )
            # check that the header and user shapes match
            header_shape = (field[0].nx, field[0].ny, field[0].nz, field[0].nw)
            header_shape = [x for x in header_shape if x > 0]
            if shape is not None:
                if not all([x == y for x, y in zip_longest(shape, header_shape)]):
                   raise ValueError(
                       "User-provided shape does not match shape in header"
                   )
            # check that setting the compression parameters based on user input
            # does not change the stream mode (i.e., the compression parameters
            # provided by the user and in the header match)
            if tolerance >= 0 or rate >= 0 or precision >= 0:
                header_mode = zfp_stream_mode(stream)
                ndim = len(header_shape)
                set_compression_mode(
                    stream,
                    ztype,
                    ndim,
                    tolerance,
                    rate,
                    precision
                )
                mode = zfp_stream_mode(stream)
                if mode != mode_null and mode != header_mode:
                    raise ValueError(
                        "User-provided zfp_mode {} does not match zfp_mode "
                        "in header {}".format(
                            mode,
                            header_mode,
                        )
                    )
            # check that the shape and strides have the same number of dimensions
            if shape is not None and sum(strides) > 0:
                stride_dims = sum([1 for x in strides if x > 0])
                shape_dims = sum([1 for x in shape if x > 0])
                if len(strides) != len(shape):
                    raise ValueError(
                        "Mis-match in shape and stride lengths"
                    )
        try:
            if len(strides) > 4:
                raise ValueError(
                    "User-provided strides has too many dimensions "
                    "(up to 4 supported)"
                )
            elif len(strides) <= 0:
                raise ValueError(
                        "User-provided strides needs at least one dimension"
                )
            else: # pad the shape with zeros to reach len == 4
                strides = gen_padded_int_list(strides, pad=0, length=4)
        except TypeError:
            raise TypeError(
                "User-provided strides is not an iterable"
            )

        field[0].sx, field[0].sy, field[0].sz, field[0].sw = strides

        if out is None:
            output = np.asarray(_decompress_with_view(field, stream))
        else:
            header_dtype = ztype_to_dtype(field[0]._type)
            header_shape = (field[0].nx, field[0].ny, field[0].nz, field[0].nw)
            header_shape = [x for x in header_shape if x > 0]
            if isinstance(out, np.ndarray):
                if out.dtype != header_dtype:
                    raise ValueError(
                        "Out ndarray has dtype {} but decompression is using "
                        "{}".format(
                            out.dtype,
                            header_dtype
                        )
                    )
                numpy_shape = [int(x) for x in out.shape[:ndim]]
                if not all([x == y for x, y in zip_longest(numpy_shape, header_shape)]):
                    raise ValueError(
                        "Out ndarray has shape {} but decompression is using "
                        "{}".format(
                            numpy_shape,
                            header_shape
                        )
                    )
                output = out
            else:
                output = np.frombuffer(out, dtype=header_dtype)
                output = output.reshape(header_shape)

            _decompress_with_user_array(field, stream, <void *>output.data)
    finally:
        zfp_field_free(field)
        zfp_stream_close(stream)
        stream_close(bstream)

    return output
