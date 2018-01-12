#ifndef ZFP_VERSIONING_PREFIX_H
#define ZFP_VERSIONING_PREFIX_H

/* enums */
#define zfp_type_none zfp_v5_type_none
#define zfp_type_int32 zfp_v5_type_int32
#define zfp_type_int64 zfp_v5_type_int64
#define zfp_type_float zfp_v5_type_float
#define zfp_type_double zfp_v5_type_double

/* constant variables */
#define zfp_codec_version zfp_v5_codec_version
#define zfp_library_version zfp_v5_library_version
#define zfp_version_string zfp_v5_version_string

/* types */
#define zfp_stream zfp_v5_stream
#define zfp_type zfp_v5_type
#define zfp_field zfp_v5_field

/* functions */
/* high-level API */
#define zfp_type_size(x) zfp_v5_type_size(x)
#define zfp_stream_open(x) zfp_v5_stream_open(x)
#define zfp_stream_close(x) zfp_v5_stream_close(x)

#define zfp_stream_bit_stream(x) zfp_v5_stream_bit_stream(x)
#define zfp_stream_mode(x) zfp_v5_stream_mode(x)
#define zfp_stream_params(x, a, b, c, d) zfp_v5_stream_params(x, a, b, c, d)
#define zfp_stream_compressed_size(x) zfp_v5_stream_compressed_size(x)
#define zfp_stream_maximum_size(x, y) zfp_v5_stream_maximum_size(x, y)

#define zfp_stream_set_bit_stream(x, y) zfp_v5_stream_set_bit_stream(x, y)
#define zfp_stream_set_rate(x, a, b, c, d) zfp_v5_stream_set_rate(x, a, b, c, d)
#define zfp_stream_set_precision(x, a) zfp_v5_stream_set_precision(x, a)
#define zfp_stream_set_accuracy(x, a) zfp_v5_stream_set_accuracy(x, a)
#define zfp_stream_set_mode(x, a) zfp_v5_stream_set_mode(x, a)
#define zfp_stream_set_params(x, a, b, c, d) zfp_v5_stream_set_params(x, a, b, c, d)

#define zfp_field_alloc() zfp_v5_field_alloc()
#define zfp_field_1d(x, a, b) zfp_v5_field_1d(x, a, b)
#define zfp_field_2d(x, a, b, c) zfp_v5_field_2d(x, a, b, c)
#define zfp_field_3d(x, a, b, c, d) zfp_v5_field_3d(x, a, b, c, d)
#define zfp_field_free(x) zfp_v5_field_free(x)

#define zfp_field_pointer(x) zfp_v5_field_pointer(x)
#define zfp_field_type(x) zfp_v5_field_type(x)
#define zfp_field_precision(x) zfp_v5_field_precision(x)
#define zfp_field_dimensionality(x) zfp_v5_field_dimensionality(x)
#define zfp_field_size(x, a) zfp_v5_field_size(x, a)
#define zfp_field_stride(x, a) zfp_v5_field_stride(x, a)
#define zfp_field_metadata(x) zfp_v5_field_metadata(x)

#define zfp_field_set_pointer(x, y) zfp_v5_field_set_pointer(x, y)
#define zfp_field_set_type(x, y) zfp_v5_field_set_type(x, y)
#define zfp_field_set_size_1d(x, a) zfp_v5_field_set_size_1d(x, a)
#define zfp_field_set_size_2d(x, a, b) zfp_v5_field_set_size_2d(x, a, b)
#define zfp_field_set_size_3d(x, a, b, c) zfp_v5_field_set_size_3d(x, a, b, c)
#define zfp_field_set_stride_1d(x, a) zfp_v5_field_set_stride_1d(x, a)
#define zfp_field_set_stride_2d(x, a, b) zfp_v5_field_set_stride_2d(x, a, b)
#define zfp_field_set_stride_3d(x, a, b, c) zfp_v5_field_set_stride_3d(x, a, b, c)
#define zfp_field_set_metadata(x, a) zfp_v5_field_set_metadata(x, a)

#define zfp_compress(x, y) zfp_v5_compress(x, y)
#define zfp_decompress(x, y) zfp_v5_decompress(x, y)
#define zfp_write_header(x, a, b) zfp_v5_write_header(x, a, b)
#define zfp_read_header(x, a, b) zfp_v5_read_header(x, a, b)

/* low-level API: stream manipulation */
#define zfp_stream_flush(x) zfp_v5_stream_flush(x)
#define zfp_stream_align(x) zfp_v5_stream_align(x)
#define zfp_stream_rewind(x) zfp_v5_stream_rewind(x)

/* low-level API: encoder */
#define zfp_encode_block_int32_1(x, y) zfp_v5_encode_block_int32_1(x, y)
#define zfp_encode_block_int64_1(x, y) zfp_v5_encode_block_int64_1(x, y)
#define zfp_encode_block_float_1(x, y) zfp_v5_encode_block_float_1(x, y)
#define zfp_encode_block_double_1(x, y) zfp_v5_encode_block_double_1(x, y)

#define zfp_encode_block_strided_int32_1(x, y, a) zfp_v5_encode_block_strided_int32_1(x, y, a)
#define zfp_encode_block_strided_int64_1(x, y, a) zfp_v5_encode_block_strided_int64_1(x, y, a)
#define zfp_encode_block_strided_float_1(x, y, a) zfp_v5_encode_block_strided_float_1(x, y, a)
#define zfp_encode_block_strided_double_1(x, y, a) zfp_v5_encode_block_strided_double_1(x, y, a)
#define zfp_encode_partial_block_strided_int32_1(x, y, a, b) zfp_v5_encode_partial_block_strided_int32_1(x, y, a, b)
#define zfp_encode_partial_block_strided_int64_1(x, y, a, b) zfp_v5_encode_partial_block_strided_int64_1(x, y, a, b)
#define zfp_encode_partial_block_strided_float_1(x, y, a, b) zfp_v5_encode_partial_block_strided_float_1(x, y, a, b)
#define zfp_encode_partial_block_strided_double_1(x, y, a, b) zfp_v5_encode_partial_block_strided_double_1(x, y, a, b)

#define zfp_encode_block_int32_2(x, y) zfp_v5_encode_block_int32_2(x, y)
#define zfp_encode_block_int64_2(x, y) zfp_v5_encode_block_int64_2(x, y)
#define zfp_encode_block_float_2(x, y) zfp_v5_encode_block_float_2(x, y)
#define zfp_encode_block_double_2(x, y) zfp_v5_encode_block_double_2(x, y)

#define zfp_encode_block_strided_int32_2(x, y, a, b) zfp_v5_encode_block_strided_int32_2(x, y, a, b)
#define zfp_encode_block_strided_int64_2(x, y, a, b) zfp_v5_encode_block_strided_int64_2(x, y, a, b)
#define zfp_encode_block_strided_float_2(x, y, a, b) zfp_v5_encode_block_strided_float_2(x, y, a, b)
#define zfp_encode_block_strided_double_2(x, y, a, b) zfp_v5_encode_block_strided_double_2(x, y, a, b)
#define zfp_encode_partial_block_strided_int32_2(x, y, a, b, c, d) zfp_v5_encode_partial_block_strided_int32_2(x, y, a, b, c, d)
#define zfp_encode_partial_block_strided_int64_2(x, y, a, b, c, d) zfp_v5_encode_partial_block_strided_int64_2(x, y, a, b, c, d)
#define zfp_encode_partial_block_strided_float_2(x, y, a, b, c, d) zfp_v5_encode_partial_block_strided_float_2(x, y, a, b, c, d)
#define zfp_encode_partial_block_strided_double_2(x, y, a, b, c, d) zfp_v5_encode_partial_block_strided_double_2(x, y, a, b, c, d)

#define zfp_encode_block_int32_3(x, y) zfp_v5_encode_block_int32_3(x, y)
#define zfp_encode_block_int64_3(x, y) zfp_v5_encode_block_int64_3(x, y)
#define zfp_encode_block_float_3(x, y) zfp_v5_encode_block_float_3(x, y)
#define zfp_encode_block_double_3(x, y) zfp_v5_encode_block_double_3(x, y)

#define zfp_encode_block_strided_int32_3(x, y, a, b, c) zfp_v5_encode_block_strided_int32_3(x, y, a, b, c)
#define zfp_encode_block_strided_int64_3(x, y, a, b, c) zfp_v5_encode_block_strided_int64_3(x, y, a, b, c)
#define zfp_encode_block_strided_float_3(x, y, a, b, c) zfp_v5_encode_block_strided_float_3(x, y, a, b, c)
#define zfp_encode_block_strided_double_3(x, y, a, b, c) zfp_v5_encode_block_strided_double_3(x, y, a, b, c)
#define zfp_encode_partial_block_strided_int32_3(x, y, a, b, c, d, e, f) zfp_v5_encode_partial_block_strided_int32_3(x, y, a, b, c, d, e, f)
#define zfp_encode_partial_block_strided_int64_3(x, y, a, b, c, d, e, f) zfp_v5_encode_partial_block_strided_int64_3(x, y, a, b, c, d, e, f)
#define zfp_encode_partial_block_strided_float_3(x, y, a, b, c, d, e, f) zfp_v5_encode_partial_block_strided_float_3(x, y, a, b, c, d, e, f)
#define zfp_encode_partial_block_strided_double_3(x, y, a, b, c, d, e, f) zfp_v5_encode_partial_block_strided_double_3(x, y, a, b, c, d, e, f)

/* low-level API: decoder */
#define zfp_decode_block_int32_1(x, y) zfp_v5_decode_block_int32_1(x, y)
#define zfp_decode_block_int64_1(x, y) zfp_v5_decode_block_int64_1(x, y)
#define zfp_decode_block_float_1(x, y) zfp_v5_decode_block_float_1(x, y)
#define zfp_decode_block_double_1(x, y) zfp_v5_decode_block_double_1(x, y)

#define zfp_decode_block_strided_int32_1(x, y, a) zfp_v5_decode_block_strided_int32_1(x, y, a)
#define zfp_decode_block_strided_int64_1(x, y, a) zfp_v5_decode_block_strided_int64_1(x, y, a)
#define zfp_decode_block_strided_float_1(x, y, a) zfp_v5_decode_block_strided_float_1(x, y, a)
#define zfp_decode_block_strided_double_1(x, y, a) zfp_v5_decode_block_strided_double_1(x, y, a)
#define zfp_decode_partial_block_strided_int32_1(x, y, a, b) zfp_v5_decode_partial_block_strided_int32_1(x, y, a, b)
#define zfp_decode_partial_block_strided_int64_1(x, y, a, b) zfp_v5_decode_partial_block_strided_int64_1(x, y, a, b)
#define zfp_decode_partial_block_strided_float_1(x, y, a, b) zfp_v5_decode_partial_block_strided_float_1(x, y, a, b)
#define zfp_decode_partial_block_strided_double_1(x, y, a, b) zfp_v5_decode_partial_block_strided_double_1(x, y, a, b)

#define zfp_decode_block_int32_2(x, y) zfp_v5_decode_block_int32_2(x, y)
#define zfp_decode_block_int64_2(x, y) zfp_v5_decode_block_int64_2(x, y)
#define zfp_decode_block_float_2(x, y) zfp_v5_decode_block_float_2(x, y)
#define zfp_decode_block_double_2(x, y) zfp_v5_decode_block_double_2(x, y)

#define zfp_decode_block_strided_int32_2(x, y, a, b) zfp_v5_decode_block_strided_int32_2(x, y, a, b)
#define zfp_decode_block_strided_int64_2(x, y, a, b) zfp_v5_decode_block_strided_int64_2(x, y, a, b)
#define zfp_decode_block_strided_float_2(x, y, a, b) zfp_v5_decode_block_strided_float_2(x, y, a, b)
#define zfp_decode_block_strided_double_2(x, y, a, b) zfp_v5_decode_block_strided_double_2(x, y, a, b)
#define zfp_decode_partial_block_strided_int32_2(x, y, a, b, c, d) zfp_v5_decode_partial_block_strided_int32_2(x, y, a, b, c, d)
#define zfp_decode_partial_block_strided_int64_2(x, y, a, b, c, d) zfp_v5_decode_partial_block_strided_int64_2(x, y, a, b, c, d)
#define zfp_decode_partial_block_strided_float_2(x, y, a, b, c, d) zfp_v5_decode_partial_block_strided_float_2(x, y, a, b, c, d)
#define zfp_decode_partial_block_strided_double_2(x, y, a, b, c, d) zfp_v5_decode_partial_block_strided_double_2(x, y, a, b, c, d)

#define zfp_decode_block_int32_3(x, y) zfp_v5_decode_block_int32_3(x, y)
#define zfp_decode_block_int64_3(x, y) zfp_v5_decode_block_int64_3(x, y)
#define zfp_decode_block_float_3(x, y) zfp_v5_decode_block_float_3(x, y)
#define zfp_decode_block_double_3(x, y) zfp_v5_decode_block_double_3(x, y)

#define zfp_decode_block_strided_int32_3(x, y, a, b, c) zfp_v5_decode_block_strided_int32_3(x, y, a, b, c)
#define zfp_decode_block_strided_int64_3(x, y, a, b, c) zfp_v5_decode_block_strided_int64_3(x, y, a, b, c)
#define zfp_decode_block_strided_float_3(x, y, a, b, c) zfp_v5_decode_block_strided_float_3(x, y, a, b, c)
#define zfp_decode_block_strided_double_3(x, y, a, b, c) zfp_v5_decode_block_strided_double_3(x, y, a, b, c)
#define zfp_decode_partial_block_strided_int32_3(x, y, a, b, c, d, e, f) zfp_v5_decode_partial_block_strided_int32_3(x, y, a, b, c, d, e, f)
#define zfp_decode_partial_block_strided_int64_3(x, y, a, b, c, d, e, f) zfp_v5_decode_partial_block_strided_int64_3(x, y, a, b, c, d, e, f)
#define zfp_decode_partial_block_strided_float_3(x, y, a, b, c, d, e, f) zfp_v5_decode_partial_block_strided_float_3(x, y, a, b, c, d, e, f)
#define zfp_decode_partial_block_strided_double_3(x, y, a, b, c, d, e, f) zfp_v5_decode_partial_block_strided_double_3(x, y, a, b, c, d, e, f)

/* low-level API: utility functions */
#define zfp_promote_int8_to_int32(x, y, a) zfp_v5_promote_int8_to_int32(x, y, a)
#define zfp_promote_uint8_to_int32(x, y, a) zfp_v5_promote_uint8_to_int32(x, y, a)
#define zfp_promote_int16_to_int32(x, y, a) zfp_v5_promote_int16_to_int32(x, y, a)
#define zfp_promote_uint16_to_int32(x, y, a) zfp_v5_promote_uint16_to_int32(x, y, a)

#define zfp_demote_int32_to_int8(x, y, a) zfp_v5_demote_int32_to_int8(x, y, a)
#define zfp_demote_int32_to_uint8(x, y, a) zfp_v5_demote_int32_to_uint8(x, y, a)
#define zfp_demote_int32_to_int16(x, y, a) zfp_v5_demote_int32_to_int16(x, y, a)
#define zfp_demote_int32_to_uint16(x, y, a) zfp_v5_demote_int32_to_uint16(x, y, a)

#endif
