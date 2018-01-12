#undef ZFP_VERSIONING_PREFIX_H

/* enums */
#undef zfp_type_none
#undef zfp_type_int32
#undef zfp_type_int64
#undef zfp_type_float
#undef zfp_type_double

/* constant variables */
#undef zfp_codec_version
#undef zfp_library_version
#undef zfp_version_string

/* types */
#undef zfp_stream
#undef zfp_type
#undef zfp_field

/* functions */
/* high-level API */
#undef zfp_type_size
#undef zfp_stream_open
#undef zfp_stream_close

#undef zfp_stream_bit_stream
#undef zfp_stream_mode
#undef zfp_stream_params
#undef zfp_stream_compressed_size
#undef zfp_stream_maximum_size

#undef zfp_stream_set_bit_stream
#undef zfp_stream_set_rate
#undef zfp_stream_set_precision
#undef zfp_stream_set_accuracy
#undef zfp_stream_set_mode
#undef zfp_stream_set_params

#undef zfp_field_alloc
#undef zfp_field_1d
#undef zfp_field_2d
#undef zfp_field_3d
#undef zfp_field_free

#undef zfp_field_pointer
#undef zfp_field_type
#undef zfp_field_precision
#undef zfp_field_dimensionality
#undef zfp_field_size
#undef zfp_field_stride
#undef zfp_field_metadata

#undef zfp_field_set_pointer
#undef zfp_field_set_type
#undef zfp_field_set_size_1d
#undef zfp_field_set_size_2d
#undef zfp_field_set_size_3d
#undef zfp_field_set_stride_1d
#undef zfp_field_set_stride_2d
#undef zfp_field_set_stride_3d
#undef zfp_field_set_metadata

#undef zfp_compress
#undef zfp_decompress
#undef zfp_write_header
#undef zfp_read_header

/* low-level API: stream manipulation */
#undef zfp_stream_flush
#undef zfp_stream_align
#undef zfp_stream_rewind

/* low-level API: encoder */
#undef zfp_encode_block_int32_1
#undef zfp_encode_block_int64_1
#undef zfp_encode_block_float_1
#undef zfp_encode_block_double_1

#undef zfp_encode_block_strided_int32_1
#undef zfp_encode_block_strided_int64_1
#undef zfp_encode_block_strided_float_1
#undef zfp_encode_block_strided_double_1
#undef zfp_encode_partial_block_strided_int32_1
#undef zfp_encode_partial_block_strided_int64_1
#undef zfp_encode_partial_block_strided_float_1
#undef zfp_encode_partial_block_strided_double_1

#undef zfp_encode_block_int32_2
#undef zfp_encode_block_int64_2
#undef zfp_encode_block_float_2
#undef zfp_encode_block_double_2

#undef zfp_encode_block_strided_int32_2
#undef zfp_encode_block_strided_int64_2
#undef zfp_encode_block_strided_float_2
#undef zfp_encode_block_strided_double_2
#undef zfp_encode_partial_block_strided_int32_2
#undef zfp_encode_partial_block_strided_int64_2
#undef zfp_encode_partial_block_strided_float_2
#undef zfp_encode_partial_block_strided_double_2

#undef zfp_encode_block_int32_3
#undef zfp_encode_block_int64_3
#undef zfp_encode_block_float_3
#undef zfp_encode_block_double_3

#undef zfp_encode_block_strided_int32_3
#undef zfp_encode_block_strided_int64_3
#undef zfp_encode_block_strided_float_3
#undef zfp_encode_block_strided_double_3
#undef zfp_encode_partial_block_strided_int32_3
#undef zfp_encode_partial_block_strided_int64_3
#undef zfp_encode_partial_block_strided_float_3
#undef zfp_encode_partial_block_strided_double_3

/* low-level API: decoder */
#undef zfp_decode_block_int32_1
#undef zfp_decode_block_int64_1
#undef zfp_decode_block_float_1
#undef zfp_decode_block_double_1

#undef zfp_decode_block_strided_int32_1
#undef zfp_decode_block_strided_int64_1
#undef zfp_decode_block_strided_float_1
#undef zfp_decode_block_strided_double_1
#undef zfp_decode_partial_block_strided_int32_1
#undef zfp_decode_partial_block_strided_int64_1
#undef zfp_decode_partial_block_strided_float_1
#undef zfp_decode_partial_block_strided_double_1

#undef zfp_decode_block_int32_2
#undef zfp_decode_block_int64_2
#undef zfp_decode_block_float_2
#undef zfp_decode_block_double_2

#undef zfp_decode_block_strided_int32_2
#undef zfp_decode_block_strided_int64_2
#undef zfp_decode_block_strided_float_2
#undef zfp_decode_block_strided_double_2
#undef zfp_decode_partial_block_strided_int32_2
#undef zfp_decode_partial_block_strided_int64_2
#undef zfp_decode_partial_block_strided_float_2
#undef zfp_decode_partial_block_strided_double_2

#undef zfp_decode_block_int32_3
#undef zfp_decode_block_int64_3
#undef zfp_decode_block_float_3
#undef zfp_decode_block_double_3

#undef zfp_decode_block_strided_int32_3
#undef zfp_decode_block_strided_int64_3
#undef zfp_decode_block_strided_float_3
#undef zfp_decode_block_strided_double_3
#undef zfp_decode_partial_block_strided_int32_3
#undef zfp_decode_partial_block_strided_int64_3
#undef zfp_decode_partial_block_strided_float_3
#undef zfp_decode_partial_block_strided_double_3

/* low-level API: utility functions */
#undef zfp_promote_int8_to_int32
#undef zfp_promote_uint8_to_int32
#undef zfp_promote_int16_to_int32
#undef zfp_promote_uint16_to_int32

#undef zfp_demote_int32_to_int8
#undef zfp_demote_int32_to_uint8
#undef zfp_demote_int32_to_int16
#undef zfp_demote_int32_to_uint16
