#include "hipZFP.h"
#include "encode1.h"
#include "encode2.h"
#include "encode3.h"
#include "decode1.h"
#include "decode2.h"
#include "decode3.h"
#include "ErrorCheck.h"
#include "pointers.h"
#include "type_info.h"

// we need to know about bitstream, but we don't want duplicate symbols
#ifndef inline_
  #define inline_ inline
#endif

#include "../inline/bitstream.c"

namespace internal {

// advance pointer from d_begin to address difference between h_ptr and h_begin
template <typename T>
void* device_pointer(void* d_begin, void* h_begin, void* h_ptr)
{
  return (void*)((T*)d_begin + ((T*)h_ptr - (T*)h_begin));
}

void* device_pointer(void* d_begin, void* h_begin, void* h_ptr, zfp_type type)
{
  switch (type) {
    case zfp_type_int32:  return device_pointer<int>(d_begin, h_begin, h_ptr);
    case zfp_type_int64:  return device_pointer<long long int>(d_begin, h_begin, h_ptr);
    case zfp_type_float:  return device_pointer<float>(d_begin, h_begin, h_ptr);
    case zfp_type_double: return device_pointer<double>(d_begin, h_begin, h_ptr);
    default:              return NULL;
  }
}

Word* setup_device_stream_compress(zfp_stream* stream)
{
  Word* d_stream = (Word*)stream->stream->begin;
  if (!hipZFP::is_gpu_ptr(d_stream)) {
    // allocate device memory for compressed data
    size_t size = stream_capacity(stream->stream);
    if (hipMalloc(&d_stream, size) != hipSuccess)
      std::cerr << "failed to allocate device memory for stream" << std::endl;
  }

  return d_stream;
}

Word* setup_device_stream_decompress(zfp_stream* stream)
{
  Word* d_stream = (Word*)stream->stream->begin;
  if (!hipZFP::is_gpu_ptr(d_stream)) {
    // copy compressed data to device memory
    size_t size = stream_capacity(stream->stream);
    if (hipMalloc(&d_stream, size) != hipSuccess) {
      std::cerr << "failed to allocate device memory for stream" << std::endl;
      return NULL;
    }
    if (hipMemcpy(d_stream, stream->stream->begin, size, hipMemcpyHostToDevice) != hipSuccess) {
      std::cerr << "failed to copy stream from host to device" << std::endl;
      hipFree(d_stream);
      return NULL;
    }
  }

  return d_stream;
}

Word* setup_device_index(zfp_stream* stream)
{
  Word* d_index = (Word*)stream->index->data;
  if (!hipZFP::is_gpu_ptr(d_index)) {
    // copy index to device memory
    size_t size = stream->index->size;
    if (hipMalloc(&d_index, size) != hipSuccess) {
      std::cerr << "failed to allocate device memory for index" << std::endl;
      return NULL;
    }
    if (hipMemcpy(d_index, stream->index->data, size, hipMemcpyHostToDevice) != hipSuccess) {
      std::cerr << "failed to copy stream from host to device" << std::endl;
      hipFree(d_index);
      return NULL;
    }
  }

  return d_index;
}

void* setup_device_field_compress(const zfp_field* field, void*& d_begin)
{
  void* d_data = field->data;
  if (hipZFP::is_gpu_ptr(d_data)) {
    // field already resides on device
    d_begin = zfp_field_begin(field);
    return d_data;
  }
  else {
    // GPU implementation currently requires contiguous field
    if (zfp_field_is_contiguous(field)) {
      // copy field from host to device
      size_t size = zfp_field_size(field, NULL) * zfp_type_size(field->type);
      if (hipMalloc(&d_begin, size) != hipSuccess) {
        std::cerr << "failed to allocate device memory for field" << std::endl;
        return NULL;
      }
      // in case of negative strides, find lowest memory address spanned by field
      void* h_begin = zfp_field_begin(field);
      if (hipMemcpy(d_begin, h_begin, size, hipMemcpyHostToDevice) != hipSuccess) {
        std::cerr << "failed to copy field from host to device" << std::endl;
        hipFree(d_begin);
        return NULL;
      }
      // in case of negative strides, advance device pointer into buffer
      return device_pointer(d_begin, h_begin, d_data, field->type);
    }
    else
      return NULL;
  }
}

void* setup_device_field_decompress(const zfp_field* field, void*& d_begin)
{
  void* d_data = field->data;
  if (hipZFP::is_gpu_ptr(d_data)) {
    // field has already been allocated on device
    d_begin = zfp_field_begin(field);
    return d_data;
  }
  else {
    // GPU implementation currently requires contiguous field
    if (zfp_field_is_contiguous(field)) {
      // allocate device memory for decompressed field
      size_t size = zfp_field_size(field, NULL) * zfp_type_size(field->type);
      if (hipMalloc(&d_begin, size) != hipSuccess) {
        std::cerr << "failed to allocate device memory for field" << std::endl;
        return NULL;
      }
      void* h_begin = zfp_field_begin(field);
      // in case of negative strides, advance device pointer into buffer
      return device_pointer(d_begin, h_begin, d_data, field->type);
    }
    else
      return NULL;
  }
}

// copy from device to host (if needed) and deallocate device memory
void cleanup_device(void* begin, void* d_begin, size_t bytes = 0)
{
  if (begin != d_begin) {
    // copy data from device to host and free device memory
    if (bytes)
      hipMemcpy(begin, d_begin, bytes, hipMemcpyDeviceToHost);
    hipFree(d_begin);
  }
}

// encode field from d_data to d_stream
template <typename T>
size_t
encode(
  const T* d_data,          // field data device pointer
  const size_t size[],      // field dimensions
  const ptrdiff_t stride[], // field strides
  Word* d_stream,           // compressed bit stream device pointer
  uint maxbits              // compressed #bits/block
)
{
  size_t bits_written = 0;

  ErrorCheck errors;

  uint dims = size[0] ? size[1] ? size[2] ? 3 : 2 : 1 : 0;
  switch (dims) {
    case 1:
      bits_written = hipZFP::encode1<T>(d_data, size, stride, d_stream, maxbits);
      break;
    case 2:
      bits_written = hipZFP::encode2<T>(d_data, size, stride, d_stream, maxbits);
      break;
    case 3:
      bits_written = hipZFP::encode3<T>(d_data, size, stride, d_stream, maxbits);
      break;
    default:
      break;
  }

  errors.chk("Encode");

  return bits_written;
}

// decode field from d_stream to d_data
template <typename T>
size_t
decode(
  T* d_data,                 // field data device pointer
  const size_t size[],       // field dimensions
  const ptrdiff_t stride[],  // field strides
  const Word* d_stream,      // compressed bit stream device pointer
  zfp_mode mode,             // compression mode
  int decode_parameter,      // compression parameter
  const Word* d_index,       // block index device pointer
  zfp_index_type index_type, // block index type
  uint granularity           // block index granularity in blocks/entry
)
{
  size_t bits_read = 0;

  ErrorCheck errors;

  uint dims = size[0] ? size[1] ? size[2] ? 3 : 2 : 1 : 0;
  switch (dims) {
    case 1:
      bits_read = hipZFP::decode1<T>(d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    case 2:
      bits_read = hipZFP::decode2<T>(d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    case 3:
      bits_read = hipZFP::decode3<T>(d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    default:
      break;
  }

  errors.chk("Decode");

  return bits_read;
}

} // namespace internal

// TODO: move out of global namespace
zfp_bool
hip_init(zfp_stream* stream)
{
  // ensure GPU word size equals CPU word size
  if (sizeof(Word) != sizeof(word))
    return false;

  static bool initialized = false;
  static hipDeviceProp_t prop;
  if (!initialized && hipGetDeviceProperties(&prop, 0) != hipSuccess)
    return zfp_false;

  initialized = true;
  // TODO: take advantage of cached grid size
  stream->exec.params.hip.grid_size[0] = prop.maxGridSize[0];
  stream->exec.params.hip.grid_size[1] = prop.maxGridSize[1];
  stream->exec.params.hip.grid_size[2] = prop.maxGridSize[2];

  // TODO: launch warm-up kernel

  return zfp_true;
}

size_t
hip_compress(zfp_stream* stream, const zfp_field* field)
{
  // determine field dimensions
  size_t size[3];
  size[0] = field->nx;
  size[1] = field->ny;
  size[2] = field->nz;

  // determine field strides
  ptrdiff_t stride[3];
  stride[0] = field->sx ? field->sx : 1;
  stride[1] = field->sy ? field->sy : (ptrdiff_t)field->nx;
  stride[2] = field->sz ? field->sz : (ptrdiff_t)field->nx * (ptrdiff_t)field->ny;

  void* d_begin = NULL;
  void* d_data = internal::setup_device_field_compress(field, d_begin);

  // null means the array is non-contiguous host memory, which is not supported
  if (!d_data)
    return 0;

  Word* d_stream = internal::setup_device_stream_compress(stream);

  // encode data
  size_t bits_written = 0;
  size_t pos = stream_wtell(stream->stream);
  switch (field->type) {
    case zfp_type_int32:
      bits_written = internal::encode((int*)d_data, size, stride, d_stream, stream->maxbits);
      break;
    case zfp_type_int64:
      bits_written = internal::encode((long long int*)d_data, size, stride, d_stream, stream->maxbits);
      break;
    case zfp_type_float:
      bits_written = internal::encode((float*)d_data, size, stride, d_stream, stream->maxbits);
      break;
    case zfp_type_double:
      bits_written = internal::encode((double*)d_data, size, stride, d_stream, stream->maxbits);
      break;
    default:
      break;
  }

  // copy stream from device to host if needed and free temporary buffers
  size_t stream_bytes = hipZFP::round_up((bits_written + CHAR_BIT - 1) / CHAR_BIT, sizeof(Word));
  internal::cleanup_device(stream->stream->begin, d_stream, stream_bytes);
  internal::cleanup_device(zfp_field_begin(field), d_begin);

  // update bit stream to point just past produced data
  if (bits_written)
    stream_wseek(stream->stream, pos + bits_written);

  return bits_written;
}

size_t
hip_decompress(zfp_stream* stream, zfp_field* field)
{
  // determine field dimensions
  size_t size[3];
  size[0] = field->nx;
  size[1] = field->ny;
  size[2] = field->nz;

  // determine field strides
  ptrdiff_t stride[3];
  stride[0] = field->sx ? field->sx : 1;
  stride[1] = field->sy ? field->sy : (ptrdiff_t)field->nx;
  stride[2] = field->sz ? field->sz : (ptrdiff_t)field->nx * (ptrdiff_t)field->ny;

  void* d_begin;
  void* d_data = internal::setup_device_field_decompress(field, d_begin);

  // null means the array is non-contiguous host memory, which is not supported
  if (!d_data)
    return 0;

  Word* d_stream = internal::setup_device_stream_decompress(stream);
  Word* d_index = NULL;

  // decode_parameter differs per execution policy
  zfp_mode mode = zfp_stream_compression_mode(stream);
  int decode_parameter;
  zfp_index_type index_type = zfp_index_none;
  uint granularity;

  switch (mode) {
    case zfp_mode_fixed_rate:
      decode_parameter = (int)stream->maxbits;
      granularity = 1;
      break;
    case zfp_mode_fixed_precision:
    case zfp_mode_fixed_accuracy:
      decode_parameter = (mode == zfp_mode_fixed_precision ? (int)stream->maxprec : (int)stream->minexp);
      if (!stream->index) {
        std::cerr << "zfp variable-rate decompression requires block index" << std::endl;
        return 0;
      }
      index_type = stream->index->type;
      if (index_type != zfp_index_offset && index_type != zfp_index_hybrid) {
        std::cerr << "zfp index type not supported on GPU" << std::endl;
        return 0;
      }
      granularity = stream->index->granularity;
      d_index = internal::setup_device_index(stream);
      break;
    default:
      std::cerr << "zfp compression mode not supported on GPU" << std::endl;
      return 0;
  }

  // decode compressed data
  size_t bits_read = 0;
  size_t pos = stream_rtell(stream->stream);
  switch (field->type) {
    case zfp_type_int32:
      bits_read = internal::decode((int*)d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    case zfp_type_int64:
      bits_read = internal::decode((long long int*)d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    case zfp_type_float:
      bits_read = internal::decode((float*)d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    case zfp_type_double:
      bits_read = internal::decode((double*)d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    default:
      break;
  }

  // copy field from device to host if needed and free temporary buffers
  size_t field_bytes = zfp_field_size(field, NULL) * zfp_type_size(field->type);
  internal::cleanup_device(zfp_field_begin(field), d_begin, field_bytes);
  internal::cleanup_device(stream->stream->begin, d_stream);
  if (d_index)
    internal::cleanup_device(stream->index->data, d_index);

  // update bit stream to point just past consumed data
  if (bits_read)
    stream_rseek(stream->stream, pos + bits_read);

  return bits_read;
}
