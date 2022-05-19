#include <assert.h>

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

// we need to know about bitstream, but we don't 
// want duplicate symbols.
#ifndef inline_
  #define inline_ inline
#endif

#include "../inline/bitstream.c"

namespace internal {

// TODO: replace with zfp_field_is_contiguous
bool is_contiguous3d(const uint dims[3], const int3 &stride, long long int &offset)
{
  typedef long long int int64;
  int64 idims[3];
  idims[0] = dims[0];
  idims[1] = dims[1];
  idims[2] = dims[2];

  int64 imin = std::min(stride.x,0) * (idims[0] - 1) + 
               std::min(stride.y,0) * (idims[1] - 1) + 
               std::min(stride.z,0) * (idims[2] - 1);

  int64 imax = std::max(stride.x,0) * (idims[0] - 1) + 
               std::max(stride.y,0) * (idims[1] - 1) + 
               std::max(stride.z,0) * (idims[2] - 1);
  offset = imin;
  int64 ns = idims[0] * idims[1] * idims[2];

  return (imax - imin + 1 == ns);
}

bool is_contiguous2d(const uint dims[3], const int3 &stride, long long int &offset)
{
  typedef long long int int64;
  int64 idims[2];
  idims[0] = dims[0];
  idims[1] = dims[1];

  int64 imin = std::min(stride.x,0) * (idims[0] - 1) + 
               std::min(stride.y,0) * (idims[1] - 1);

  int64  imax = std::max(stride.x,0) * (idims[0] - 1) + 
                std::max(stride.y,0) * (idims[1] - 1); 

  offset = imin;
  return (imax - imin + 1) == (idims[0] * idims[1]);
}

bool is_contiguous1d(uint dim, const int &stride, long long int &offset)
{
  offset = 0;
  if (stride < 0) offset = stride * (int(dim) - 1);
  return std::abs(stride) == 1;
}

bool is_contiguous(const uint dims[3], const int3 &stride, long long int &offset)
{
  int d = 0;

  if (dims[0] != 0) d++;
  if (dims[1] != 0) d++;
  if (dims[2] != 0) d++;

  if (d == 3)
    return is_contiguous3d(dims, stride, offset);
  else if (d == 2)
   return is_contiguous2d(dims, stride, offset);
  else
    return is_contiguous1d(dims[0], stride.x, offset);
}

//
// encode expects device pointers
//
template <typename T>
size_t encode(uint dims[3], int3 stride, int bits_per_block, T *d_data, Word *d_stream)
{
  size_t bits_written = 0;

  int d = 0;
  for (int i = 0; i < 3; ++i)
    if (dims[i] != 0)
      d++;

  ErrorCheck errors;

  if (d == 1) {
    int dim = dims[0];
    int s = stride.x;
    bits_written = hipZFP::encode1<T>(dim, s, d_data, d_stream, bits_per_block); 
  }
  else if (d == 2) {
    uint2 ndims = make_uint2(dims[0], dims[1]);
    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 
    bits_written = hipZFP::encode2<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }
  else if (d == 3) {
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 
    bits_written = hipZFP::encode3<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }

  errors.chk("Encode");

  return bits_written;
}

template <typename T>
size_t decode(uint dims[3], int3 stride, Word *stream, Word *index, T *out, int decode_parameter, uint granularity, zfp_mode mode, zfp_index_type index_type)
{
  size_t bits_read = 0;

  int d = 0;
  for (int i = 0; i < 3; ++i)
    if (dims[i] != 0)
      d++;

  if (d == 1) {
    uint dim = dims[0];
    int sx = stride.x;
    bits_read = hipZFP::decode1<T>(dim, sx, stream, index, out, decode_parameter, granularity, mode, index_type);
  }
  else if (d == 2) {
    uint2 ndims;
    ndims.x = dims[0];
    ndims.y = dims[1];
    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 
    bits_read = hipZFP::decode2<T>(ndims, s, stream, index, out, decode_parameter, granularity, mode, index_type);
  }
  else if (d == 3) {
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 
    bits_read = hipZFP::decode3<T>(ndims, s, stream, index, out, decode_parameter, granularity, mode, index_type);
  }

  return bits_read;
}

Word *setup_device_stream_compress(zfp_stream *stream, const zfp_field *field)
{
  bool stream_device = hipZFP::is_gpu_ptr(stream->stream->begin);
  // TODO: remove all assertions
  assert(sizeof(word) == sizeof(Word)); // CUDA version currently only supports 64bit words

  if (stream_device)
    return (Word*)stream->stream->begin;

  Word *d_stream = NULL;
  size_t max_size = zfp_stream_maximum_size(stream, field);
  if (hipMalloc(&d_stream, max_size) != hipSuccess)
    std::cerr << "failed to allocate device memory for stream" << std::endl;
  return d_stream;
}

Word *setup_device_stream_decompress(zfp_stream *stream, const zfp_field *field)
{
  bool stream_device = hipZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(word) == sizeof(Word)); // CUDA version currently only supports 64bit words;

  if (stream_device)
    return (Word*)stream->stream->begin;

  Word *d_stream = NULL;
  // TODO: change maximum_size to compressed stream size
  size_t size = zfp_stream_maximum_size(stream, field);
  if (hipMalloc(&d_stream, size) != hipSuccess)
    std::cerr << "failed to allocate device memory for stream" << std::endl;
  if (hipMemcpy(d_stream, stream->stream->begin, size, hipMemcpyHostToDevice) != hipSuccess)
    std::cerr << "failed to copy stream from host to device" << std::endl;
  return d_stream;
}

Word *setup_device_index(zfp_stream *stream, const size_t size)
{
  bool stream_device = hipZFP::is_gpu_ptr(stream->index->data);
  assert(sizeof(uint64) == sizeof(Word)); // CUDA version currently only supports 64bit words;

  if (stream_device)
    return (Word*)stream->index->data;

  Word *d_index = NULL;
  if (hipMalloc(&d_index, size) != hipSuccess)
    std::cerr << "failed to allocate device memory for index" << std::endl;
  if (hipMemcpy(d_index, stream->index->data, size, hipMemcpyHostToDevice) != hipSuccess)
    std::cerr << "failed to copy stream from host to device" << std::endl;
  return d_index;
}

void* offset_void(zfp_type type, void *ptr, long long int offset)
{
  void* offset_ptr = NULL;
  if (type == zfp_type_float) {
    float* data = (float*)ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if (type == zfp_type_double) {
    double* data = (double*)ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if (type == zfp_type_int32) {
    int* data = (int*)ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if (type == zfp_type_int64) {
    long long int* data = (long long int*)ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  return offset_ptr;
}

void *setup_device_field_compress(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = hipZFP::is_gpu_ptr(field->data);

  if (field_device) {
    offset = 0;
    return field->data;
  }

  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for (int i = 0; i < 3; i++)
    if (dims[i] != 0)
      field_size *= dims[i];

  bool contig = internal::is_contiguous(dims, stride, offset);

  void* host_ptr = offset_void(field->type, field->data, offset);;

  void *d_data = NULL;
  if (contig) {
    size_t field_bytes = type_size * field_size;
    if (hipMalloc(&d_data, field_bytes) != hipSuccess)
      std::cerr << "failed to allocate device memory for field" << std::endl;
    if (hipMemcpy(d_data, host_ptr, field_bytes, hipMemcpyHostToDevice) != hipSuccess)
      std::cerr << "failed to copy field from host to device" << std::endl;
  }
  return offset_void(field->type, d_data, -offset);
}

void *setup_device_field_decompress(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = hipZFP::is_gpu_ptr(field->data);

  if (field_device) {
    offset = 0;
    return field->data;
  }

  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for (int i = 0; i < 3; i++)
    if (dims[i] != 0)
      field_size *= dims[i];

  bool contig = internal::is_contiguous(dims, stride, offset);

  void* host_ptr = offset_void(field->type, field->data, offset);

  void *d_data = NULL;
  if (contig) {
    size_t field_bytes = type_size * field_size;
    if (hipMalloc(&d_data, field_bytes) != hipSuccess)
      std::cerr << "failed to allocate device memory for field" << std::endl;
  }

  return offset_void(field->type, d_data, -offset);
}


void cleanup_device_ptr(void *orig_ptr, void *d_ptr, size_t bytes, long long int offset, zfp_type type)
{
  bool device = hipZFP::is_gpu_ptr(orig_ptr);
  if (device)
    return;

  // from whence it came
  void *d_offset_ptr = offset_void(type, d_ptr, offset);
  void *h_offset_ptr = offset_void(type, orig_ptr, offset);

  if (bytes > 0)
    hipMemcpy(h_offset_ptr, d_offset_ptr, bytes, hipMemcpyDeviceToHost);

  hipFree(d_offset_ptr);
}

} // namespace internal

size_t
hip_compress(zfp_stream *stream, const zfp_field *field)
{
  // determine field dimensions
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  // determine field strides
  int3 stride;
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;
  
  long long int offset = 0; 
  void *d_data = internal::setup_device_field_compress(field, stride, offset);

  if (d_data == NULL) {
    // null means the array is non-contiguous host mem which is not supported
    return 0;
  }

  Word *d_stream = internal::setup_device_stream_compress(stream, field);

  // encode data
  size_t pos = stream_wtell(stream->stream);
  size_t bits_written = 0;
  if (field->type == zfp_type_float) {
    float* data = (float*)d_data;
    bits_written = internal::encode<float>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if (field->type == zfp_type_double) {
    double* data = (double*)d_data;
    bits_written = internal::encode<double>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if (field->type == zfp_type_int32) {
    int* data = (int*)d_data;
    bits_written = internal::encode<int>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if (field->type == zfp_type_int64) {
    long long int* data = (long long int*)d_data;
    bits_written = internal::encode<long long int>(dims, stride, (int)stream->maxbits, data, d_stream);
  }

  size_t stream_bytes = (bits_written + CHAR_BIT - 1) / CHAR_BIT;
  internal::cleanup_device_ptr(stream->stream->begin, d_stream, stream_bytes, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, 0, offset, field->type);

  // update bit stream to point just past produced data
  if (bits_written)
    stream_wseek(stream->stream, pos + bits_written);

  return bits_written;
}

size_t
hip_decompress(zfp_stream *stream, zfp_field *field)
{
  // determine field dimensions
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  // determine field strides
  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;

  long long int offset = 0;
  void *d_data = internal::setup_device_field_decompress(field, stride, offset);

  if (d_data == NULL) {
    // null means the array is non-contiguous host mem which is not supported
    return 0;
  }

  Word *d_stream = internal::setup_device_stream_decompress(stream, field);
  Word *d_index = NULL;

  // determine number of blocks to decompress
  uint blocks = 1;
  for (int i = 0; i < 3; i++)
    if (dims[i])
      blocks *= (dims[i] + 3) / 4;

  // decode_parameter differs per execution policy
  // TODO: Decide if we want to pass maxbits, minexp and maxprec for all cases or not
  size_t index_size;
  uint granularity;
  int decode_parameter;
  zfp_index_type index_type = zfp_index_none;
  zfp_mode mode = zfp_stream_compression_mode(stream);

  if (mode == zfp_mode_fixed_rate) {
    decode_parameter = (int)stream->maxbits;
    granularity = 1;
  }
  else if (mode == zfp_mode_fixed_precision || mode == zfp_mode_fixed_accuracy) {
    decode_parameter = (mode == zfp_mode_fixed_precision ? (int)stream->maxprec : (int)stream->minexp);
    granularity = stream->index->granularity;
    index_type = stream->index->type;
    uint chunks = (blocks + granularity - 1) / granularity;
    if (index_type == zfp_index_offset)
      index_size = (size_t)chunks * sizeof(uint64);
    else if (index_type == zfp_index_hybrid) {
      // TODO: check if we want to support variable partition size (recommended to not do so for GPU)
      size_t partitions = (chunks + ZFP_PARTITION_SIZE - 1) / ZFP_PARTITION_SIZE;
      index_size = partitions * (sizeof(uint64) + ZFP_PARTITION_SIZE * sizeof(uint16));
    }
    else {
      std::cerr << "zfp unsupported index type for GPU" << std::endl;
      return 0;
    }
    d_index = internal::setup_device_index(stream, index_size);
  }
  else {
    std::cerr << "zfp expert mode not supported on GPU" << std::endl;
    return 0;
  }

  // decode compressed data
  size_t pos = stream_rtell(stream->stream);
  size_t bits_read = 0;
  if (field->type == zfp_type_float) {
    float* data = (float*)d_data;
    bits_read = internal::decode(dims, stride, d_stream, d_index, data, decode_parameter, granularity, mode, index_type);
    d_data = (void*)data;
  }
  else if (field->type == zfp_type_double) {
    double* data = (double*)d_data;
    bits_read = internal::decode(dims, stride, d_stream, d_index, data, decode_parameter, granularity, mode, index_type);
    d_data = (void*)data;
  }
  else if (field->type == zfp_type_int32) {
    int* data = (int*)d_data;
    bits_read = internal::decode(dims, stride, d_stream, d_index, data, decode_parameter, granularity, mode, index_type);
    d_data = (void*)data;
  }
  else if (field->type == zfp_type_int64) {
    long long int* data = (long long int*)d_data;
    bits_read = internal::decode(dims, stride, d_stream, d_index, data, decode_parameter, granularity, mode, index_type);
    d_data = (void*)data;
  }

  // clean up
  size_t bytes_written = zfp_type_size(field->type);
  for (int i = 0; i < 3; ++i)
    if (dims[i] != 0)
      bytes_written *= dims[i];
  internal::cleanup_device_ptr(stream->stream, d_stream, 0, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, bytes_written, offset, field->type);
  if (d_index)
    hipFree(d_index);

  // update bit stream to point just past consumed data
  if (bits_read)
    stream_rseek(stream->stream, pos + bits_read);

  return bits_read;
}
