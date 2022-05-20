#include <assert.h>

#include "cuZFP.h"

#include "encode1.cuh"
#include "encode2.cuh"
#include "encode3.cuh"

#include "decode1.cuh"
#include "decode2.cuh"
#include "decode3.cuh"

#include "ErrorCheck.h"

#include "pointers.cuh"
#include "type_info.cuh"

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
    bits_written = cuZFP::encode1<T>(dim, s, d_data, d_stream, bits_per_block); 
  }
  else if (d == 2) {
    uint2 ndims = make_uint2(dims[0], dims[1]);
    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 
    bits_written = cuZFP::encode2<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }
  else if (d == 3) {
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 
    bits_written = cuZFP::encode3<T>(ndims, s, d_data, d_stream, bits_per_block); 
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
    bits_read = cuZFP::decode1<T>(dim, sx, stream, index, out, decode_parameter, granularity, mode, index_type);
  }
  else if (d == 2) {
    uint2 ndims;
    ndims.x = dims[0];
    ndims.y = dims[1];
    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 
    bits_read = cuZFP::decode2<T>(ndims, s, stream, index, out, decode_parameter, granularity, mode, index_type);
  }
  else if (d == 3) {
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 
    bits_read = cuZFP::decode3<T>(ndims, s, stream, index, out, decode_parameter, granularity, mode, index_type);
  }

  return bits_read;
}

Word* setup_device_stream_compress(zfp_stream *stream, const zfp_field *field)
{
  // TODO: remove all assertions
  assert(sizeof(word) == sizeof(Word)); // GPU version currently only supports 64-bit words

  Word* d_stream = (Word*)stream->stream->begin;
  if (!cuZFP::is_gpu_ptr(d_stream)) {
    // allocate device memory for compressed data
    size_t size = zfp_stream_maximum_size(stream, field);
    if (cudaMalloc(&d_stream, size) != cudaSuccess)
      std::cerr << "failed to allocate device memory for stream" << std::endl;
  }

  return d_stream;
}

Word* setup_device_stream_decompress(zfp_stream* stream)
{
  assert(sizeof(word) == sizeof(Word)); // GPU version currently only supports 64-bit words

  Word* d_stream = (Word*)stream->stream->begin;
  if (!cuZFP::is_gpu_ptr(d_stream)) {
    // copy compressed data to device memory
    size_t size = stream_capacity(stream->stream);
    if (cudaMalloc(&d_stream, size) != cudaSuccess)
      std::cerr << "failed to allocate device memory for stream" << std::endl;
    if (cudaMemcpy(d_stream, stream->stream->begin, size, cudaMemcpyHostToDevice) != cudaSuccess)
      std::cerr << "failed to copy stream from host to device" << std::endl;
  }

  return d_stream;
}

Word* setup_device_index(zfp_stream* stream)
{
  assert(sizeof(uint64) == sizeof(Word)); // GPU version currently only supports 64-bit words

  Word* d_index = (Word*)stream->index->data;
  if (!cuZFP::is_gpu_ptr(d_index)) {
    size_t size = stream->index->size;
    if (cudaMalloc(&d_index, size) != cudaSuccess)
      std::cerr << "failed to allocate device memory for index" << std::endl;
    if (cudaMemcpy(d_index, stream->index->data, size, cudaMemcpyHostToDevice) != cudaSuccess)
      std::cerr << "failed to copy stream from host to device" << std::endl;
  }

  return d_index;
}

void* offset_void(zfp_type type, void* ptr, long long int offset)
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

void* setup_device_field_compress(const zfp_field* field, const int3& stride, long long int& offset)
{
  if (cuZFP::is_gpu_ptr(field->data)) {
    // field already resides on device
    offset = 0;
    return field->data;
  }
  else {
    uint dims[3];
    dims[0] = field->nx;
    dims[1] = field->ny;
    dims[2] = field->nz;
    // GPU implementation currently requires contiguous field
    if (internal::is_contiguous(dims, stride, offset)) {
      // copy field from host to device
      void* d_data = NULL;
      void* h_data = offset_void(field->type, field->data, offset);
      size_t size = zfp_field_size(field, NULL) * zfp_type_size(field->type);
      if (cudaMalloc(&d_data, size) != cudaSuccess)
        std::cerr << "failed to allocate device memory for field" << std::endl;
      if (cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice) != cudaSuccess)
        std::cerr << "failed to copy field from host to device" << std::endl;
      return offset_void(field->type, d_data, -offset);
    }
    else
      return NULL;
  }
}

void* setup_device_field_decompress(const zfp_field* field, const int3& stride, long long int& offset)
{
  if (cuZFP::is_gpu_ptr(field->data)) {
    // field has already been allocated on device
    offset = 0;
    return field->data;
  }
  else {
    uint dims[3];
    dims[0] = field->nx;
    dims[1] = field->ny;
    dims[2] = field->nz;
    // GPU implementation currently requires contiguous field
    if (internal::is_contiguous(dims, stride, offset)) {
      // allocate device memory for decompressed field
      void *d_data = NULL;
      size_t size = zfp_field_size(field, NULL) * zfp_type_size(field->type);
      if (cudaMalloc(&d_data, size) != cudaSuccess)
        std::cerr << "failed to allocate device memory for field" << std::endl;
      return offset_void(field->type, d_data, -offset);
    }
    else
      return NULL;
  }
}

void cleanup_device_ptr(void* ptr, void* d_ptr, size_t bytes, long long int offset, zfp_type type)
{
  if (!cuZFP::is_gpu_ptr(ptr)) {
    // copy data from device to host and free device memory
    void *d_offset_ptr = offset_void(type, d_ptr, offset);
    void *h_offset_ptr = offset_void(type, ptr, offset);
    if (bytes > 0)
      cudaMemcpy(h_offset_ptr, d_offset_ptr, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_offset_ptr);
  }
}

} // namespace internal

size_t
cuda_compress(zfp_stream *stream, const zfp_field *field)
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
cuda_decompress(zfp_stream *stream, zfp_field *field)
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

  Word *d_stream = internal::setup_device_stream_decompress(stream);
  Word *d_index = NULL;

  // decode_parameter differs per execution policy
  // TODO: Decide if we want to pass maxbits, minexp and maxprec for all cases or not
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
    if (index_type != zfp_index_offset && index_type != zfp_index_hybrid) {
      std::cerr << "zfp index type not supported on GPU" << std::endl;
      return 0;
    }
    d_index = internal::setup_device_index(stream);
  }
  else {
    std::cerr << "zfp compression mode not supported on GPU" << std::endl;
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
  size_t bytes_written = zfp_field_size(field, NULL) * zfp_type_size(field->type);
  internal::cleanup_device_ptr(stream->stream, d_stream, 0, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, bytes_written, offset, field->type);
  if (d_index)
    cudaFree(d_index);

  // update bit stream to point just past consumed data
  if (bits_read)
    stream_rseek(stream->stream, pos + bits_read);

  return bits_read;
}
