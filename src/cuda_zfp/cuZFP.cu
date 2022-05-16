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
  int d = 0;
  size_t len = 1;
  for (int i = 0; i < 3; ++i) {
    if (dims[i] != 0) {
      d++;
      len *= dims[i];
    }
  }

  ErrorCheck errors;
  size_t stream_size = 0;
  if (d == 1) {
    int dim = dims[0];
    int s = stride.x;
    stream_size = cuZFP::encode1<T>(dim, s, d_data, d_stream, bits_per_block); 
  }
  else if (d == 2) {
    uint2 ndims = make_uint2(dims[0], dims[1]);
    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 
    stream_size = cuZFP::encode2<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }
  else if (d == 3) {
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 
    stream_size = cuZFP::encode3<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }

  errors.chk("Encode");

  return stream_size; 
}

template <typename T>
size_t decode(uint ndims[3], int3 stride, Word *stream, Word *index, T *out, int decode_parameter, uint granularity, zfp_mode mode, zfp_index_type index_type)
{
  size_t out_size = 1;
  size_t stream_bytes = 0;

  int d = 0;
  for (int i = 0; i < 3; ++i)
    if (ndims[i] != 0) {
      d++;
      out_size *= ndims[i];
    }

  if (d == 3) {
    uint3 dims = make_uint3(ndims[0], ndims[1], ndims[2]);
    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 
    stream_bytes = cuZFP::decode3<T>(dims, s, stream, index, out, decode_parameter, granularity, mode, index_type);
  }
  else if (d == 2) {
    uint2 dims;
    dims.x = ndims[0];
    dims.y = ndims[1];
    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 
    stream_bytes = cuZFP::decode2<T>(dims, s, stream, index, out, decode_parameter, granularity, mode, index_type);
  }
  else if (d == 1) {
    uint dim = ndims[0];
    int sx = stride.x;
    stream_bytes = cuZFP::decode1<T>(dim, sx, stream, index, out, decode_parameter, granularity, mode, index_type);
  }
  else
    std::cerr << "d == " << d << " not implemented\n";
 
  return stream_bytes;
}

Word *setup_device_stream_compress(zfp_stream *stream, const zfp_field *field)
{
  bool stream_device = cuZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(word) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

  if (stream_device) {
    return (Word*) stream->stream->begin;
  }

  Word *d_stream = NULL;
  size_t max_size = zfp_stream_maximum_size(stream, field);
  if (cudaMalloc(&d_stream, max_size) != cudaSuccess) {
    std::cerr<<"failed to allocate device memory for stream\n";
  }
  return d_stream;
}

Word *setup_device_stream_decompress(zfp_stream *stream, const zfp_field *field)
{
  bool stream_device = cuZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(word) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

  if (stream_device) {
    return (Word*) stream->stream->begin;
  }

  Word *d_stream = NULL;
  //TODO: change maximum_size to compressed stream size
  size_t size = zfp_stream_maximum_size(stream, field);
  if (cudaMalloc(&d_stream, size) != cudaSuccess) {
    std::cerr<<"failed to allocate device memory for stream\n";
  }
  if (cudaMemcpy(d_stream, stream->stream->begin, size, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr<<"failed to copy stream from host to device\n";
  }
  return d_stream;
}

Word *setup_device_index(zfp_stream *stream, const size_t size)
{
  bool stream_device = cuZFP::is_gpu_ptr(stream->index->data);
  assert(sizeof(uint64) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

  if (stream_device) {
    return (Word*) stream->index->data;
  }

  Word *d_index = NULL;
  if (cudaMalloc(&d_index, size) != cudaSuccess) {
    std::cerr<<"failed to allocate device memory for index\n";
  }
  if (cudaMemcpy(d_index, stream->index->data, size, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr<<"failed to copy stream from host to device\n";
  }
  return d_index;
}

void * offset_void(zfp_type type, void *ptr, long long int offset)
{
  void * offset_ptr = NULL;
  if (type == zfp_type_float) {
    float* data = (float*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if (type == zfp_type_double) {
    double* data = (double*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if (type == zfp_type_int32) {
    int * data = (int*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if (type == zfp_type_int64) {
    long long int * data = (long long int*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  return offset_ptr;
}

void *setup_device_field_compress(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = cuZFP::is_gpu_ptr(field->data);

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
  for (int i = 0; i < 3; ++i) {
    if (dims[i] != 0) {
      field_size *= dims[i];
    }
  }

  bool contig = internal::is_contiguous(dims, stride, offset);

  void * host_ptr = offset_void(field->type, field->data, offset);;

  void *d_data = NULL;
  if (contig) {
    size_t field_bytes = type_size * field_size;
    if (cudaMalloc(&d_data, field_bytes) != cudaSuccess)
      std::cerr<<"failed to allocate device memory for field\n";
    if (cudaMemcpy(d_data, host_ptr, field_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
      std::cerr<<"failed to copy field from host to device\n";
  }
  return offset_void(field->type, d_data, -offset);
}

void *setup_device_field_decompress(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = cuZFP::is_gpu_ptr(field->data);

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
  for (int i = 0; i < 3; ++i)
    if (dims[i] != 0)
      field_size *= dims[i];

  bool contig = internal::is_contiguous(dims, stride, offset);

  void * host_ptr = offset_void(field->type, field->data, offset);

  void *d_data = NULL;
  if (contig) {
    size_t field_bytes = type_size * field_size;
    if (cudaMalloc(&d_data, field_bytes) != cudaSuccess)
      std::cerr<<"failed to allocate device memory for field\n";
  }

  return offset_void(field->type, d_data, -offset);
}


void cleanup_device_ptr(void *orig_ptr, void *d_ptr, size_t bytes, long long int offset, zfp_type type)
{
  bool device = cuZFP::is_gpu_ptr(orig_ptr);
  if (device) {
    return;
  }
  // from whence it came
  void *d_offset_ptr = offset_void(type, d_ptr, offset);
  void *h_offset_ptr = offset_void(type, orig_ptr, offset);

  if (bytes > 0) {
    cudaMemcpy(h_offset_ptr, d_offset_ptr, bytes, cudaMemcpyDeviceToHost);
  }

  cudaFree(d_offset_ptr);
}

} // namespace internal

size_t
cuda_compress(zfp_stream *stream, const zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int3 stride;
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;
  
  size_t stream_bytes = 0;
  long long int offset = 0; 
  void *d_data = internal::setup_device_field_compress(field, stride, offset);

  if (d_data == NULL) {
    // null means the array is non-contiguous host mem which is not supported
    return 0;
  }

  Word *d_stream = internal::setup_device_stream_compress(stream, field);

  if (field->type == zfp_type_float) {
    float* data = (float*) d_data;
    stream_bytes = internal::encode<float>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if (field->type == zfp_type_double) {
    double* data = (double*) d_data;
    stream_bytes = internal::encode<double>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if (field->type == zfp_type_int32) {
    int * data = (int*) d_data;
    stream_bytes = internal::encode<int>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if (field->type == zfp_type_int64) {
    long long int * data = (long long int*) d_data;
    stream_bytes = internal::encode<long long int>(dims, stride, (int)stream->maxbits, data, d_stream);
  }

  internal::cleanup_device_ptr(stream->stream->begin, d_stream, stream_bytes, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, 0, offset, field->type);

  // zfp wants to flush the stream.
  // set bits to wsize because we already did that.
  size_t compressed_size = stream_bytes / sizeof(Word);
  stream->stream->bits = wsize;
  // set stream pointer to end of stream
  stream->stream->ptr = stream->stream->begin + compressed_size;

  return stream_bytes;
}

void
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
    return;
  }

  Word *d_stream = internal::setup_device_stream_decompress(stream, field);
  Word *d_index = NULL;

  // keep track of bit stream offset
  size_t pos = stream_rtell(stream->stream);
  size_t bits_read = 0;

  // determine number of blocks to decompress
  uint blocks = 1;
  for (int i = 0; i < 3; i++)
    if (dims[i])
      blocks *= (dims[i] + 3) / 4;

  // decode_parameter differs per execution policy
  // TODO: Decide if we want to pass maxbits, minexp and maxprec for all cases or not
  // TODO: decide if casting the uints maxbits & maxprec to int is acceptable or not
  size_t index_size;
  uint granularity;
  int decode_parameter;
  zfp_index_type index_type = zfp_index_none;
  zfp_mode mode = zfp_stream_compression_mode(stream);

  if (mode == zfp_mode_fixed_rate) {
    decode_parameter = (int)stream->maxbits;
    granularity = 1;
    bits_read = (size_t)blocks * stream->maxbits;
  }
  else if (mode == zfp_mode_fixed_precision || mode == zfp_mode_fixed_accuracy) {
    decode_parameter = (mode == zfp_mode_fixed_precision ? (int)stream->maxprec : (int)stream->minexp);
    granularity = stream->index->granularity;
    // TODO: fetch bits_read from index
    bits_read = 0;
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
      return;
    }
    d_index = internal::setup_device_index(stream, index_size);
  }
  else {
    std::cerr << "zfp expert mode not supported on GPU" << std::endl;
    return;
  }

  // decode compressed data
  size_t success = 0;
  if (field->type == zfp_type_float) {
    float *data = (float*)d_data;
    success = internal::decode(dims, stride, d_stream, d_index, data, decode_parameter, granularity, mode, index_type);
    d_data = (void*)data;
  }
  else if (field->type == zfp_type_double) {
    double *data = (double*)d_data;
    success = internal::decode(dims, stride, d_stream, d_index, data, decode_parameter, granularity, mode, index_type);
    d_data = (void*)data;
  }
  else if (field->type == zfp_type_int32) {
    int *data = (int*)d_data;
    success = internal::decode(dims, stride, d_stream, d_index, data, decode_parameter, granularity, mode, index_type);
    d_data = (void*)data;
  }
  else if (field->type == zfp_type_int64) {
    long long int *data = (long long int*)d_data;
    success = internal::decode(dims, stride, d_stream, d_index, data, decode_parameter, granularity, mode, index_type);
    d_data = (void*)data;
  }
  else {
    // TODO: suppress error messages
    std::cerr << "zfp cuda_decompress: unknown scalar type" << std::endl;
  }

  // clean up
  size_t bytes_written = zfp_type_size(field->type);
  for (int i = 0; i < 3; ++i)
    if (dims[i] != 0)
      bytes_written *= dims[i];
  internal::cleanup_device_ptr(stream->stream, d_stream, 0, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, bytes_written, offset, field->type);
  if (d_index)
    cudaFree(d_index);

  // update bit stream to point just past consumed data
  if (success)
    stream_rseek(stream->stream, pos + bits_read);
}
