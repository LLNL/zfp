#include <assert.h>
#include "cuZFP.h"

#include "encode1.cuh"
#include "encode2.cuh"
#include "encode3.cuh"

#include "ErrorCheck.h"

#include "decode1.cuh"
#include "decode2.cuh"
#include "decode3.cuh"

#include "constant_setup.cuh"
#include "pointers.cuh"
#include "type_info.cuh"
#include <iostream>

namespace cuZFP {
namespace internal {

//
// encode expects device pointers
//
template<typename T>
size_t encode(uint dims[3], int bits_per_block, T *d_data, Word *d_stream)
{

  int d = 0;
  size_t len = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      d++;
      len *= dims[i];
    }
  }

  ErrorCheck errors;
  size_t stream_size = 0;
  if(d == 1)
  {
    int dim = dims[0];
    ConstantSetup::setup_1d();
    stream_size = cuZFP::encode1<T>(dim, d_data, d_stream, bits_per_block); 
  }
  else if(d == 2)
  {
    uint2 ndims = make_uint2(dims[0], dims[1]);
    ConstantSetup::setup_2d();
    stream_size = cuZFP::encode2<T>(ndims, d_data, d_stream, bits_per_block); 
  }
  else if(d == 3)
  {
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    ConstantSetup::setup_3d();
    stream_size = cuZFP::encode<T>(ndims, d_data, d_stream, bits_per_block); 
  }

  errors.chk("Encode");
  return stream_size; 
}

template<typename T>
void decode(uint ndims[3], int bits_per_block, Word *stream, T *out)
{

  int d = 0;
  size_t out_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(ndims[i] != 0)
    {
      d++;
      out_size *= ndims[i];
    }
  }

  if(d == 3)
  {
    uint3 dims = make_uint3(ndims[0], ndims[1], ndims[2]);
    ConstantSetup::setup_3d();
    cuZFP::decode3<T>(dims, stream, out, bits_per_block); 
  }
  else if(d == 1)
  {
    uint dim = ndims[0];
    ConstantSetup::setup_1d();
    cuZFP::decode1<T>(dim, stream, out, bits_per_block); 

  }
  else if(d == 2)
  {
    uint2 dims;
    dims.x = ndims[0];
    dims.y = ndims[1];
    ConstantSetup::setup_2d();
    cuZFP::decode2<T>(dims, stream, out, bits_per_block); 

  }
  else std::cerr<<" d ==  "<<d<<" not implemented\n";
  
}

Word *setup_device_stream(zfp_stream *stream, zfp_field *field)
{
  bool stream_device = is_gpu_ptr(stream->stream);

  if(stream_device)
  {
    return stream->stream;
  }

  Word *d_stream = NULL;
  // TODO: we we have a real stream we can just ask it how big it is
  size_t max_size = zfp_stream_maximum_size(stream, field);
  cudaMalloc(&d_stream, max_size);
  cudaMemcpy(d_stream, stream->stream, max_size, cudaMemcpyHostToDevice);
  return d_stream;
}

void *setup_device_field(zfp_field *field)
{
  bool field_device = is_gpu_ptr(field->data);

  if(field_device)
  {
    return field->data;
  }
  
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }

  void *d_data = NULL;

  size_t field_bytes = type_size * field_size;
  cudaMalloc(&d_data, field_bytes);
  cudaMemcpy(d_data, field->data, field_bytes, cudaMemcpyHostToDevice);
  return d_data;
}

void cleanup_device_ptr(void *orig_ptr, void *d_ptr, size_t bytes)
{
  bool device = is_gpu_ptr(orig_ptr);
  if(device)
  {
    return;
  }
  // from whence it came
  if(bytes > 0)
  {
    cudaMemcpy(orig_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost);
  }
  cudaFree(d_ptr);
}

} // namespace internal

size_t
compress(zfp_stream *stream, zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;
  size_t stream_bytes = 0;
  
  void *d_data = internal::setup_device_field(field);
  Word *d_stream = internal::setup_device_stream(stream, field);

  if(field->type == zfp_type_float)
  {
    float* data = (float*) d_data;
    stream_bytes = internal::encode<float>(dims, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_double)
  {
    double* data = (double*) d_data;
    stream_bytes = internal::encode<double>(dims, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_int32)
  {
    int * data = (int*) d_data;
    stream_bytes = internal::encode<int>(dims, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_int64)
  {
    long long int * data = (long long int*) d_data;
    stream_bytes = internal::encode<long long int>(dims, (int)stream->maxbits, data, d_stream);
  }

  internal::cleanup_device_ptr(stream->stream, d_stream, stream_bytes);
  internal::cleanup_device_ptr(field->data, d_data, 0);
  return stream_bytes;
}
  
void 
decompress(zfp_stream *stream, zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;
    

  void *d_data = internal::setup_device_field(field);
  Word *d_stream = internal::setup_device_stream(stream, field);

  if(field->type == zfp_type_float)
  {
    float *data = (float*) d_data;
    internal::decode(dims, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_double)
  {
    double *data = (double*) d_data;
    internal::decode(dims, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int32)
  {
    int *data = (int*) d_data;
    internal::decode(dims, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int64)
  {
    long long int *data = (long long int*) d_data;
    internal::decode(dims, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else
  {
    std::cerr<<"Cannot decompress: type unknown\n";
  }

   
  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }
  
  size_t bytes = type_size * field_size;
  internal::cleanup_device_ptr(stream->stream, d_stream,0);
  internal::cleanup_device_ptr(field->data, d_data, bytes);
  
}

} // namespace cuZFP

