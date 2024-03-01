#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>

#include "syclZFP.h"

#include "encode1.dp.hpp"
#include "encode2.dp.hpp"
#include "encode3.dp.hpp"

#include "decode1.dp.hpp"
#include "decode2.dp.hpp"
#include "decode3.dp.hpp"

#include "ErrorCheck.h"

#include "pointers.dp.hpp"
#include "type_info.dp.hpp"
#include <iostream>
#include <assert.h>

// we need to know about bitstream, but we don't 
// want duplicate symbols.
#ifndef inline_
  #define inline_ inline
#endif

#include "../inline/bitstream.c"
namespace internal 
{

bool is_contigous3d(const uint dims[3], const sycl::int3 &stride,
                    long long int &offset)
{
  typedef long long int int64;
  int64 idims[3];
  idims[0] = dims[0];
  idims[1] = dims[1];
  idims[2] = dims[2];

  int64 imin = std::min(stride.x(), 0) * (idims[0] - 1) +
               std::min(stride.y(), 0) * (idims[1] - 1) +
               std::min(stride.z(), 0) * (idims[2] - 1);

  int64 imax = std::max(stride.x(), 0) * (idims[0] - 1) +
               std::max(stride.y(), 0) * (idims[1] - 1) +
               std::max(stride.z(), 0) * (idims[2] - 1);
  offset = imin;
  int64 ns = idims[0] * idims[1] * idims[2];

  return (imax - imin + 1 == ns);
}

bool is_contigous2d(const uint dims[3], const sycl::int3 &stride,
                    long long int &offset)
{
  typedef long long int int64;
  int64 idims[2];
  idims[0] = dims[0];
  idims[1] = dims[1];

  int64 imin = std::min(stride.x(), 0) * (idims[0] - 1) +
               std::min(stride.y(), 0) * (idims[1] - 1);

  int64 imax = std::max(stride.x(), 0) * (idims[0] - 1) +
               std::max(stride.y(), 0) * (idims[1] - 1);

  offset = imin;
  return (imax - imin + 1) == (idims[0] * idims[1]);
}

bool is_contigous1d(uint dim, const int &stride, long long int &offset)
{
  offset = 0;
  if(stride < 0) offset = stride * (int(dim) - 1);
  return std::abs(stride) == 1;
}

bool is_contigous(const uint dims[3], const sycl::int3 &stride,
                  long long int &offset)
{
  int d = 0;
  
  if(dims[0] != 0) d++;
  if(dims[1] != 0) d++;
  if(dims[2] != 0) d++;

  if(d == 3)
  {
    return is_contigous3d(dims, stride, offset);
  }
  else if(d == 2)
  {
   return is_contigous2d(dims, stride, offset);
  }
  else
  {
    return is_contigous1d(dims[0], stride.x(), offset);
  } 

}
//
// encode expects device pointers
//
template <typename T>
size_t encode(uint dims[3], sycl::int3 stride, int bits_per_block, T *d_data,
              Word *d_stream)
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
    int sx = stride.x();
    stream_size = syclZFP::encode1<T>(dim, sx, d_data, d_stream, bits_per_block); 
  }
  else if(d == 2)
  {
    sycl::uint2 ndims = sycl::uint2(dims[0], dims[1]);
    sycl::int2 s;
    s.x() = stride.x();
    s.y() = stride.y();
    stream_size = syclZFP::encode2<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }
  else if(d == 3)
  {
    sycl::int3 s;
    s.x() = stride.x();
    s.y() = stride.y();
    s.z() = stride.z();
    sycl::uint3 ndims = sycl::uint3(dims[0], dims[1], dims[2]);
    stream_size = syclZFP::encode<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }

  errors.chk("Encode");
  
  return stream_size; 
}

template <typename T>
size_t decode(uint ndims[3], sycl::int3 stride, int bits_per_block,
              Word *stream, T *out)
{

  int d = 0;
  size_t out_size = 1;
  size_t stream_bytes = 0;
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
    sycl::uint3 dims = sycl::uint3(ndims[0], ndims[1], ndims[2]);

    sycl::int3 s;
    s.x() = stride.x();
    s.y() = stride.y();
    s.z() = stride.z();

    stream_bytes = syclZFP::decode3<T>(dims, s, stream, out, bits_per_block); 
  }
  else if(d == 1)
  {
    uint dim = ndims[0];
    int sx = stride.x();

    stream_bytes = syclZFP::decode1<T>(dim, sx, stream, out, bits_per_block); 

  }
  else if(d == 2)
  {
    sycl::uint2 dims;
    dims.x() = ndims[0];
    dims.y() = ndims[1];

    sycl::int2 s;
    s.x() = stride.x();
    s.y() = stride.y();

    stream_bytes = syclZFP::decode2<T>(dims, s, stream, out, bits_per_block); 
  }
  else std::cerr<<" d ==  "<<d<<" not implemented\n";
 
  return stream_bytes;
}

Word *setup_device_stream_compress(zfp_stream *stream,const zfp_field *field)
{
  bool stream_device = syclZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(word) == sizeof(Word)); // "SYCL version currently only supports 64bit words");

  if(stream_device)
  {
    return (Word*) stream->stream->begin;
  }

  Word *d_stream = NULL;
  size_t max_size = zfp_stream_maximum_size(stream, field);
  d_stream = (Word *)sycl::malloc_device(max_size, dpct::get_default_queue());
  return d_stream;
}

Word *setup_device_stream_decompress(zfp_stream *stream,const zfp_field *field)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  bool stream_device = syclZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(word) == sizeof(Word)); // "SYCL version currently only supports 64bit words");

  if(stream_device)
  {
    return (Word*) stream->stream->begin;
  }

  Word *d_stream = NULL;
  //TODO: change maximum_size to compressed stream size
  size_t size = zfp_stream_maximum_size(stream, field);
  d_stream = (Word *)sycl::malloc_device(size, q_ct1);
  q_ct1.memcpy(d_stream, stream->stream->begin, size).wait();
  return d_stream;
}

void * offset_void(zfp_type type, void *ptr, long long int offset)
{
  void * offset_ptr = NULL;
  if(type == zfp_type_float)
  {
    float* data = (float*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_double)
  {
    double* data = (double*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_int32)
  {
    int * data = (int*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_int64)
  {
    long long int * data = (long long int*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  return offset_ptr;
}

void *setup_device_field_compress(const zfp_field *field,
                                  const sycl::int3 &stride,
                                  long long int &offset)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  bool field_device = syclZFP::is_gpu_ptr(field->data);

  if(field_device)
  {
    offset = 0;
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

  bool contig = internal::is_contigous(dims, stride, offset);
  
  void * host_ptr = offset_void(field->type, field->data, offset);;

  void *d_data = NULL;
  if(contig)
  {
    size_t field_bytes = type_size * field_size;
    d_data = (void *)sycl::malloc_device(field_bytes, q_ct1);
    //d_data = (void *)sycl::malloc_shared(field_bytes, q_ct1);

    q_ct1.memcpy(d_data, host_ptr, field_bytes).wait();
  }
  return offset_void(field->type, d_data, -offset);
}

void *setup_device_field_decompress(const zfp_field *field,
                                    const sycl::int3 &stride,
                                    long long int &offset)
{
  bool field_device = syclZFP::is_gpu_ptr(field->data);

  if(field_device)
  {
    offset = 0;
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

  bool contig = internal::is_contigous(dims, stride, offset);

  void *d_data = NULL;
  if(contig)
  {
    size_t field_bytes = type_size * field_size;
    d_data = (void *)sycl::malloc_device(field_bytes, dpct::get_default_queue());
  }
  return offset_void(field->type, d_data, -offset);
}

void cleanup_device_ptr(void *orig_ptr, void *d_ptr, size_t bytes, long long int offset, zfp_type type)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  bool device = syclZFP::is_gpu_ptr(orig_ptr);
  if(device)
  {
    return;
  }
  // from whence it came
  void *d_offset_ptr = offset_void(type, d_ptr, offset);
  void *h_offset_ptr = offset_void(type, orig_ptr, offset);

  if(bytes > 0)
  {
    q_ct1.memcpy(h_offset_ptr, d_offset_ptr, bytes).wait();
  }

  sycl::free(d_offset_ptr, q_ct1);
}

} // namespace internal

size_t
sycl_compress(zfp_stream *stream, const zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  sycl::int3 stride;
  stride.x() = field->sx ? field->sx : 1;
  stride.y() = field->sy ? field->sy : field->nx;
  stride.z() = field->sz ? field->sz : field->nx * field->ny;

  size_t stream_bytes = 0;
  long long int offset = 0; 
  void *d_data = internal::setup_device_field_compress(field, stride, offset);

  if(d_data == NULL)
  {
    // null means the array is non-contiguous host mem which is not supported
    return 0;
  }

  Word *d_stream = internal::setup_device_stream_compress(stream, field);

  if(field->type == zfp_type_float)
  {
    float* data = (float*) d_data;
    stream_bytes = internal::encode<float>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_double)
  {
    double* data = (double*) d_data;
    stream_bytes = internal::encode<double>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_int32)
  {
    int * data = (int*) d_data;
    stream_bytes = internal::encode<int>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_int64)
  {
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
sycl_decompress(zfp_stream *stream, zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  sycl::int3 stride;
  stride.x() = field->sx ? field->sx : 1;
  stride.y() = field->sy ? field->sy : field->nx;
  stride.z() = field->sz ? field->sz : field->nx * field->ny;

  size_t decoded_bytes = 0;
  long long int offset = 0;
  void *d_data = internal::setup_device_field_decompress(field, stride, offset);
  
  if(d_data == NULL)
  {
    // null means the array is non-contiguous host mem which is not supported
    return;
  }

  Word *d_stream = internal::setup_device_stream_decompress(stream, field);

  if(field->type == zfp_type_float)
  {
    float *data = (float*) d_data;
    decoded_bytes = internal::decode(dims, stride, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_double)
  {
    double *data = (double*) d_data;
    decoded_bytes = internal::decode(dims, stride, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int32)
  {
    int *data = (int*) d_data;
    decoded_bytes = internal::decode(dims, stride, (int)stream->maxbits, d_stream, data);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int64)
  {
    long long int *data = (long long int*) d_data;
    decoded_bytes = internal::decode(dims, stride, (int)stream->maxbits, d_stream, data);
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
  internal::cleanup_device_ptr(stream->stream->begin, d_stream, 0, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, bytes, offset, field->type);
  
  // this is how zfp determins if this was a success
  size_t words_read = decoded_bytes / sizeof(Word);
  stream->stream->bits = wsize;
  // set stream pointer to end of stream
  stream->stream->ptr = stream->stream->begin + words_read;
}
