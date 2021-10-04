
#include <hip/hip_runtime.h>
#include <assert.h>

#include <hipcub/hipcub.hpp>

#include "hipZFP.h"

#include "encode1.h"
#include "encode2.h"
#include "encode3.h"

#include "decode1.h"
#include "decode2.h"
#include "decode3.h"

#include "variable.h"

#include "ErrorCheck.h"

#include "pointers.h"
#include "type_info.h"
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

bool is_contigous3d(const uint dims[3], const int3 &stride, long long int &offset)
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

bool is_contigous2d(const uint dims[3], const int3 &stride, long long int &offset)
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

bool is_contigous1d(uint dim, const int &stride, long long int &offset)
{
  offset = 0;
  if(stride < 0) offset = stride * (int(dim) - 1);
  return std::abs(stride) == 1;
}

bool is_contigous(const uint dims[3], const int3 &stride, long long int &offset)
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
    return is_contigous1d(dims[0], stride.x, offset);
  } 

}
//
// encode expects device pointers
//
template <typename T, bool variable_rate>
size_t encode(uint dims[3], int3 stride, int minbits, int maxbits,
              int maxprec, int minexp, T *d_data, Word *d_stream, ushort *d_bitlengths)
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
    int sx = stride.x;
    stream_size = hipZFP::encode1<T, variable_rate>(dim, sx, d_data, d_stream, d_bitlengths,
                                                    minbits, maxbits, maxprec, minexp);
  }
  else if(d == 2)
  {
    uint2 ndims = make_uint2(dims[0], dims[1]);
    int2 s;
    s.x = stride.x;
    s.y = stride.y;
    stream_size = hipZFP::encode2<T, variable_rate>(ndims, s, d_data, d_stream, d_bitlengths,
                                                    minbits, maxbits, maxprec, minexp);
  }
  else if(d == 3)
  {
    int3 s;
    s.x = stride.x;
    s.y = stride.y;
    s.z = stride.z;
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    stream_size = hipZFP::encode<T, variable_rate>(ndims, s, d_data, d_stream, d_bitlengths,
                                                  minbits, maxbits, maxprec, minexp);
  }

  errors.chk("Encode");

  return stream_size;
}

template<typename T>
size_t decode(uint ndims[3], int3 stride, int bits_per_block, Word *stream, T *out)
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
    uint3 dims = make_uint3(ndims[0], ndims[1], ndims[2]);

    int3 s;
    s.x = stride.x;
    s.y = stride.y;
    s.z = stride.z;

    stream_bytes = hipZFP::decode3<T>(dims, s, stream, out, bits_per_block);
  }
  else if(d == 1)
  {
    uint dim = ndims[0];
    int sx = stride.x;

    stream_bytes = hipZFP::decode1<T>(dim, sx, stream, out, bits_per_block);


  }
  else if(d == 2)
  {
    uint2 dims;
    dims.x = ndims[0];
    dims.y = ndims[1];

    int2 s;
    s.x = stride.x;
    s.y = stride.y;

    stream_bytes = hipZFP::decode2<T>(dims, s, stream, out, bits_per_block);
  }
  else std::cerr<<" d ==  "<<d<<" not implemented\n";
  return stream_bytes;
}

Word *setup_device_stream_compress(zfp_stream *stream,const zfp_field *field)
{
  bool stream_device = hipZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(word) == sizeof(Word)); // "HIP version currently only supports 64bit words");
  
  if(stream_device)
  {
    return (Word*) stream->stream->begin;
  }

  Word *d_stream = NULL;
  size_t max_size = zfp_stream_maximum_size(stream, field);
//#if (HIPRT_VERSION >= 11020)
//  hipMallocAsync(&d_stream, max_size, 0);
//#else
  hipMalloc(&d_stream, max_size);
  hipMemset(d_stream, 0, max_size);
//#endif
  return d_stream;
}

Word *setup_device_stream_decompress(zfp_stream *stream,const zfp_field *field)
{
  bool stream_device = hipZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(word) == sizeof(Word)); // "HIP version currently only supports 64bit words");

  if(stream_device)
  {
    return (Word*) stream->stream->begin;
  }

  Word *d_stream = NULL;
  //TODO: change maximum_size to compressed stream size
  size_t size = zfp_stream_maximum_size(stream, field);
//#if (HIPRT_VERSION >= 11020)
//  hipMallocAsync(&d_stream, size, 0);
//#else
  hipMalloc(&d_stream, size);
  hipMemset(d_stream, 0, size);
//#endif


  hipMemcpy(d_stream, stream->stream->begin, size, hipMemcpyHostToDevice);
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

void *setup_device_field_compress(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = hipZFP::is_gpu_ptr(field->data);

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
//#if (HIPRT_VERSION >= 11020)
//    hipMallocAsync(&d_data, field_bytes, 0);
//#else
    hipMalloc(&d_data, field_bytes);
    hipMemset(d_data, 0, field_bytes);
//#endif


    hipMemcpy(d_data, host_ptr, field_bytes, hipMemcpyHostToDevice);
  }
  return offset_void(field->type, d_data, -offset);
}

void *setup_device_field_decompress(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = hipZFP::is_gpu_ptr(field->data);

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
//#if (HIPRT_VERSION >= 11020)
//    hipMallocAsync(&d_data, field_bytes, 0);
//#else
    hipMalloc(&d_data, field_bytes);
    hipMemset(d_data, 0, field_bytes);
//#endif
  }
  return offset_void(field->type, d_data, -offset);
}

ushort *setup_device_nbits_compress(zfp_stream *stream, const zfp_field *field, int variable_rate)
{
  if (!variable_rate)
    return NULL;

  bool device_mem = hipZFP::is_gpu_ptr(stream->stream->bitlengths);
  if (device_mem)
    return (ushort *)stream->stream->bitlengths;

  ushort *d_bitlengths = NULL;
  size_t size = zfp_field_num_blocks(field) * sizeof(ushort);
//#if (HIPRT_VERSION >= 11020)
//  hipMallocAsync(&d_bitlengths, size, 0);
//#else
  hipMalloc(&d_bitlengths, size);
	hipMemset(d_bitlengths, 0, size);
//#endif
  return d_bitlengths;
}

ushort *setup_device_nbits_decompress(zfp_stream *stream, const zfp_field *field, int variable_rate)
{
  if (!variable_rate)
    return NULL;

  if (hipZFP::is_gpu_ptr(stream->stream->bitlengths))
    return stream->stream->bitlengths;

  ushort *d_bitlengths = NULL;
  size_t size = zfp_field_num_blocks(field) * sizeof(ushort);
//#if (HIPRT_VERSION >= 11020)
//  hipMallocAsync(&d_bitlengths, size, 0);
//#else
  hipMalloc(&d_bitlengths, size);
  hipMemset(d_bitlengths, 0, size);
//#endif
  hipMemcpy(d_bitlengths, stream->stream->bitlengths, size, hipMemcpyHostToDevice);
  return d_bitlengths;
}

void cleanup_device_nbits(zfp_stream *stream, const zfp_field *field,
                          ushort *d_bitlengths, int variable_rate, int copy)
{
  if (!variable_rate)
    return;

  if (hipZFP::is_gpu_ptr(stream->stream->bitlengths))
    return;

  size_t size = zfp_field_num_blocks(field) * sizeof(ushort);
  if (copy)
    hipMemcpy(stream->stream->bitlengths, d_bitlengths, size, hipMemcpyDeviceToHost);

//#if (HIPRT_VERSION >= 11020)
//  hipFreeAsync(d_bitlengths, 0);
//#else
  hipFree(d_bitlengths);
//#endif
}

void setup_device_chunking(int *chunk_size, unsigned long long **d_offsets, size_t *lcubtemp,
                            void **d_cubtemp, int num_sm, int variable_rate)
{
  if (!variable_rate)
    return;

  // TODO : Error handling for HIP malloc and HIPB?
  // Assuming 1 thread = 1 ZFP block,
  // launching 1024 threads per SM should give a decent ochippancy
  *chunk_size = num_sm * 1024;
  size_t size = (*chunk_size + 1) * sizeof(unsigned long long);
//#if (HIPRT_VERSION >= 11020)
//  hipMallocAsync(d_offsets, size, 0);
//#else
  hipMalloc(d_offsets, size);
  hipMemset(*d_offsets, 0, size);
//#endif
  // Using HIPB for the prefix sum. HIPB needs a bit of temp memory too
  size_t tempsize;
  hipcub::DeviceScan::InclusiveSum(nullptr, tempsize, *d_offsets, *d_offsets, *chunk_size + 1);
  *lcubtemp = tempsize;
//#if (HIPRT_VERSION >= 11020)
//  hipMallocAsync(d_cubtemp, tempsize, 0);
//#else
  hipMalloc(d_cubtemp, tempsize);
  hipMemset(*d_cubtemp, 0, tempsize);
//#endif
}

void cleanup_device_ptr(void *orig_ptr, void *d_ptr, size_t bytes, long long int offset, zfp_type type)
{
  bool device = hipZFP::is_gpu_ptr(orig_ptr);
  if(device)
  {
    return;
  }
  // from whence it came
  void *d_offset_ptr = offset_void(type, d_ptr, offset);
  void *h_offset_ptr = offset_void(type, orig_ptr, offset);

  if(bytes > 0)
  {
    hipMemcpy(h_offset_ptr, d_offset_ptr, bytes, hipMemcpyDeviceToHost);
  }

//#if (HIPRT_VERSION >= 11020)
//  hipFreeAsync(d_offset_ptr, 0);
//#else
  hipFree(d_offset_ptr);
//#endif

}

} // namespace internal

size_t
hip_compress(zfp_stream *stream, const zfp_field *field, int variable_rate)
{

/*
#if (HIPRT_VERSION < 9000)
  if (variable_rate)
    return 0; // Variable rate requires HIP >= 9
#endif
*/
  if (zfp_stream_compression_mode(stream) == zfp_mode_reversible)
  {
    // Reversible mode not supported on GPU
    return 0;
  }

  stream_memset(stream->stream);

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

  if(d_data == NULL)
  {
    // null means the array is non-contiguous host mem which is not supported
    return 0;
  }

  int num_sm;
  hipDeviceGetAttribute(&num_sm, hipDeviceAttributeMultiprocessorCount, 0);

  Word *d_stream = internal::setup_device_stream_compress(stream, field);

  ushort *d_bitlengths = internal::setup_device_nbits_compress(stream, field, variable_rate);

  int chunk_size;
  unsigned long long *d_offsets;
  size_t lcubtemp;
  void *d_cubtemp;
  internal::setup_device_chunking(&chunk_size, &d_offsets, &lcubtemp, &d_cubtemp, num_sm, variable_rate);

  uint buffer_maxbits = MIN (stream->maxbits, zfp_block_maxbits(stream, field));

  if(field->type == zfp_type_float)
  {
    float* data = (float*) d_data;
    if (variable_rate)
      stream_bytes = internal::encode<float, true>(dims, stride, stream->minbits, (int)buffer_maxbits,
                                                   stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
    else
      stream_bytes = internal::encode<float, false>(dims, stride, stream->minbits, (int)buffer_maxbits,
                                                    stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
  }
  else if(field->type == zfp_type_double)
  {
    double* data = (double*) d_data;
    if (variable_rate)
      stream_bytes = internal::encode<double, true>(dims, stride, stream->minbits, (int)buffer_maxbits,
                                                    stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
    else
      stream_bytes = internal::encode<double, false>(dims, stride, stream->minbits, (int)buffer_maxbits,
                                                     stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
  }
  else if(field->type == zfp_type_int32)
  {
    int * data = (int*) d_data;
    if (variable_rate)
      stream_bytes = internal::encode<int, true>(dims, stride, stream->minbits, (int)buffer_maxbits,
                                                 stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
    else
      stream_bytes = internal::encode<int, false>(dims, stride, stream->minbits, (int)buffer_maxbits,
                                                  stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
  }
  else if(field->type == zfp_type_int64)
  {
    long long int * data = (long long int*) d_data;
    if (variable_rate)
      stream_bytes = internal::encode<long long int, true>(dims, stride, stream->minbits, (int)buffer_maxbits,
                                                           stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
    else
      stream_bytes = internal::encode<long long int, false>(dims, stride, stream->minbits, (int)buffer_maxbits,
                                                            stream->maxprec, stream->minexp, data, d_stream, d_bitlengths);
  }

  if (variable_rate)
  {
    size_t blocks = zfp_field_num_blocks(field);
    for (size_t i = 0; i < blocks; i += chunk_size)
    {
      int hipr_blocks = chunk_size;
      bool last_chunk = false;
      if (i + chunk_size > blocks)
      {
        hipr_blocks = (int)(blocks - i);
        last_chunk = true;
      }
      // Copy the 16-bit lengths in the offset array
      hipZFP::copy_length_launch(d_bitlengths, d_offsets, i, hipr_blocks);

      // Prefix sum to turn length into offsets
      hipcub::DeviceScan::InclusiveSum(d_cubtemp, lcubtemp, d_offsets, d_offsets, hipr_blocks + 1);

      // Compact the stream array in-place
      hipZFP::chunk_process_launch((uint*)d_stream, d_offsets, i, hipr_blocks, last_chunk, buffer_maxbits, num_sm);
    }
    // The total length in bits is now in the base of the prefix sum.
    hipMemcpy (&stream_bytes, d_offsets, sizeof (unsigned long long), hipMemcpyDeviceToHost);
    stream_bytes = (stream_bytes + 7) / 8;
  }

  internal::cleanup_device_ptr(stream->stream->begin, d_stream, stream_bytes, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, 0, offset, field->type);
  if (variable_rate)
  {
    if (stream->stream->bitlengths) // Saving the individual block lengths if a pointer exists
    {
      size_t size = zfp_field_num_blocks(field) * sizeof(ushort);
      internal::cleanup_device_ptr(stream->stream->bitlengths, d_bitlengths, size, 0, zfp_type_none);
    }
    internal::cleanup_device_ptr(NULL, d_offsets, 0, 0, zfp_type_none);
    internal::cleanup_device_ptr(NULL, d_cubtemp, 0, 0, zfp_type_none);
  }

  // zfp wants to flush the stream.
  // set bits to wsize because we already did that.
  size_t compressed_size = (stream_bytes + sizeof(Word) - 1) / sizeof(Word);
  stream->stream->bits = wsize;
  // set stream pointer to end of stream
  stream->stream->ptr = stream->stream->begin + compressed_size;

  return stream_bytes;
}

void
hip_decompress(zfp_stream *stream, zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int3 stride;

  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;

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
