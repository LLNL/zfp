#include <iostream>
#include <cub/cub.cuh>
#include "cuZFP.h"
#include "error.cuh"
#include "pointers.cuh"
#include "traits.cuh"
#include "encode.cuh"
#include "encode1.cuh"
#include "encode2.cuh"
#include "encode3.cuh"
#include "variable.cuh"
#include "decode.cuh"
#include "decode1.cuh"
#include "decode2.cuh"
#include "decode3.cuh"

// we need to know about bitstream, but we don't want duplicate symbols
#ifndef inline_
  #define inline_ inline
#endif

#include "zfp/bitstream.inl"

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

// allocate device memory
template <typename T>
bool device_malloc(T** d_pointer, size_t size, const char* what = 0)
{
  bool success;

#if CUDART_VERSION >= 11020
  success = (cudaMallocAsync(d_pointer, size, 0) == cudaSuccess);
#else
  success = (cudaMalloc(d_pointer, size) == cudaSuccess);
#endif

  if (!success) {
    std::cerr << "failed to allocate device memory";
    if (what)
      std::cerr << " for " << what;
    std::cerr << std::endl;
  }

  return true;
}

// allocate device memory and copy from host
template <typename T>
bool device_copy_from_host(T** d_pointer, size_t size, void* h_pointer, const char* what = 0)
{
  if (!device_malloc(d_pointer, size, what))
    return false;
  if (cudaMemcpy(*d_pointer, h_pointer, size, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "failed to copy " << (what ? what : "data") << " from host to device" << std::endl;
    cudaFree(*d_pointer);
    *d_pointer = NULL;
    return false;
  }
  return true;
}

Word* setup_device_stream_compress(zfp_stream* stream)
{
  Word* d_stream = (Word*)stream->stream->begin;
  if (!cuZFP::is_gpu_ptr(d_stream)) {
    // allocate device memory for compressed data
    size_t size = stream_capacity(stream->stream);
    device_malloc(&d_stream, size, "stream");
  }

  return d_stream;
}

Word* setup_device_stream_decompress(zfp_stream* stream)
{
  Word* d_stream = (Word*)stream->stream->begin;
  if (!cuZFP::is_gpu_ptr(d_stream)) {
    // copy compressed data to device memory
    size_t size = stream_capacity(stream->stream);
    device_copy_from_host(&d_stream, size, stream->stream->begin, "stream");
  }

  return d_stream;
}

ushort* setup_device_index_compress(zfp_stream *stream, const zfp_field *field)
{
  ushort* d_index = stream->index ? (ushort*)stream->index->data : NULL;
  if (!cuZFP::is_gpu_ptr(d_index)) {
    // allocate device memory for block index
    size_t size = zfp_field_blocks(field) * sizeof(ushort);
    device_malloc(&d_index, size, "index");
  }

  return d_index;
}

Word* setup_device_index_decompress(zfp_stream* stream)
{
  Word* d_index = (Word*)stream->index->data;
  if (!cuZFP::is_gpu_ptr(d_index)) {
    // copy index to device memory
    size_t size = stream->index->size;
    device_copy_from_host(&d_index, size, stream->index->data, "index");
  }

  return d_index;
}

bool setup_device_chunking(size_t* chunk_size, unsigned long long** d_offsets, size_t* lcubtemp, void** d_cubtemp, uint processors)
{
  // TODO : Error handling for CUDA malloc and CUB?
  // Assuming 1 thread = 1 ZFP block,
  // launching 1024 threads per SM should give a decent occupancy
  *chunk_size = processors * 1024; 
  size_t size = (*chunk_size + 1) * sizeof(unsigned long long);
  if (!device_malloc(d_offsets, size, "offsets"))
    return false;
  cudaMemset(*d_offsets, 0, size); // ensure offsets are zeroed

  // Using CUB for the prefix sum. CUB needs a bit of temp memory too
  size_t tempsize;
  cub::DeviceScan::InclusiveSum(nullptr, tempsize, *d_offsets, *d_offsets, *chunk_size + 1);
  *lcubtemp = tempsize;
  if (!device_malloc(d_cubtemp, tempsize, "offsets")) {
    cudaFree(*d_offsets);
    *d_offsets = NULL;
    return false;
  }

  return true;
}

void* setup_device_field_compress(const zfp_field* field, void*& d_begin)
{
  void* d_data = field->data;
  if (cuZFP::is_gpu_ptr(d_data)) {
    // field already resides on device
    d_begin = zfp_field_begin(field);
    return d_data;
  }
  else {
    // GPU implementation currently requires contiguous field
    if (zfp_field_is_contiguous(field)) {
      // copy field from host to device
      size_t size = zfp_field_size(field, NULL) * zfp_type_size(field->type);
      void* h_begin = zfp_field_begin(field);
      if (!device_copy_from_host(&d_begin, size, h_begin, "field"))
        return NULL;
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
  if (cuZFP::is_gpu_ptr(d_data)) {
    // field has already been allocated on device
    d_begin = zfp_field_begin(field);
    return d_data;
  }
  else {
    // GPU implementation currently requires contiguous field
    if (zfp_field_is_contiguous(field)) {
      // allocate device memory for decompressed field
      size_t size = zfp_field_size(field, NULL) * zfp_type_size(field->type);
      if (!device_malloc(&d_begin, size, "field"))
        return NULL;
      void* h_begin = zfp_field_begin(field);
      // in case of negative strides, advance device pointer into buffer
      return device_pointer(d_begin, h_begin, d_data, field->type);
    }
    else
      return NULL;
  }
}

// copy from device to host (if needed) and deallocate device memory
// TODO: d_begin should be first argument, with begin = NULL as default
void cleanup_device(void* begin, void* d_begin, size_t bytes = 0)
{
  if (begin != d_begin) {
    // copy data from device to host and free device memory
    if (bytes)
      cudaMemcpy(begin, d_begin, bytes, cudaMemcpyDeviceToHost);
#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_begin, 0);
#else
    cudaFree(d_begin);
#endif
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
  ushort* d_index,          // block index device pointer
  uint minbits,             // minimum compressed #bits/block
  uint maxbits,             // maximum compressed #bits/block
  uint maxprec,             // maximum uncompressed #bits/value
  int minexp                // minimum bit plane index
)
{
  size_t bits_written = 0;

  ErrorCheck errors;

  uint dims = size[0] ? size[1] ? size[2] ? 3 : 2 : 1 : 0;
  switch (dims) {
    case 1:
      bits_written = cuZFP::encode1<T>(d_data, size, stride, d_stream, d_index, minbits, maxbits, maxprec, minexp);
      break;
    case 2:
      bits_written = cuZFP::encode2<T>(d_data, size, stride, d_stream, d_index, minbits, maxbits, maxprec, minexp);
      break;
    case 3:
      bits_written = cuZFP::encode3<T>(d_data, size, stride, d_stream, d_index, minbits, maxbits, maxprec, minexp);
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
      bits_read = cuZFP::decode1<T>(d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    case 2:
      bits_read = cuZFP::decode2<T>(d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    case 3:
      bits_read = cuZFP::decode3<T>(d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    default:
      break;
  }

  errors.chk("Decode");

  return bits_read;
}

// compact variable-rate stream
unsigned long long
compact_stream(
  Word* d_stream,
  uint maxbits,
  ushort* d_index,
  size_t blocks,
  size_t processors
)
{
  unsigned long long bits_written = 0;
  size_t chunk_size;
  unsigned long long *d_offsets;
  size_t lcubtemp;
  void *d_cubtemp;

  if (internal::setup_device_chunking(&chunk_size, &d_offsets, &lcubtemp, &d_cubtemp, processors)) {
    // in-place compact variable-length blocks stored as fixed-length records
    for (size_t i = 0; i < blocks; i += chunk_size) { 
      int cur_blocks = chunk_size;
      bool last_chunk = false;
      if (i + chunk_size > blocks) { 
        cur_blocks = (int)(blocks - i);
        last_chunk = true;
      }
      // copy the 16-bit lengths in the offset array
      cuZFP::copy_length_launch(d_index, d_offsets, i, cur_blocks);
    
      // prefix sum to turn length into offsets
      cub::DeviceScan::InclusiveSum(d_cubtemp, lcubtemp, d_offsets, d_offsets, cur_blocks + 1);
    
      // compact the stream array in-place
      cuZFP::chunk_process_launch((uint*)d_stream, d_offsets, i, cur_blocks, last_chunk, maxbits, processors);
    }
    // update compressed size and pad to whole words
    cudaMemcpy(&bits_written, d_offsets, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    bits_written = cuZFP::round_up(bits_written, sizeof(Word) * CHAR_BIT);

    // free temporary buffers
    internal::cleanup_device(NULL, d_offsets);
    internal::cleanup_device(NULL, d_cubtemp);
  }

  return bits_written;
}

} // namespace internal

// TODO: move out of global namespace
zfp_bool
cuda_init(zfp_exec_params_cuda* params)
{
  // ensure GPU word size equals CPU word size
  if (sizeof(Word) != sizeof(bitstream_word))
    return false;

  static bool initialized = false;
  static cudaDeviceProp prop;
  if (!initialized && cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
    return zfp_false;
  initialized = true;

  // TODO: take advantage of cached grid size
  params->processors = prop.multiProcessorCount;
  params->grid_size[0] = prop.maxGridSize[0];
  params->grid_size[1] = prop.maxGridSize[1];
  params->grid_size[2] = prop.maxGridSize[2];

  // TODO: launch warm-up kernel

  return zfp_true;
}

size_t
cuda_compress(zfp_stream* stream, const zfp_field* field)
{
  // determine compression mode and ensure it is supported
  bool variable_rate = false;
  switch (zfp_stream_compression_mode(stream)) {
    case zfp_mode_fixed_rate:
      break;
    case zfp_mode_fixed_precision:
    case zfp_mode_fixed_accuracy:
    case zfp_mode_expert:
      variable_rate = true;
      break;
    default:
      // unsupported compression mode
      return 0;
  }

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

  // copy field to device if not already there
  void* d_begin = NULL;
  void* d_data = internal::setup_device_field_compress(field, d_begin);

  // null means the array is non-contiguous host memory, which is not supported
  if (!d_data)
    return 0;

  // allocate compressed buffer
  Word* d_stream = internal::setup_device_stream_compress(stream);
  // TODO: populate stream->index even in fixed-rate mode if non-null
  ushort* d_index = variable_rate ? internal::setup_device_index_compress(stream, field) : NULL;

  // determine minimal slot needed to hold a compressed block
  uint maxbits = (uint)zfp_maximum_block_size_bits(stream, field);

  // encode data
  const bitstream_offset pos = stream_wtell(stream->stream);
  unsigned long long bits_written = 0;
  // TODO: internal::encode() should return ull
  switch (field->type) {
    case zfp_type_int32:
      bits_written = internal::encode((int*)d_data, size, stride, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    case zfp_type_int64:
      bits_written = internal::encode((long long int*)d_data, size, stride, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    case zfp_type_float:
      bits_written = internal::encode((float*)d_data, size, stride, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    case zfp_type_double:
      bits_written = internal::encode((double*)d_data, size, stride, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    default:
      break;
  }

  // compact stream of variable-length blocks stored in fixed-length slots
  if (variable_rate) {
    const size_t blocks = zfp_field_blocks(field);
    const size_t processors = ((zfp_exec_params_cuda*)stream->exec.params)->processors;
    bits_written = internal::compact_stream(d_stream, maxbits, d_index, blocks, processors);
  }

  const size_t stream_bytes = cuZFP::round_up((bits_written + CHAR_BIT - 1) / CHAR_BIT, sizeof(Word));

  if (d_index) {
    const size_t size = zfp_field_blocks(field) * sizeof(ushort);
    // TODO: assumes index stores block sizes
    internal::cleanup_device(stream->index ? stream->index->data : NULL, d_index, size);
  }

  // copy stream from device to host if needed and free temporary buffers
  internal::cleanup_device(stream->stream->begin, d_stream, stream_bytes);
  internal::cleanup_device(zfp_field_begin(field), d_begin);

  // update bit stream to point just past produced data
  if (bits_written)
    stream_wseek(stream->stream, pos + bits_written);

  return bits_written;
}

size_t
cuda_decompress(zfp_stream* stream, zfp_field* field)
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
      d_index = internal::setup_device_index_decompress(stream);
      break;
    default:
      // TODO: clean up device to avoid memory leak
      std::cerr << "zfp compression mode not supported on GPU" << std::endl;
      return 0;
  }

  // decode compressed data
  const bitstream_offset pos = stream_rtell(stream->stream);
  unsigned long long bits_read = 0;
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
