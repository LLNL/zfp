#ifndef ZFP_CUDA_DEVICE_CUH
#define ZFP_CUDA_DEVICE_CUH

#define ZFP_MAGIC 0x7a667000u

namespace zfp {
namespace cuda {
namespace internal {

// dummy kernel for warming the GPU
__global__
void device_init_kernel(unsigned int* d_magic)
{
  *d_magic = ZFP_MAGIC;
}

// warm up device by launching dummy kernel
bool device_init()
{
  bool success = true;
  ErrorCheck error;

  // allocate device memory
  unsigned int* d_word = NULL;
  cudaMalloc(&d_word, sizeof(*d_word));
  success &= error.check("zfp device init - cudaMalloc");

  // launch a dummy kernel
  device_init_kernel<<<1, 1>>>(d_word);
  success &= error.check("zfp device init - kernel");

  // allocate host memory
  unsigned int* h_word = NULL;
  cudaMallocHost(&h_word, sizeof(*h_word));
  success &= error.check("zfp device init - cudaMallocHost");

  // copy from device to host
  if (success) {
    cudaMemcpy(h_word, d_word, sizeof(*h_word), cudaMemcpyDeviceToHost);
    success &= error.check("zfp device init - cudaMemcpy");
    success &= (*h_word == ZFP_MAGIC);
  }

  // free host memory
  cudaFreeHost(h_word);
  success &= error.check("zfp device init - cudaFreeHost");

  // free device memory
  cudaFree(d_word);
  success &= error.check("zfp device init - cudaFree");

  return success;
}

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
  bool success = malloc_async(d_pointer, size);

#ifdef ZFP_DEBUG
  if (!success) {
    std::cerr << "zfp_cuda : failed to allocate device memory";
    if (what)
      std::cerr << " for " << what;
    std::cerr << std::endl;
  }
#endif

  return success;
}

// allocate device memory and copy from host
template <typename T>
bool device_copy_from_host(T** d_pointer, size_t size, void* h_pointer, const char* what = 0)
{
  if (!device_malloc(d_pointer, size, what))
    return false;
  if (cudaMemcpy(*d_pointer, h_pointer, size, cudaMemcpyHostToDevice) != cudaSuccess) {
#ifdef ZFP_DEBUG
    std::cerr << "zfp_cuda : failed to copy " << (what ? what : "data") << " from host to device" << std::endl;
#endif
    cudaFree(*d_pointer);
    *d_pointer = NULL;
    return false;
  }
  return true;
}

Word* setup_device_stream_compress(zfp_stream* stream)
{
  Word* d_stream = (Word*)stream->stream->begin;
  if (!is_gpu_ptr(d_stream)) {
    // allocate device memory for compressed data
    size_t size = stream_capacity(stream->stream);
    device_malloc(&d_stream, size, "stream");
  }

  return d_stream;
}

Word* setup_device_stream_decompress(zfp_stream* stream)
{
  Word* d_stream = (Word*)stream->stream->begin;
  if (!is_gpu_ptr(d_stream)) {
    // copy compressed data to device memory
    size_t size = stream_capacity(stream->stream);
    device_copy_from_host(&d_stream, size, stream->stream->begin, "stream");
  }

  return d_stream;
}

ushort* setup_device_index_compress(zfp_stream *stream, const zfp_field *field)
{
  ushort* d_index = stream->index ? (ushort*)stream->index->data : NULL;
  if (!is_gpu_ptr(d_index)) {
    // allocate device memory for block index
    size_t size = zfp_field_blocks(field) * sizeof(ushort);
    device_malloc(&d_index, size, "index");
  }

  return d_index;
}

Word* setup_device_index_decompress(zfp_stream* stream)
{
  Word* d_index = (Word*)stream->index->data;
  if (!is_gpu_ptr(d_index)) {
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
  if (is_gpu_ptr(d_data)) {
    // field already resides on device
    d_begin = zfp_field_begin(field);
    return d_data;
  }
  else {
    // GPU implementation currently requires contiguous field
    if (zfp_field_is_contiguous(field)) {
      // copy field from host to device
      size_t size = zfp_field_size_bytes(field);
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
  if (is_gpu_ptr(d_data)) {
    // field has already been allocated on device
    d_begin = zfp_field_begin(field);
    return d_data;
  }
  else {
    // GPU implementation currently requires contiguous field
    if (zfp_field_is_contiguous(field)) {
      // allocate device memory for decompressed field
      size_t size = zfp_field_size_bytes(field);
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
  if (d_begin != begin) {
    // copy data from device to host and free device memory
    if (begin && bytes)
      cudaMemcpy(begin, d_begin, bytes, cudaMemcpyDeviceToHost);
    free_async(d_begin);
  }
}

} // namespace internal
} // namespace cuda
} // namespace zfp

#endif
