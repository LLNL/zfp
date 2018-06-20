#ifndef CUZFP_DEBUG_UTILS_CUH
#define CUZFP_DEBUG_UTILS_CUH

template<typename T>
__device__ 
void Print(int tid, T val, const char* msg);

template<>
__device__ 
void Print<long long unsigned>(int tid, long long unsigned val, const char*msg)
{
  printf(" %s tid(%d): %llu\n", msg, tid, val);
}

template<>
__device__ 
void Print<unsigned int>(int tid, unsigned int val, const char*msg)
{
  printf(" %s tid(%d): %u\n", msg, tid, val);
}

template<>
__device__ 
void Print<unsigned char>(int tid, unsigned char val, const char*msg)
{
  printf(" %s tid(%d): %d\n", msg, tid, (int)val);
}

#endif
