#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef syclZFP_TYPE_INFO
#define syclZFP_TYPE_INFO

#include <cfloat>

namespace syclZFP {

template<typename T> inline int get_ebias();
template<> inline int get_ebias<double>() { return 1023; }
template<> inline int get_ebias<float>() { return 127; }
template<> inline int get_ebias<long long int>() { return 0; }
template<> inline int get_ebias<int>() { return 0; }

template<typename T> inline int get_ebits();
template<> inline int get_ebits<double>() { return 11; }
template<> inline int get_ebits<float>() { return 8; }
template<> inline int get_ebits<int>() { return 0; }
template<> inline int get_ebits<long long int>() { return 0; }

template<typename T> inline int get_precision();
template<> inline int get_precision<double>() { return 64; }
template<> inline int get_precision<long long int>() { return 64; }
template<> inline int get_precision<float>() { return 32; }
template<> inline int get_precision<int>() { return 32; }

template<typename T> inline int get_min_exp();
template<> inline int get_min_exp<double>() { return -1074; }
template<> inline int get_min_exp<float>() { return -1074; }
template<> inline int get_min_exp<long long int>() { return 0; }
template<> inline int get_min_exp<int>() { return 0; }

template<typename T> inline T get_scalar_min();
template<> inline float get_scalar_min<float>() { return FLT_MIN; }
template<> inline double get_scalar_min<double>() { return DBL_MIN; }
template<> inline long long int get_scalar_min<long long int>() { return 0; }
template<> inline int get_scalar_min<int>() { return 0; }

template<typename T> inline int scalar_sizeof();

template<> inline int scalar_sizeof<double>() { return 8; }
template<> inline int scalar_sizeof<long long int>() { return 8; }
template<> inline int scalar_sizeof<float>() { return 4; }
template<> inline int scalar_sizeof<int>() { return 4; }

template<typename T> inline T get_nbmask();

template<> inline unsigned int get_nbmask<unsigned int>() { return 0xaaaaaaaau; }
template<> inline unsigned long long int get_nbmask<unsigned long long int>() { return 0xaaaaaaaaaaaaaaaaull; }

template<typename T> struct zfp_traits;

template<> struct zfp_traits<double>
{
  typedef unsigned long long int UInt;
  typedef long long int Int;
};

template<> struct zfp_traits<long long int>
{
  typedef unsigned long long int UInt;
  typedef long long int Int;
};

template<> struct zfp_traits<float>
{
  typedef unsigned int UInt;
  typedef int Int;
};

template<> struct zfp_traits<int>
{
  typedef unsigned int UInt;
  typedef int Int;
};

template<typename T> inline bool is_int()
{
  return false;
}

template<> inline bool is_int<int>()
{
  return true;
}

template<> inline bool is_int<long long int>()
{
  return true;
}
#if 0
template<int T> struct block_traits;

template<> struct block_traits<1>
{
  typedef unsigned char PlaneType;
};

template<> struct block_traits<2>
{
  typedef unsigned short PlaneType;
};
#endif

} // namespace syclZFP
#endif
