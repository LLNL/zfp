#ifdef ZFP_WITH_CUDA
#include <cuZFP.h>
#elif defined ZFP_WITH_SYCL
#include <syclZFP.h>
#endif

#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <cstdio>


int main()
{
  //
  // this test is a simple sanity check to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 128;
  int y = 128;
  int z = 16;
	
  const int size = x * y * z;
  std::vector<float> test_data;
  test_data.resize(size);
  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }

  zfp_stream zfp;  
  zfp_field *field;  

  field = zfp_field_3d(&test_data[0], 
                       zfp_type_float,
                       x, y, z);
  
  int rate = 8;

  zfp_stream_set_rate(&zfp, rate, field->type, 3, 0);

  size_t buffsize = zfp_stream_maximum_size(&zfp, field);
  unsigned char* buffer = new unsigned char[buffsize];
  bitstream* s = stream_open(buffer, buffsize);
  //zfp.stream = (uchar*) buffer;
  zfp_stream_set_bit_stream(&zfp, s);
#ifdef ZFP_WITH_CUDA
  cuda_compress(&zfp, field);
#elif defined ZFP_WITH_SYCL
  sycl_compress(&zfp, field);
#endif

  std::vector<float> test_data_out;
  test_data_out.resize(size);

  zfp_field *out_field;  

  out_field = zfp_field_3d(&test_data_out[0], 
                           zfp_type_float,
                           x, y, z);

#ifdef ZFP_WITH_CUDA
  cuda_decompress(&zfp, out_field);
#elif defined ZFP_WITH_SYCL
  sycl_decompress(&zfp, out_field);
#endif

  bool isError = false;
  for(int i = 0; i < size; ++i)
  {
    if(i != static_cast<int>(test_data_out[i])){
	std::cout<<"i = "<< i << "value= "<< static_cast<int>(test_data_out[i])<< "Error\n";
    	isError = true;
    }
  }
  
  if (!isError)
  std::cout<<"Results match !!\n";	  


  zfp_field_free(out_field);
  zfp_field_free(field);
  delete[] buffer;

}

