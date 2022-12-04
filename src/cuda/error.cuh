#ifndef CUZFP_ERROR_CUH
#define CUZFP_ERROR_CUH

#include <iostream>
#include <sstream>
#include <string>

using std::stringstream;

// TODO: put in appropriate namespace
class ErrorCheck {
public:
  ErrorCheck()
  {
  }

  void chk(std::string msg)
  {
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::cout << msg << " : " << error;
      std::cout << " " << cudaGetErrorString(error) << std::endl;
    }
  }

  void chk()
  {
    chk(str.str());
    str.str("");
  }

  cudaError error;
  stringstream str;
};

#endif // CUZFP_ERROR_CUH
