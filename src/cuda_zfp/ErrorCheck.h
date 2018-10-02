#ifndef ERRORCHECK_H
#define ERRORCHECK_H
#include <iostream>
#include <string>
#include <sstream>

using std::stringstream;
class ErrorCheck
{
public:
  ErrorCheck()
  {

  }

  void chk(std::string msg)
  {
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
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

#endif // ERRORCHECK_H
