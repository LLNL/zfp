#ifndef ERRORCHECK_H
#define ERRORCHECK_H
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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
    error = 0;
    if (error != 0)
    {
      std::cout << msg << " : " << error;
      std::cout
          << " "
          << "cudaGetErrorString not supported" /*cudaGetErrorString(error)*/
          << std::endl;
    }
  }

  void chk()
  {
    chk(str.str());
    str.str("");
  }
  int error;
  stringstream str;
};

#endif // ERRORCHECK_H
