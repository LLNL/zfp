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
    error = hipGetLastError();
    if (error != hipSuccess)
    {
      std::cout << msg << " : " << error;
      std::cout << " " << hipGetErrorString(error) << std::endl;
    }
  }

  void chk()
  {
    chk(str.str());
    str.str("");
  }
  hipError_t error;
  stringstream str;
};

#endif // ERRORCHECK_H
