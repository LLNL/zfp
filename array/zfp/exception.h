#ifndef ZFP_EXCEPTION_H
#define ZFP_EXCEPTION_H

namespace zfp {

// generic exception thrown by array constructors
class exception : public std::runtime_error {
public:
  exception(const std::string& msg) : runtime_error(msg) {}
  virtual ~exception() throw() {}
};

}

#endif
