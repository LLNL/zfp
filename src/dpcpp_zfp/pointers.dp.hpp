#ifndef SYCLZFP_POINTERS_CUH
#define SYCLZFP_POINTERS_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ErrorCheck.h"
#include <iostream>


namespace syclZFP
{
// https://gitlab.kitware.com/third-party/nvpipe/blob/master/encode.c
bool is_gpu_ptr(const void *ptr) try {
  int error = 0;

  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  auto ctxt = q_ct1.get_context();
  
  return ((get_pointer_type(ptr, ctxt) == sycl::usm::alloc::shared) || 
	(get_pointer_type(ptr, ctxt) == sycl::usm::alloc::device));

}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

} // namespace syclZFP

#endif
