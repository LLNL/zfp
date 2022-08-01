#ifndef ZFP_HANDLE1_HPP
#define ZFP_HANDLE1_HPP

namespace zfp {
namespace internal {
namespace dim1 {

// forward declarations
template <class Container> class const_reference;
template <class Container> class const_pointer;
template <class Container> class const_iterator;
template <class Container> class reference;
template <class Container> class pointer;
template <class Container> class iterator;

// const handle to a 1D array or view element
template <class Container>
class const_handle {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

protected:
  // protected constructor
  explicit const_handle(const container_type* container, size_t x) : container(const_cast<container_type*>(container)), x(x) {}

  // dereference handle
  value_type get() const { return container->get(x); }

  container_type* container; // container
  size_t x;                  // global element index
};

} // dim1
} // internal
} // zfp

#endif
