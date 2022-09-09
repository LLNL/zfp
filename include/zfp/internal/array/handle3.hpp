#ifndef ZFP_HANDLE3_HPP
#define ZFP_HANDLE3_HPP

namespace zfp {
namespace internal {
namespace dim3 {

// forward declarations
template <class Container> class const_reference;
template <class Container> class const_pointer;
template <class Container> class const_iterator;
template <class Container> class reference;
template <class Container> class pointer;
template <class Container> class iterator;

// const handle to a 3D array or view element
template <class Container>
class const_handle {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

protected:
  // protected constructor
  explicit const_handle(const container_type* container, size_t x, size_t y, size_t z) : container(const_cast<container_type*>(container)), x(x), y(y), z(z) {}

  // dereference handle
  value_type get() const { return container->get(x, y, z); }

  container_type* container; // container
  size_t x, y, z;            // global element index
};

} // dim3
} // internal
} // zfp

#endif
