#ifndef ZFP_REFERENCE3_HPP
#define ZFP_REFERENCE3_HPP

namespace zfp {
namespace internal {
namespace dim3 {

// const reference to a 3D array or view element
template <class Container>
class const_reference : const_handle<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

  // constructor
  explicit const_reference(const container_type* container, size_t x, size_t y, size_t z) : const_handle<Container>(container, x, y, z) {}

  // inspector
  operator value_type() const { return get(); }

  // pointer to referenced element
  const_pointer<Container> operator&() const { return const_pointer<Container>(container, x, y, z); }

protected:
  using const_handle<Container>::get;
  using const_handle<Container>::container;
  using const_handle<Container>::x;
  using const_handle<Container>::y;
  using const_handle<Container>::z;
};

// reference to a 3D array or view element
template <class Container>
class reference : public const_reference<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

  // constructor
  explicit reference(container_type* container, size_t x, size_t y, size_t z) : const_reference<Container>(container, x, y, z) {}

  // copy constructor (to satisfy rule of three)
  reference(const reference& r) : const_reference<Container>(r.container, r.x, r.y, r.z) {}

  // assignment
  reference operator=(const reference& r) { set(r.get()); return *this; }
  reference operator=(value_type val) { set(val); return *this; }

  // compound assignment
  reference operator+=(value_type val) { container->add(x, y, z, val); return *this; }
  reference operator-=(value_type val) { container->sub(x, y, z, val); return *this; }
  reference operator*=(value_type val) { container->mul(x, y, z, val); return *this; }
  reference operator/=(value_type val) { container->div(x, y, z, val); return *this; }

  // pointer to referenced element
  pointer<Container> operator&() const { return pointer<Container>(container, x, y, z); }

  // swap two array elements via proxy references
  friend void swap(reference a, reference b)
  {
    value_type x = a.get();
    value_type y = b.get();
    b.set(x);
    a.set(y);
  }

protected:
  // assign value through reference
  void set(value_type val) { container->set(x, y, z, val); }

  using const_reference<Container>::get;
  using const_reference<Container>::container;
  using const_reference<Container>::x;
  using const_reference<Container>::y;
  using const_reference<Container>::z;
};

} // dim3
} // internal
} // zfp

#endif
