#ifndef ZFP_REFERENCE1_HPP
#define ZFP_REFERENCE1_HPP

namespace zfp {
namespace internal {
namespace dim1 {

// const reference to a 1D array or view element
template <class Container>
class const_reference : const_handle<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

  // constructor
  explicit const_reference(const container_type* container, size_t x) : const_handle<Container>(container, x) {}

  // inspector
  operator value_type() const { return get(); }

  // pointer to referenced element
  const_pointer<Container> operator&() const { return const_pointer<Container>(container, x); }

protected:
  using const_handle<Container>::get;
  using const_handle<Container>::container;
  using const_handle<Container>::x;
};

// reference to a 1D array or view element
template <class Container>
class reference : public const_reference<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

  // constructor
  explicit reference(container_type* container, size_t x) : const_reference<Container>(container, x) {}

  // copy constructor (to satisfy rule of three)
  reference(const reference& r) : const_reference<Container>(r.container, r.x) {}

  // assignment
  reference operator=(const reference& r) { set(r.get()); return *this; }
  reference operator=(value_type val) { set(val); return *this; }

  // compound assignment
  reference operator+=(value_type val) { container->add(x, val); return *this; }
  reference operator-=(value_type val) { container->sub(x, val); return *this; }
  reference operator*=(value_type val) { container->mul(x, val); return *this; }
  reference operator/=(value_type val) { container->div(x, val); return *this; }

  // pointer to referenced element
  pointer<Container> operator&() const { return pointer<Container>(container, x); }

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
  void set(value_type val) { container->set(x, val); }

  using const_reference<Container>::get;
  using const_reference<Container>::container;
  using const_reference<Container>::x;
};

} // dim1
} // internal
} // zfp

#endif
