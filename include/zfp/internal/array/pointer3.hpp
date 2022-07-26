#ifndef ZFP_POINTER3_HPP
#define ZFP_POINTER3_HPP

namespace zfp {
namespace internal {
namespace dim3 {

// const pointer to a 3D array or view element
template <class Container>
class const_pointer : public const_handle<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

  // default constructor
  const_pointer() : const_handle<Container>(0, 0, 0, 0) {}
#if defined(__cplusplus) && __cplusplus >= 201103L
  const_pointer(std::nullptr_t) : const_handle<Container>(0, 0, 0, 0) {}
#endif

  // constructor
  explicit const_pointer(const container_type* container, size_t x, size_t y, size_t z) : const_handle<Container>(container, x, y, z) {}

  // dereference pointer
  const_reference<Container> operator*() const { return const_reference<Container>(container, x, y, z); }
  const_reference<Container> operator[](ptrdiff_t d) const { return *operator+(d); }

  // pointer arithmetic
  const_pointer operator+(ptrdiff_t d) const { const_pointer p = *this; p.advance(d); return p; }
  const_pointer operator-(ptrdiff_t d) const { return operator+(-d); }
  ptrdiff_t operator-(const const_pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const const_pointer& p) const { return container == p.container && x == p.x && y == p.y && z == p.z; }
  bool operator!=(const const_pointer& p) const { return !operator==(p); }

  // relational operators
  bool operator<=(const const_pointer& p) const { return container == p.container && offset() <= p.offset(); }
  bool operator>=(const const_pointer& p) const { return container == p.container && offset() >= p.offset(); }
  bool operator<(const const_pointer& p) const { return container == p.container && offset() < p.offset(); }
  bool operator>(const const_pointer& p) const { return container == p.container && offset() > p.offset(); }

  // increment and decrement
  const_pointer& operator++() { increment(); return *this; }
  const_pointer& operator--() { decrement(); return *this; }
  const_pointer operator++(int) { const_pointer p = *this; increment(); return p; }
  const_pointer operator--(int) { const_pointer p = *this; decrement(); return p; }
  const_pointer operator+=(ptrdiff_t d) { advance(+d); return *this; }
  const_pointer operator-=(ptrdiff_t d) { advance(-d); return *this; }

protected:
  ptrdiff_t offset(ptrdiff_t d = 0) const { return static_cast<ptrdiff_t>(x - container->min_x() + container->size_x() * (y - container->min_y() + container->size_y() * (z - container->min_z()))) + d; }
  void index(size_t& x, size_t& y, size_t& z, ptrdiff_t p) const
  {
    x = container->min_x() + size_t(p % ptrdiff_t(container->size_x())); p /= container->size_x();
    y = container->min_y() + size_t(p % ptrdiff_t(container->size_y())); p /= container->size_y();
    z = container->min_z() + size_t(p);
  }
  void advance(ptrdiff_t d) { index(x, y, z, offset(d)); }
  void increment()
  {
    if (++x == container->max_x()) {
      x = container->min_x();
      if (++y == container->max_y()) {
        y = container->min_y();
        ++z;
      }
    }
  }
  void decrement()
  {
    if (x-- == container->min_x()) {
      x += container->size_x();
      if (y-- == container->min_y()) {
        y += container->size_y();
        --z;
      }
    }
  }

  using const_handle<Container>::container;
  using const_handle<Container>::x;
  using const_handle<Container>::y;
  using const_handle<Container>::z;
};

// pointer to a 3D array or view element
template <class Container>
class pointer : public const_pointer<Container> {
public:
  typedef Container container_type;
  typedef typename container_type::value_type value_type;

  // default constructor
  pointer() : const_pointer<Container>(0, 0, 0, 0) {}
#if defined(__cplusplus) && __cplusplus >= 201103L
  pointer(std::nullptr_t) : const_pointer<Container>(0, 0, 0, 0) {}
#endif

  // constructor
  explicit pointer(container_type* container, size_t x, size_t y, size_t z) : const_pointer<Container>(container, x, y, z) {}

  // dereference pointer
  reference<Container> operator*() const { return reference<Container>(container, x, y, z); }
  reference<Container> operator[](ptrdiff_t d) const { return *operator+(d); }

  // pointer arithmetic
  pointer operator+(ptrdiff_t d) const { pointer p = *this; p.advance(d); return p; }
  pointer operator-(ptrdiff_t d) const { return operator+(-d); }
  ptrdiff_t operator-(const pointer& p) const { return offset() - p.offset(); }

  // equality operators
  bool operator==(const pointer& p) const { return container == p.container && x == p.x && y == p.y && z == p.z; }
  bool operator!=(const pointer& p) const { return !operator==(p); }

  // relational operators
  bool operator<=(const pointer& p) const { return container == p.container && offset() <= p.offset(); }
  bool operator>=(const pointer& p) const { return container == p.container && offset() >= p.offset(); }
  bool operator<(const pointer& p) const { return container == p.container && offset() < p.offset(); }
  bool operator>(const pointer& p) const { return container == p.container && offset() > p.offset(); }

  // increment and decrement
  pointer& operator++() { increment(); return *this; }
  pointer& operator--() { decrement(); return *this; }
  pointer operator++(int) { pointer p = *this; increment(); return p; }
  pointer operator--(int) { pointer p = *this; decrement(); return p; }
  pointer operator+=(ptrdiff_t d) { advance(+d); return *this; }
  pointer operator-=(ptrdiff_t d) { advance(-d); return *this; }

protected:
  using const_pointer<Container>::offset;
  using const_pointer<Container>::advance;
  using const_pointer<Container>::increment;
  using const_pointer<Container>::decrement;
  using const_pointer<Container>::container;
  using const_pointer<Container>::x;
  using const_pointer<Container>::y;
  using const_pointer<Container>::z;
};

} // dim3
} // internal
} // zfp

#endif
