#ifndef ZFP_ITERATOR1_HPP
#define ZFP_ITERATOR1_HPP

namespace zfp {
namespace internal {
namespace dim1 {

// random access const iterator that visits 1D array or view block by block
template <class Container>
class const_iterator : public const_handle<Container> {
public:
  // typedefs for STL compatibility
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef ptrdiff_t difference_type;
  typedef zfp::internal::dim1::reference<Container> reference;
  typedef zfp::internal::dim1::pointer<Container> pointer;
  typedef std::random_access_iterator_tag iterator_category;

  typedef zfp::internal::dim1::const_reference<Container> const_reference;
  typedef zfp::internal::dim1::const_pointer<Container> const_pointer;

  // default constructor
  const_iterator() : const_handle<Container>(0, 0) {}

  // constructor
  explicit const_iterator(const container_type* container, size_t x) : const_handle<Container>(container, x) {}

  // dereference iterator
  const_reference operator*() const { return const_reference(container, x); }
  const_reference operator[](difference_type d) const { return *operator+(d); }

  // iterator arithmetic
  const_iterator operator+(difference_type d) const { const_iterator it = *this; it.advance(d); return it; }
  const_iterator operator-(difference_type d) const { return operator+(-d); }
  difference_type operator-(const const_iterator& it) const { return offset() - it.offset(); }

  // equality operators
  bool operator==(const const_iterator& it) const { return container == it.container && x == it.x; }
  bool operator!=(const const_iterator& it) const { return !operator==(it); }

  // relational operators
  bool operator<=(const const_iterator& it) const { return container == it.container && offset() <= it.offset(); }
  bool operator>=(const const_iterator& it) const { return container == it.container && offset() >= it.offset(); }
  bool operator<(const const_iterator& it) const { return container == it.container && offset() < it.offset(); }
  bool operator>(const const_iterator& it) const { return container == it.container && offset() > it.offset(); }

  // increment and decrement
  const_iterator& operator++() { increment(); return *this; }
  const_iterator& operator--() { decrement(); return *this; }
  const_iterator operator++(int) { const_iterator it = *this; increment(); return it; }
  const_iterator operator--(int) { const_iterator it = *this; decrement(); return it; }
  const_iterator operator+=(difference_type d) { advance(+d); return *this; }
  const_iterator operator-=(difference_type d) { advance(-d); return *this; }

  // local container index of value referenced by iterator
  size_t i() const { return x - container->min_x(); }

protected:
  // sequential offset associated with index x plus delta d
  difference_type offset(difference_type d = 0) const { return static_cast<difference_type>(x - container->min_x() + size_t(d)); }

  // index x associated with sequential offset p
  void index(size_t& x, difference_type p) const { x = container->min_x() + size_t(p); }

  // advance iterator by d
  void advance(difference_type d) { index(x, offset(d)); }

  // increment iterator to next element
  void increment() { ++x; }

  // decrement iterator to previous element
  void decrement() { --x; }

  using const_handle<Container>::container;
  using const_handle<Container>::x;
};

// random access iterator that visits 1D array or view block by block
template <class Container>
class iterator : public const_iterator<Container> {
public:
  // typedefs for STL compatibility
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef ptrdiff_t difference_type;
  typedef zfp::internal::dim1::reference<Container> reference;
  typedef zfp::internal::dim1::pointer<Container> pointer;
  typedef std::random_access_iterator_tag iterator_category;

  // default constructor
  iterator() : const_iterator<Container>(0, 0) {}

  // constructor
  explicit iterator(container_type* container, size_t i) : const_iterator<Container>(container, i) {}

  // dereference iterator
  reference operator*() const { return reference(container, x); }
  reference operator[](difference_type d) const { return *operator+(d); }

  // iterator arithmetic
  iterator operator+(difference_type d) const { iterator it = *this; it.advance(d); return it; }
  iterator operator-(difference_type d) const { return operator+(-d); }
  difference_type operator-(const iterator& it) const { return offset() - it.offset(); }

  // equality operators
  bool operator==(const iterator& it) const { return container == it.container && x == it.x; }
  bool operator!=(const iterator& it) const { return !operator==(it); }

  // relational operators
  bool operator<=(const iterator& it) const { return container == it.container && offset() <= it.offset(); }
  bool operator>=(const iterator& it) const { return container == it.container && offset() >= it.offset(); }
  bool operator<(const iterator& it) const { return container == it.container && offset() < it.offset(); }
  bool operator>(const iterator& it) const { return container == it.container && offset() > it.offset(); }

  // increment and decrement
  iterator& operator++() { increment(); return *this; }
  iterator& operator--() { decrement(); return *this; }
  iterator operator++(int) { iterator it = *this; increment(); return it; }
  iterator operator--(int) { iterator it = *this; decrement(); return it; }
  iterator operator+=(difference_type d) { advance(+d); return *this; }
  iterator operator-=(difference_type d) { advance(-d); return *this; }

protected:
  using const_iterator<Container>::offset;
  using const_iterator<Container>::advance;
  using const_iterator<Container>::increment;
  using const_iterator<Container>::decrement;
  using const_iterator<Container>::container;
  using const_iterator<Container>::x;
};

} // dim1
} // internal
} // zfp

#endif
