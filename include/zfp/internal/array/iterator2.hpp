#ifndef ZFP_ITERATOR2_HPP
#define ZFP_ITERATOR2_HPP

namespace zfp {
namespace internal {
namespace dim2 {

// random access const iterator that visits 2D array or view block by block
template <class Container>
class const_iterator : public const_handle<Container> {
public:
  // typedefs for STL compatibility
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef ptrdiff_t difference_type;
  typedef zfp::internal::dim2::reference<Container> reference;
  typedef zfp::internal::dim2::pointer<Container> pointer;
  typedef std::random_access_iterator_tag iterator_category;

  typedef zfp::internal::dim2::const_reference<Container> const_reference;
  typedef zfp::internal::dim2::const_pointer<Container> const_pointer;

  // default constructor
  const_iterator() : const_handle<Container>(0, 0, 0) {}

  // constructor
  explicit const_iterator(const container_type* container, size_t x, size_t y) : const_handle<Container>(container, x, y) {}

  // dereference iterator
  const_reference operator*() const { return const_reference(container, x, y); }
  const_reference operator[](difference_type d) const { return *operator+(d); }

  // iterator arithmetic
  const_iterator operator+(difference_type d) const { const_iterator it = *this; it.advance(d); return it; }
  const_iterator operator-(difference_type d) const { return operator+(-d); }
  difference_type operator-(const const_iterator& it) const { return offset() - it.offset(); }

  // equality operators
  bool operator==(const const_iterator& it) const { return container == it.container && x == it.x && y == it.y; }
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
  size_t j() const { return y - container->min_y(); }

protected:
  // sequential offset associated with index (x, y) plus delta d
  difference_type offset(difference_type d = 0) const
  {
    difference_type p = d;
    size_t xmin = container->min_x();
    size_t xmax = container->max_x();
    size_t ymin = container->min_y();
    size_t ymax = container->max_y();
    size_t nx = xmax - xmin;
    size_t ny = ymax - ymin;
    if (y == ymax)
      p += nx * ny;
    else {
      size_t m = ~size_t(3);
      size_t by = std::max(y & m, ymin); size_t sy = std::min((by + 4) & m, ymax) - by; p += (by - ymin) * nx;
      size_t bx = std::max(x & m, xmin); size_t sx = std::min((bx + 4) & m, xmax) - bx; p += (bx - xmin) * sy;
      p += (y - by) * sx;
      p += (x - bx);
    }
    return p;
  }

  // index (x, y) associated with sequential offset p
  void index(size_t& x, size_t& y, difference_type p) const
  {
    size_t xmin = container->min_x();
    size_t xmax = container->max_x();
    size_t ymin = container->min_y();
    size_t ymax = container->max_y();
    size_t nx = xmax - xmin;
    size_t ny = ymax - ymin;
    if (size_t(p) == nx * ny) {
      x = xmin;
      y = ymax;
    }
    else {
      size_t m = ~size_t(3);
      size_t by = std::max((ymin + size_t(p / ptrdiff_t(nx))) & m, ymin); size_t sy = std::min((by + 4) & m, ymax) - by; p -= (by - ymin) * nx;
      size_t bx = std::max((xmin + size_t(p / ptrdiff_t(sy))) & m, xmin); size_t sx = std::min((bx + 4) & m, xmax) - bx; p -= (bx - xmin) * sy;
      y = by + size_t(p / ptrdiff_t(sx)); p -= (y - by) * sx;
      x = bx + size_t(p);                 p -= (x - bx);
    }
  }

  // advance iterator by d
  void advance(difference_type d) { index(x, y, offset(d)); }

  // increment iterator to next element
  void increment()
  {
    size_t xmin = container->min_x();
    size_t xmax = container->max_x();
    size_t ymin = container->min_y();
    size_t ymax = container->max_y();
    size_t m = ~size_t(3);
    ++x;
    if (!(x & 3u) || x == xmax) {
      x = std::max((x - 1) & m, xmin);
      ++y;
      if (!(y & 3u) || y == ymax) {
        y = std::max((y - 1) & m, ymin);
        // done with block; advance to next
        x = (x + 4) & m;
        if (x >= xmax) {
          x = xmin;
          y = (y + 4) & m;
          if (y >= ymax)
            y = ymax;
        }
      }
    }
  }

  // decrement iterator to previous element
  void decrement()
  {
    size_t xmin = container->min_x();
    size_t xmax = container->max_x();
    size_t ymin = container->min_y();
    size_t ymax = container->max_y();
    size_t m = ~size_t(3);
    if (y == ymax) {
      x = xmax - 1;
      y = ymax - 1;
    }
    else {
      if (!(x & 3u) || x == xmin) {
        x = std::min((x + 4) & m, xmax);
        if (!(y & 3u) || y == ymin) {
          y = std::min((y + 4) & m, ymax);
          // done with block; advance to next
          x = (x - 1) & m;
          if (x <= xmin) {
            x = xmax;
            y = (y - 1) & m;
            if (y <= ymin)
              y = ymin;
          }
        }
        --y;
      }
      --x;
    }
  }

  using const_handle<Container>::container;
  using const_handle<Container>::x;
  using const_handle<Container>::y;
};

// random access iterator that visits 2D array or view block by block
template <class Container>
class iterator : public const_iterator<Container> {
public:
  // typedefs for STL compatibility
  typedef Container container_type;
  typedef typename container_type::value_type value_type;
  typedef ptrdiff_t difference_type;
  typedef zfp::internal::dim2::reference<Container> reference;
  typedef zfp::internal::dim2::pointer<Container> pointer;
  typedef std::random_access_iterator_tag iterator_category;

  // default constructor
  iterator() : const_iterator<Container>(0, 0, 0) {}

  // constructor
  explicit iterator(container_type* container, size_t x, size_t y) : const_iterator<Container>(container, x, y) {}

  // dereference iterator
  reference operator*() const { return reference(container, x, y); }
  reference operator[](difference_type d) const { return *operator+(d); }

  // iterator arithmetic
  iterator operator+(difference_type d) const { iterator it = *this; it.advance(d); return it; }
  iterator operator-(difference_type d) const { return operator+(-d); }
  difference_type operator-(const iterator& it) const { return offset() - it.offset(); }

  // equality operators
  bool operator==(const iterator& it) const { return container == it.container && x == it.x && y == it.y; }
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
  using const_iterator<Container>::y;
};

} // dim2
} // internal
} // zfp

#endif
