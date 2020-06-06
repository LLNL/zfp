// const reference to a 2D array or view element; this class is nested within container_type
class const_reference : const_handle {
public:
  typedef typename container_type::value_type value_type;

  // constructor
  explicit const_reference(const container_type* container, size_t x, size_t y) : const_handle(container, x, y) {}

  // inspector
  operator value_type() const { return get(); }

  // pointer to referenced element
  const_pointer operator&() const { return const_pointer(container, x, y); }

protected:
  using const_handle::get;
  using const_handle::container;
  using const_handle::x;
  using const_handle::y;
};

// reference to a 2D array or view element; this class is nested within container_type
class reference : public const_reference {
public:
  // constructor
  explicit reference(container_type* container, size_t x, size_t y) : const_reference(container, x, y) {}

  // assignment
  reference operator=(const reference& r) { set(r.get()); return *this; }
  reference operator=(value_type val) { set(val); return *this; }

  // compound assignment
  reference operator+=(value_type val) { container->add(x, y, val); return *this; }
  reference operator-=(value_type val) { container->sub(x, y, val); return *this; }
  reference operator*=(value_type val) { container->mul(x, y, val); return *this; }
  reference operator/=(value_type val) { container->div(x, y, val); return *this; }

  // pointer to referenced element
  pointer operator&() const { return pointer(container, x, y); }

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
  void set(value_type val) { container->set(x, y, val); }

  using const_reference::get;
  using const_reference::container;
  using const_reference::x;
  using const_reference::y;
};
