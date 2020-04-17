// const reference to a 3D array or view element; this class is nested within container_type
class const_reference : const_handle {
public:
  typedef container_type::value_type value_type;

  // constructor
  explicit const_reference(container_type* container, uint i, uint j, uint k) : const_handle(container, i, j, k) {}

  // inspector
  operator value_type() const { return get(); }

  // pointer to referenced element
  const_pointer operator&() const { return const_pointer(container, i, j, k); }

protected:
  using const_handle::get;
  using const_handle::container;
  using const_handle::i;
  using const_handle::j;
  using const_handle::k;
};

// reference to a 3D array or view element; this class is nested within container_type
class reference : public const_reference {
public:
  // constructor
  explicit reference(container_type* container, uint i, uint j, uint k) : const_reference(container, i, j, k) {}

  // assignment
  reference operator=(const reference& r) { set(r.get()); return *this; }
  reference operator=(value_type val) { set(val); return *this; }

  // compound assignment
  reference operator+=(value_type val) { container->add(i, j, k, val); return *this; }
  reference operator-=(value_type val) { container->sub(i, j, k, val); return *this; }
  reference operator*=(value_type val) { container->mul(i, j, k, val); return *this; }
  reference operator/=(value_type val) { container->div(i, j, k, val); return *this; }

  // pointer to referenced element
  pointer operator&() const { return pointer(container, i, j, k); }

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
  void set(value_type val) { container->set(i, j, k, val); }

  using const_reference::get;
  using const_reference::container;
  using const_reference::i;
  using const_reference::j;
  using const_reference::k;
};
