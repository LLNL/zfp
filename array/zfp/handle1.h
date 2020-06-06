// const handle to a 1D array or view element; this class is nested within container_type
class const_handle {
public:
  typedef typename container_type::value_type value_type;

protected:
  // protected constructor
  explicit const_handle(const container_type* container, size_t x) : container(const_cast<container_type*>(container)), x(x) {}

  // derefence handle
  value_type get() const { return container->get(x); }

  container_type* container; // container
  size_t x;                  // global element index
};

// reference to a 1D array or view element; this class is nested within container_type
class handle : public const_handle {
protected:
  // protected constructor
  explicit handle(container_type* container, size_t x) : const_handle(container, x) {}

  // assign value through handle
  void set(value_type val) { container->set(x, val); }

  using const_handle::container;
  using const_handle::x;
};
