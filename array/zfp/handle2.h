// const handle to a 2D array or view element; this class is nested within container_type
class const_handle {
public:
  typedef typename container_type::value_type value_type;

protected:
  // protected constructor
  explicit const_handle(const container_type* container, size_t x, size_t y) : container(const_cast<container_type*>(container)), x(x), y(y) {}

  // derefence handle
  value_type get() const { return container->get(x, y); }

  container_type* container; // container
  size_t x, y;               // global element index
};

// reference to a 2D array or view element; this class is nested within container_type
class handle : public const_handle {
protected:
  // protected constructor
  explicit handle(container_type* container, size_t x, size_t y) : const_handle(container, x, y) {}

  // assign value through handle
  void set(value_type val) { container->set(x, y, val); }

  using const_handle::container;
  using const_handle::x;
  using const_handle::y;
};
