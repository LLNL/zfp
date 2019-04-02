class header_exception : public std::runtime_error {
public:
  header_exception(const std::string& msg) : runtime_error(msg) {}

  virtual ~header_exception() throw (){}
};
