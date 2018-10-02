include ../../Config

CXXFLAGS += -I../../include -I../../src -I../../array
LIBDIR = ../../lib
TARGETS = $(LIBDIR)/libcfp.a $(LIBDIR)/libcfp.so
OBJECTS = cfparrays.o

static: $(LIBDIR)/libcfp.a

shared: $(LIBDIR)/libcfp.so

clean:
	rm -f $(TARGETS) $(OBJECTS)

$(LIBDIR)/libcfp.a: $(OBJECTS)
	mkdir -p $(LIBDIR)
	rm -f $@
	ar rc $@ $^

$(LIBDIR)/libcfp.so: $(OBJECTS)
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXLAGS) -shared $(SOFLAGS) $^ -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<
