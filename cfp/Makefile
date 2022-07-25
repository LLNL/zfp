include ../Config

LIBDIR = ../lib
TARGETS = $(LIBDIR)/libcfp.a $(LIBDIR)/libcfp.so
OBJECTS = cfp.o
INCS = -I../include -I../src

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
	$(CXX) $(CXXFLAGS) -shared $(SOFLAGS) $^ -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCS) -c $<
