include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/testzfp $(BINDIR)/testviews
CXXLIBS = -L../lib -lzfp

all: $(TARGETS)

$(BINDIR)/testzfp: testzfp.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) -I../array testzfp.cpp $(CXXLIBS) -o $@

$(BINDIR)/testviews: testviews.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) -I../array testviews.cpp $(CXXLIBS) -o $@

test: $(BINDIR)/testzfp
	$(BINDIR)/testzfp

clean:
	rm -f $(TARGETS)
