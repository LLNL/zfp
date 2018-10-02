include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/testzfp
CXXLIBS = -L../lib -lzfp

all: $(TARGETS)

$(BINDIR)/testzfp: testzfp.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) -I../array testzfp.cpp $(CXXLIBS) -o $@

test: $(BINDIR)/testzfp
	$(BINDIR)/testzfp

clean:
	rm -f $(TARGETS)
