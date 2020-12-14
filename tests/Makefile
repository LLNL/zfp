include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/testzfp $(BINDIR)/testviews
INCS = -I../include -I../array
LIBS = -L../lib -lzfp

all: $(TARGETS)

$(BINDIR)/testzfp: testzfp.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) testzfp.cpp $(LIBS) -o $@

$(BINDIR)/testviews: testviews.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) testviews.cpp $(LIBS) -o $@

test: $(BINDIR)/testzfp
	$(BINDIR)/testzfp

clean:
	rm -f $(TARGETS)
