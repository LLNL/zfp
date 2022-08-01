include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/testzfp
INCS = -I../include
LIBS = -L../lib -lzfp $(LDFLAGS)

all: $(TARGETS)

$(BINDIR)/testzfp: testzfp.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) testzfp.cpp $(LIBS) -o $@

test: $(BINDIR)/testzfp
	$(BINDIR)/testzfp

clean:
	rm -f $(TARGETS)
