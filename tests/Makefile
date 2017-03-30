include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/testzfp
CXXLIBS = -L../lib -lzfp

all: $(TARGETS)

$(BINDIR)/testzfp: testzfp.cpp fields.o ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -I../array testzfp.cpp fields.o $(CXXLIBS) -o $@

fields.o: fields.c fields.h
	$(CC) $(CFLAGS) -c fields.c

test: $(BINDIR)/testzfp
	$(BINDIR)/testzfp

clean:
	rm -f $(TARGETS) fields.o
