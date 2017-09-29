include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/diffusion\
	  $(BINDIR)/inplace\
	  $(BINDIR)/iterator\
	  $(BINDIR)/pgm\
	  $(BINDIR)/simple\
	  $(BINDIR)/speed
CLIBS = -L../lib -lzfp -lm
CXXLIBS = -L../lib -lzfp

all: $(TARGETS)

$(BINDIR)/diffusion: diffusion.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -I../array diffusion.cpp $(CXXLIBS) -o $@

$(BINDIR)/inplace: inplace.c ../lib/libzfp.a
	$(CC) $(CFLAGS) inplace.c $(CLIBS) -o $@

$(BINDIR)/iterator: iterator.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -I../array iterator.cpp $(CXXLIBS) -o $@

$(BINDIR)/pgm: pgm.c ../lib/libzfp.a
	$(CC) $(CFLAGS) pgm.c $(CLIBS) -o $@

$(BINDIR)/simple: simple.c ../lib/libzfp.a
	$(CC) $(CFLAGS) simple.c $(CLIBS) -o $@

$(BINDIR)/speed: speed.c ../lib/libzfp.a
	$(CC) $(CFLAGS) speed.c $(CLIBS) -o $@

clean:
	rm -f $(TARGETS)
