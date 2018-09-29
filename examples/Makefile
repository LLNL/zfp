include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/diffusion\
	  $(BINDIR)/inplace\
	  $(BINDIR)/iterator\
	  $(BINDIR)/pgm\
	  $(BINDIR)/simple\
	  $(BINDIR)/speed
LIBS = -L../lib -lzfp
CLIBS = $(LIBS) -lm
CXXLIBS = $(LIBS)

all: $(TARGETS)

$(BINDIR)/diffusion: diffusion.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) -I../array diffusion.cpp $(CXXLIBS) -o $@

$(BINDIR)/inplace: inplace.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) inplace.c $(CLIBS) -o $@

$(BINDIR)/iterator: iterator.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) -I../array iterator.cpp $(CXXLIBS) -o $@

$(BINDIR)/pgm: pgm.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) pgm.c $(CLIBS) -o $@

$(BINDIR)/simple: simple.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) simple.c $(CLIBS) -o $@

$(BINDIR)/speed: speed.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) speed.c $(CLIBS) -o $@

clean:
	rm -f $(TARGETS)
