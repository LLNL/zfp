include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/diffusion\
	  $(BINDIR)/inplace\
	  $(BINDIR)/iterator\
	  $(BINDIR)/pgm\
	  $(BINDIR)/ppm\
	  $(BINDIR)/simple\
	  $(BINDIR)/speed
INCS = -I../include
LIBS = -L../lib -lzfp
CLIBS = $(LIBS) -lm
CXXLIBS = $(LIBS)

all: $(TARGETS)

$(BINDIR)/diffusion: diffusion.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) -I../array diffusion.cpp $(CXXLIBS) -o $@

$(BINDIR)/inplace: inplace.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) $(INCS) inplace.c $(CLIBS) -o $@

$(BINDIR)/iterator: iterator.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) -I../array iterator.cpp $(CXXLIBS) -o $@

$(BINDIR)/pgm: pgm.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) $(INCS) pgm.c $(CLIBS) -o $@

$(BINDIR)/ppm: ppm.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) $(PPM_FLAGS) $(INCS) ppm.c $(CLIBS) -o $@

$(BINDIR)/simple: simple.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) $(INCS) simple.c $(CLIBS) -o $@

$(BINDIR)/speed: speed.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) $(INCS) speed.c $(CLIBS) -o $@

clean:
	rm -f $(TARGETS)
