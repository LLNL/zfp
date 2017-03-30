include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/diffusion-raw\
	  $(BINDIR)/diffusion-zfp\
	  $(BINDIR)/inplace\
	  $(BINDIR)/pgm\
	  $(BINDIR)/simple\
	  $(BINDIR)/speed
CLIBS = -L../lib -lzfp -lm
CXXLIBS = -L../lib -lzfp

all: $(TARGETS)

$(BINDIR)/diffusion-raw: diffusion.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -DWITHOUT_COMPRESSION -I../array diffusion.cpp $(CXXLIBS) -o $@

$(BINDIR)/diffusion-zfp: diffusion.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -I../array diffusion.cpp $(CXXLIBS) -o $@

$(BINDIR)/inplace: inplace.c ../lib/libzfp.a
	$(CC) $(CFLAGS) inplace.c $(CLIBS) -o $@

$(BINDIR)/pgm: pgm.c ../lib/libzfp.a
	$(CC) $(CFLAGS) pgm.c $(CLIBS) -o $@

$(BINDIR)/simple: simple.c ../lib/libzfp.a
	$(CC) $(CFLAGS) simple.c $(CLIBS) -o $@

$(BINDIR)/speed: speed.c ../lib/libzfp.a
	$(CC) $(CFLAGS) speed.c $(CLIBS) -o $@

clean:
	rm -f $(TARGETS)
