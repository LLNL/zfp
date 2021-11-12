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

# add cfp examples when BUILD_CFP is enabled
ifneq ($(BUILD_CFP),0)
  TARGETS += $(BINDIR)/diffusionC
endif


all: $(TARGETS)

$(BINDIR)/diffusion: diffusion.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) -I../array diffusion.cpp $(CXXLIBS) -o $@

$(BINDIR)/diffusionC: diffusionC.o ../lib/$(LIBZFP) ../lib/$(LIBCFP)
	$(CXX) $(CXXFLAGS) diffusionC.o -lcfp $(CLIBS) -o $@

diffusionC.o: diffusionC.c
	$(CC) $(CFLAGS) $(INCS) -I../cfp/include -c diffusionC.c

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
	rm -f $(TARGETS) diffusionC.o
