include ../Config

BINDIR = ../bin
TARGETS = $(BINDIR)/array\
	  $(BINDIR)/diffusion\
	  $(BINDIR)/inplace\
	  $(BINDIR)/iterator\
	  $(BINDIR)/pgm\
	  $(BINDIR)/ppm\
	  $(BINDIR)/simple\
	  $(BINDIR)/speed
INCS = -I../include
LIBS = -L../lib -lzfp
CLIBS = $(LIBS) $(LDFLAGS) -lm
CXXLIBS = $(LIBS) $(LDFLAGS)

# add cfp examples when BUILD_CFP is enabled
ifneq ($(BUILD_CFP),0)
  TARGETS += $(BINDIR)/diffusionC $(BINDIR)/iteratorC
endif


all: $(TARGETS)

$(BINDIR)/array: array.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) array.cpp $(CXXLIBS) -o $@

$(BINDIR)/diffusion: diffusion.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) diffusion.cpp $(CXXLIBS) -o $@

$(BINDIR)/diffusionC: diffusionC.o ../lib/$(LIBZFP) ../lib/$(LIBCFP)
	$(CXX) $(CXXFLAGS) diffusionC.o -lcfp $(CLIBS) -o $@

diffusionC.o: diffusionC.c
	$(CC) $(CFLAGS) $(INCS) -c diffusionC.c

$(BINDIR)/inplace: inplace.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) $(INCS) inplace.c $(CLIBS) -o $@

$(BINDIR)/iterator: iterator.cpp ../lib/$(LIBZFP)
	$(CXX) $(CXXFLAGS) $(INCS) iterator.cpp $(CXXLIBS) -o $@

$(BINDIR)/iteratorC: iteratorC.o ../lib/$(LIBZFP) ../lib/$(LIBCFP)
	$(CXX) $(CXXFLAGS) iteratorC.o -lcfp $(CLIBS) -o $@

iteratorC.o: iteratorC.c
	$(CC) $(CFLAGS) $(INCS) -c iteratorC.c

$(BINDIR)/pgm: pgm.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) $(INCS) pgm.c $(CLIBS) -o $@

$(BINDIR)/ppm: ppm.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) $(PPM_FLAGS) $(INCS) ppm.c $(CLIBS) -o $@

$(BINDIR)/simple: simple.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) $(INCS) simple.c $(CLIBS) -o $@

$(BINDIR)/speed: speed.c ../lib/$(LIBZFP)
	$(CC) $(CFLAGS) $(INCS) speed.c $(CLIBS) -o $@

clean:
	rm -f $(TARGETS) $(BINDIR)/diffusionC $(BINDIR)/iteratorC diffusionC.o iteratorC.o
