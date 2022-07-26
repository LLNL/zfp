include ../Config

.SUFFIXES: .f90

LIBDIR = ../lib
MODDIR = ../modules
TARGETS = $(LIBDIR)/libzFORp.a $(LIBDIR)/libzFORp.so $(MODDIR)/zforp.mod
OBJECTS = zfp.o
MODULES = zforp.mod

static: $(LIBDIR)/libzFORp.a $(MODDIR)/zforp.mod

shared: $(LIBDIR)/libzFORp.so $(MODDIR)/zforp.mod

clean:
	rm -f $(TARGETS) $(OBJECTS)

$(LIBDIR)/libzFORp.a: $(OBJECTS)
	mkdir -p $(LIBDIR)
	rm -f $@
	ar rc $@ $^

$(LIBDIR)/libzFORp.so: $(OBJECTS)
	mkdir -p $(LIBDIR)
	$(FC) $(FFLAGS) -shared $^ -o $@

$(MODDIR)/zforp.mod: $(OBJECTS)
	mkdir -p $(MODDIR)
	mv zforp.mod $(MODDIR)

.f90.o:
	$(FC) $(FFLAGS) -c $<
