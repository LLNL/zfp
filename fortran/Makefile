include ../Config

LIBDIR = ../lib
MODDIR = ../modules
TARGETS = $(LIBDIR)/libzFORp.a $(LIBDIR)/libzFORp.so $(MODDIR)/zfp.mod
OBJECTS = ../src/zfp.o
MODULES = zfp.mod

static: $(LIBDIR)/libzFORp.a $(MODDIR)/zfp.mod

shared: $(LIBDIR)/libzFORp.so $(MODDIR)/zfp.mod

clean:
	rm -f $(TARGETS) $(OBJECTS)

$(LIBDIR)/libzFORp.a: $(OBJECTS)
	mkdir -p $(LIBDIR)
	rm -f $@
	ar rc $@ $^

$(LIBDIR)/libzFORp.so: $(OBJECTS)
	mkdir -p $(LIBDIR)
	$(FC) $(FFLAGS) -shared $^ -o $@

$(MODDIR)/zfp.mod: $(OBJECTS)
	mkdir -p $(MODDIR)
	mv zfp.mod $(MODDIR)

.f.o:
	$(FC) $(FFLAGS) -c $<
