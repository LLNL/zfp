include ../Config

LIBDIR = ../lib
MODDIR = ../modules
TARGETS = $(LIBDIR)/libzFORp.a $(LIBDIR)/libzFORp.so $(MODDIR)/zforp_module.mod
OBJECTS = zfp.o
MODULES = zforp_module.mod

static: $(LIBDIR)/libzFORp.a $(MODDIR)/zforp_module.mod

shared: $(LIBDIR)/libzFORp.so $(MODDIR)/zforp_module.mod

clean:
	rm -f $(TARGETS) $(OBJECTS)

$(LIBDIR)/libzFORp.a: $(OBJECTS)
	mkdir -p $(LIBDIR)
	rm -f $@
	ar rc $@ $^

$(LIBDIR)/libzFORp.so: $(OBJECTS)
	mkdir -p $(LIBDIR)
	$(FC) $(FFLAGS) -shared $^ -o $@

$(MODDIR)/zforp_module.mod: $(OBJECTS)
	mkdir -p $(MODDIR)
	mv zforp_module.mod $(MODDIR)

.f.o:
	$(FC) $(FFLAGS) -c $<
