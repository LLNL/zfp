include ../Config

LIBDIR = ../lib
TARGETS = $(LIBDIR)/libzfp.a $(LIBDIR)/libzfp.so
OBJECTS = bitstream.o decode1i.o decode1l.o decode1f.o decode1d.o encode1i.o encode1l.o encode1f.o encode1d.o decode2i.o decode2l.o decode2f.o decode2d.o encode2i.o encode2l.o encode2f.o encode2d.o decode3i.o decode3l.o decode3f.o decode3d.o encode3i.o encode3l.o encode3f.o encode3d.o decode4i.o decode4l.o decode4f.o decode4d.o encode4i.o encode4l.o encode4f.o encode4d.o zfp.o

static: $(LIBDIR)/libzfp.a

shared: $(LIBDIR)/libzfp.so

clean:
	rm -f $(TARGETS) $(OBJECTS)

$(LIBDIR)/libzfp.a: $(OBJECTS)
	mkdir -p $(LIBDIR)
	rm -f $@
	ar rc $@ $^

$(LIBDIR)/libzfp.so: $(OBJECTS)
	mkdir -p $(LIBDIR)
	$(CC) $(CFLAGS) -shared $^ -o $@

.c.o:
	$(CC) $(CFLAGS) -c $<
