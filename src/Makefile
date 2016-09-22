include ../Config

LIB = ../lib
TARGETS = $(LIB)/libzfp.a $(LIB)/libzfp.so
OBJECTS = bitstream.o decode1f.o decode1d.o encode1f.o encode1d.o decode2f.o decode2d.o encode2f.o encode2d.o decode3f.o decode3d.o encode3f.o encode3d.o zfp.o

$(LIB)/libzfp.a: $(OBJECTS)
	mkdir -p $(LIB)
	rm -f $@
	ar rc $@ $^

$(LIB)/libzfp.so: $(OBJECTS)
	$(CC) $(CFLAGS) -shared $^ -o $@

.c.o:
	$(CC) $(CFLAGS) -c $<

static: $(LIB)/libzfp.a

shared: $(LIB)/libzfp.so

clean:
	rm -f $(TARGETS) $(OBJECTS)
	[ -d $(LIB) ] && rmdir $(LIB) || true
