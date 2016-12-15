testzfp: testzfp.cpp fields.o ../lib/libzfp.a
 $(CXX) $(CXXFLAGS) -I../array testzfp.cpp fields.o -L../lib -lzfp -o testzfp

fields.o: fields.c
 $(CC) $(CFLAGS) -c fields.c

test: testzfp
 ./testzfp
