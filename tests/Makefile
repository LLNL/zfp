all: testzfp

testzfp: testzfp.cpp fields.o ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -I../array -I ../inc testzfp.cpp fields.o ../lib/libzfp.a -o testzfp

fields.o: fields.c
	$(CC) $(CFLAGS) -c fields.c

test: testzfp
	./testzfp

clean:
	rm -f *.o testzfp 
