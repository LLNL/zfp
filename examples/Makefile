include ../Config

TARGETS = diffusion pgm speed testzfp zfp

all: $(TARGETS)

diffusion: diffusion.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -I../array diffusion.cpp -L../lib -lzfp -o diffusion

pgm: pgm.c ../lib/libzfp.a
	$(CC) $(CFLAGS) pgm.c -L../lib -lzfp -lm -o pgm

speed: speed.c ../lib/libzfp.a
	$(CC) $(CFLAGS) speed.c -L../lib -lzfp -lm -o speed

testzfp: testzfp.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -I../array testzfp.cpp -L../lib -lzfp -o testzfp
	

zfp: zfp.c ../lib/libzfp.a
	$(CC) $(CFLAGS) zfp.c -L../lib -lzfp -lm -o zfp

test: testzfp
	./testzfp

clean:
	rm -f $(TARGETS)
