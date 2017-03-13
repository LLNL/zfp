include ../Config

TARGETS = diffusion-raw diffusion-zfp pgm simple speed

all: $(TARGETS)

diffusion-raw: diffusion.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -DWITHOUT_COMPRESSION -I../array diffusion.cpp ../lib/libzfp.a -o diffusion-raw

diffusion-zfp: diffusion.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -I../array diffusion.cpp ../lib/libzfp.a -o diffusion-zfp

pgm: pgm.c ../lib/libzfp.a
	$(CC) $(CFLAGS) pgm.c ../lib/libzfp.a -lm -o pgm

simple: simple.c ../lib/libzfp.a
	$(CC) $(CFLAGS) simple.c ../lib/libzfp.a -lm -o simple

speed: speed.c ../lib/libzfp.a
	$(CC) $(CFLAGS) speed.c ../lib/libzfp.a -lm -o speed

clean:
	rm -f $(TARGETS) fields.o
