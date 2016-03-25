include ../Config

TARGETS = diffusion testzfp zfp

all: $(TARGETS)

diffusion: diffusion.cpp
	$(CXX) $(CXXFLAGS) diffusion.cpp -o diffusion

testzfp: testzfp.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) testzfp.cpp -L../lib -lzfp -o testzfp

zfp: zfp.o ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) zfp.o -L../lib -lzfp -lm -o zfp

# C++ mode
#zfp.o: zfp.cpp
#	$(CXX) $(CXXFLAGS) -c zfp.cpp

# C89 mode
zfp.o: zfp.c
	$(CC) $(CFLAGS) -c zfp.c

test: testzfp
	./testzfp

clean:
	rm -f $(TARGETS) zfp.o
