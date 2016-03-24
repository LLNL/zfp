include ../Config

CXX = g++
CC = gcc
CXXFLAGS = -g -O3 -ansi -Wall -I../inc $(DEFS)
CFLAGS = $(CXXFLAGS) -pedantic

TARGETS = diffusion zfp

all: $(TARGETS)

diffusion: diffusion.cpp
	$(CXX) $(CXXFLAGS) diffusion.cpp -o diffusion

# C++ mode
# zfp: zfp.cpp ../lib/libzfp.a
#	$(CXX) $(CXXFLAGS) -L../lib zfp.cpp -lzfp -o zfp

# C89 mode
zfp: zfp.c ../lib/libzfp.a
	$(CC) $(CFLAGS) -c zfp.c
	$(CXX) $(CXXFLAGS) zfp.o -L../lib -lzfp -lm -o zfp

clean:
	rm -f $(TARGETS) zfp.o
