# run diffusion without compression
# DEFS += -DWITHOUT_COMPRESSION

# use fast transform with only shifts and adds
# DEFS += -DZFP_FAST_TRANSFORM

# enable fixed-point range checking
# DEFS += -DFIXPT_RANGE_CHECK

# signed right shift >> is not arithmetic shift
# DEFS += -DFIXPT_NO_ARITHMETIC_SHIFT

# use aligned memory allocation
# DEFS += -DALIGNED_ALLOC

# use two-way skew-associative cache
# DEFS += -DCACHE_TWOWAY

# use faster but more collision prone hash function
# DEFS += -DCACHE_HASH

# count cache misses
# DEFS += -DCACHE_PROFILE

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
	$(CC) $(CFLAGS) -L../lib zfp.c -lzfp -lm -lstdc++ -o zfp

clean:
	rm -f $(TARGETS)
