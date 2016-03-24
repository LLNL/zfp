CXX = g++
#DEFS = -DWITHOUT_COMPRESSION
CXXFLAGS = -O3 -ansi -Wall -I../inc $(DEFS)

all: zfp diffusion

zfp: zfp.cpp zfpcompress.h zfpcompress.o
	$(CXX) $(CXXFLAGS) zfp.cpp -o zfp zfpcompress.o

zfpcompress.o: zfpcompress.cpp
	$(CXX) $(CXXFLAGS) -c zfpcompress.cpp

diffusion: diffusion.cpp
	$(CXX) $(CXXFLAGS) diffusion.cpp -o diffusion

clean:
	rm -f zfp zfpcompress.o diffusion
