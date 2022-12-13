# Makefile for automatically hipifying ../cuda to HIP source

# CUDA prerequisites
CUDA_HEADERS = $(wildcard ../cuda/*.cuh ../cuda/*.h)
CUDA_SOURCES = $(wildcard ../cuda/*.cu)

# corresponding HIP targets
HIP_HEADERS = $(subst ../cuda/,,$(CUDA_HEADERS:.cuh=.h))
HIP_SOURCES = $(subst ../cuda/,,$(CUDA_SOURCES:.cu=.cpp))

TARGETS = $(HIP_HEADERS) $(HIP_SOURCES) CMakeLists.txt

# source substitution rules
SUBSTITUTIONS = 's/_cuda/_hip/g;s/_CUDA/_HIP/g;s/\/cuda/\/hip/g;s/namespace cuda/namespace hip/g;s/::cuda/::hip/g;s/\.cuh/.h/g;s/\.cu/.cpp/g;s/_CUH/_H/g'

all: $(TARGETS)

%.h: ../cuda/%.cuh
	@echo "$< -> $@"
	@hipify-perl $< | sed $(SUBSTITUTIONS) > $@

%.cpp: ../cuda/%.cu
	@echo "$< -> $@"
	@hipify-perl $< | sed $(SUBSTITUTIONS) > $@

interface.h: ../cuda/interface.h
	@echo "$< -> $@"
	@hipify-perl $< | sed $(SUBSTITUTIONS) > $@

CMakeLists.txt: ../cuda/CMakeLists.txt
	@echo "$< -> $@"
	@hipify-perl $< | sed $(SUBSTITUTIONS) > $@

clean:
	rm -f $(TARGETS)
