#include <stddef.h>

#include "zfp/types.h"
#include "fortranMacros.h"

typedef struct zFORp_structs zFORp_structs;

void*
_prefixFOR(zFORp_bitstream_stream_open)(void** buffer, size_t* bufferSizeBytes);

void
_prefixFOR(zFORp_bitstream_stream_close)(zFORp_structs* container);

void*
_prefixFOR(zforp_stream_open)(void** bs);

void
_prefixFOR(zforp_stream_close)(zFORp_structs* container);

int
_prefixFOR(zforp_stream_compression_mode)(zFORp_structs* container);

uint64
_prefixFOR(zforp_stream_mode)(zFORp_structs* container);

double
_prefixFOR(zforp_stream_set_rate)(zFORp_structs* container, double* rate, uint* type, uint* dims, int* wra);

uint
_prefixFOR(zforp_stream_set_precision)(zFORp_structs* container, uint* prec);

double
_prefixFOR(zforp_stream_set_accuracy)(zFORp_structs* container, double* tolerance);

uint
_prefixFOR(zforp_stream_set_mode)(zFORp_structs* container, uint64* mode);

void
_prefixFOR(zforp_stream_params)(zFORp_structs* container, uint** minbits, uint** maxbits, uint**maxprec, int** minexp);

int
_prefixFOR(zforp_stream_set_params)(zFORp_structs* container, uint* minbits, uint* maxbits, uint* maxprec, int* minexp);
