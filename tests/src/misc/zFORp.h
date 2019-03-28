#include <stddef.h>

#include "zfp/types.h"

typedef struct zFORp_structs zFORp_structs;

void*
zforp_bitstream_stream_open(void** buffer, size_t* bufferSizeBytes);

void
zforp_bitstream_stream_close(zFORp_structs* container);

void*
zforp_stream_open(void** bs);

void
zforp_stream_close(zFORp_structs* container);

int
zforp_stream_compression_mode(zFORp_structs* container);

uint64
zforp_stream_mode(zFORp_structs* container);

double
zforp_stream_set_rate(zFORp_structs* container, double* rate, uint* type, uint* dims, int* wra);

uint
zforp_stream_set_precision(zFORp_structs* container, uint* prec);

double
zforp_stream_set_accuracy(zFORp_structs* container, double* tolerance);

uint
zforp_stream_set_mode(zFORp_structs* container, uint64* mode);

void
zforp_stream_params(zFORp_structs* container, uint** minbits, uint** maxbits, uint**maxprec, int** minexp);

int
zforp_stream_set_params(zFORp_structs* container, uint* minbits, uint* maxbits, uint* maxprec, int* minexp);
