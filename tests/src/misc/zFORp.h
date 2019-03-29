#include <stddef.h>

#include "zfp.h"

bitstream*
zforp_bitstream_stream_open(void** buffer, size_t* bufferSizeBytes);

void
zforp_bitstream_stream_close(bitstream** stream);

zfp_stream*
zforp_stream_open(void** bs);

void
zforp_stream_close(zfp_stream** stream);

zfp_mode
zforp_stream_compression_mode(zfp_stream** stream);

uint64
zforp_stream_mode(zfp_stream** stream);

double
zforp_stream_set_rate(zfp_stream** stream, double* rate, zfp_type* type, uint* dims, int* wra);

uint
zforp_stream_set_precision(zfp_stream** stream, uint* prec);

double
zforp_stream_set_accuracy(zfp_stream** stream, double* tolerance);

uint
zforp_stream_set_mode(zfp_stream** stream, uint64* mode);

void
zforp_stream_params(zfp_stream** stream, uint** minbits, uint** maxbits, uint**maxprec, int** minexp);

int
zforp_stream_set_params(zfp_stream** stream, uint* minbits, uint* maxbits, uint* maxprec, int* minexp);
