#include "zfp.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>

static void
compress(double* array, int nx, int ny, int nz)
{
    zfp_type        type;
    zfp_field*      field;
    zfp_stream*     stream;
    zfp_index*      index;
    bitstream*      bstream;
    void*           buffer;
    size_t          buffSz;
    size_t          compressSz;
    size_t          decompressSz;
    zfp_bool        execStatus;
    zfp_index_type  index_type;
    uint            granularity;
    zfp_exec_policy policy;

    printf("compressing %dx%dx%d array (size: %d bytes)\n", nx, ny, nz, nx*ny*nz*sizeof(double));

    /* execution policy */
    policy = zfp_exec_serial;
    /*policy = zfp_exec_omp;*/
    /*policy = zfp_exec_cuda;*/
    /*policy = zfp_exec_hip;*/

    /* setup field */
    type    = zfp_type_double;
    field   = zfp_field_3d(array, type, nx, ny, nz);

    /* initialize stream */
    stream  = zfp_stream_open(NULL);
    zfp_stream_set_accuracy(stream, 1e-3);

    /* setup index */
    index_type  = zfp_index_offset;
    granularity = 1;
    index       = zfp_index_create();
    zfp_index_set_type(index, index_type, granularity);
    zfp_stream_set_index(stream, index);

    /* setup bitstream */
    buffSz  = zfp_stream_maximum_size(stream, field);
    buffer  = malloc(buffSz);
    bstream = stream_open(buffer, buffSz);
    zfp_stream_set_bit_stream(stream, bstream);

    /* compress */
    if (policy == zfp_exec_cuda || policy == zfp_exec_hip)
        execStatus = zfp_stream_set_execution(stream, zfp_exec_serial);
    else
        execStatus = zfp_stream_set_execution(stream, policy);
    compressSz = zfp_compress(stream, field);
    zfp_stream_rewind(stream);

    printf("compressed size: %lu bytes\n", compressSz);

    /* decompress */
    if (policy == zfp_exec_omp)
        execStatus = zfp_stream_set_execution(stream, zfp_exec_serial);
    else
        execStatus = zfp_stream_set_execution(stream, policy);
    decompressSz = zfp_decompress(stream, field);

    printf("decompression %s\n", compressSz == decompressSz ? "succeeded" : "failed");

    /* cleanup */
    zfp_field_free(field);
    zfp_stream_close(stream);
    zfp_index_free(index);
    stream_close(bstream);
    free(buffer);
}

int main()
{
    int nx = 100;
    int ny = 100;
    int nz = 100;
    double* array = (double*)malloc(nx * ny * nz * sizeof(double));

    int i, j, k;
    for (k = 0; k < nz; k++)
    {
        for (j = 0; j < ny; j++)
        {
            for (i = 0; i < nx; i++)
            {
                double x = 2.0 * i / nx;
                double y = 2.0 * j / ny;
                double z = 2.0 * k / nz;
                array[i + nx * (j + ny * k)] = exp(-(x * x + y * y + z * z));
            }
        }
    }

    compress(array, nx, ny, nz); 

    free(array);
    return EXIT_SUCCESS;
}
