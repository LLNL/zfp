/*
** Copyright (c) 2014, Lawrence Livermore National Security, LLC.
** Produced at the Lawrence Livermore National Laboratory.
** Written by Peter Lindstrom.
** LLNL-CODE-663824.
** All rights reserved.
**
** This file is part of the zfp library.
** For details, see http://computation.llnl.gov/casc/zfp/.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
**
** 1. Redistributions of source code must retain the above copyright notice,
** this list of conditions and the disclaimer below.
**
** 2. Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the disclaimer (as noted below) in the
** documentation and/or other materials provided with the distribution.
**
** 3. Neither the name of the LLNS/LLNL nor the names of its contributors may
** be used to endorse or promote products derived from this software without
** specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
** ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
** LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
** INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
** (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
** LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
** ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
** THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**
**
** Additional BSD Notice
**
** 1. This notice is required to be provided under our contract with the U.S.
** Department of Energy (DOE).  This work was produced at Lawrence Livermore
** National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

** 2. Neither the United States Government nor Lawrence Livermore National
** Security, LLC nor any of their employees, makes any warranty, express or
** implied, or assumes any liability or responsibility for the accuracy,
** completeness, or usefulness of any information, apparatus, product, or
** process disclosed, or represents that its use would not infringe
** privately-owned rights.
**
** 3. Also, reference herein to any specific commercial products, process, or
** services by trade name, trademark, manufacturer or otherwise does not
** necessarily constitute or imply its endorsement, recommendation, or
** favoring by the United States Government or Lawrence Livermore National
** Security, LLC.  The views and opinions of authors expressed herein do not
** necessarily state or reflect those of the United States Government or
** Lawrence Livermore National Security, LLC, and shall not be used for
** advertising or product endorsement purposes.
*/

#ifndef ZFP_H
#define ZFP_H

#define ZFP_VERSION 0x0031 /* library version number: 0.3.1 */

#define ZFP_TYPE_FLOAT  1 /* single precision */
#define ZFP_TYPE_DOUBLE 2 /* double precision */


#ifdef __cplusplus
#include <cstddef>
extern "C" {
#else
#include <stddef.h>
#endif

typedef unsigned int uint;

/* array meta data and compression parameters */
typedef struct {
  uint type;    /* single (1) or double (2) precision */
  uint nx;      /* array x dimensions */
  uint ny;      /* array y dimensions (0 for 1D array) */
  uint nz;      /* array z dimensions (0 for 1D or 2D array) */
  uint minbits; /* minimum number of bits to store per block */
  uint maxbits; /* maximum number of bits to store per block */
  uint maxprec; /* maximum number of bitplanes to store */
  int minexp;   /* minimum bitplane number (soft error tolerance = 2^minexp) */
} zfp_params;

/* initialize parameters to default values */
void
zfp_init(
  zfp_params* params
);

/* set floating-point scalar type */
uint                  /* scalar type or zero on failure */
zfp_set_type(
  zfp_params* params, /* compression parameters */
  uint type           /* float (1) or double (2) */
);

/* set dimensions of 1D array */
void
zfp_set_size_1d(
  zfp_params* params, /* compression parameters */
  uint n              /* number of scalars */
);

/* set dimensions of 2D array */
void
zfp_set_size_2d(
  zfp_params* params, /* compression parameters */
  uint nx,            /* array x dimensions */
  uint ny             /* array y dimensions */
);

/* set dimensions of 3D array */
void
zfp_set_size_3d(
  zfp_params* params, /* compression parameters */
  uint nx,            /* array x dimensions */
  uint ny,            /* array y dimensions */
  uint nz             /* array z dimensions */
);

/* set fixed rate (params->{type,nx,ny,nz} must be set already) */
double                /* actual compression rate */
zfp_set_rate(
  zfp_params* params, /* compression parameters */
  double rate         /* desired compression rate in bits per value */
);

/* set fixed precision */
uint                  /* actual precision */
zfp_set_precision(
  zfp_params* params, /* compression parameters */
  uint precision      /* number of bits of precision (0 for full precision) */
);

/* set fixed accuracy */
double                /* actual error tolerance */
zfp_set_accuracy(
  zfp_params* params, /* compression parameters */
  double tolerance    /* absolute error tolerance */
);

/* safely estimate storage needed for compressed stream */
size_t                      /* max size of compressed stream (0 on failure) */
zfp_estimate_compressed_size(
  const zfp_params* params  /* compression parameters */
);

/* compress 1D, 2D, or 3D floating-point array */
size_t                      /* byte size of compressed stream (0 on failure) */
zfp_compress(
  const zfp_params* params, /* array meta data and compression parameters */
  const void* in,           /* uncompressed floating-point data */
  void* out,                /* compressed stream (must be large enough) */
  size_t outsize            /* bytes allocated for compressed stream */
);

/* decompress 1D, 2D, or 3D floating-point array */
int                         /* nonzero on success */
zfp_decompress(
  const zfp_params* params, /* array meta data and compression parameters */
  void* out,                /* decompressed floating-point data */
  const void* in,           /* compressed stream */
  size_t insize             /* bytes allocated for compressed stream */
);

#ifdef __cplusplus
}
#endif

#endif
