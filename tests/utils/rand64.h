#ifndef RAND_64_H
#define RAND_64_H

#include "include/zfp/types.h"

// reset seed
void
resetRandGen();

// returns integer [0, 2^63 - 1]
uint64
nextUnsignedRand();

// returns integer [-(2^62), 2^62 - 1]
int64
nextSignedRandInt();

// returns double [-(2^26), 2^26 - 2^(-26)]
double
nextSignedRandFlPt();

// returns integer [0, 2^32 - 1]
uint32
nextRand32();

#endif
