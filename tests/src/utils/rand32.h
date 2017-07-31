#ifndef RAND_32_H
#define RAND_32_H

#include "include/zfp/types.h"

// reset seed
void
resetRandGen();

// returns integer [0, 2^31 - 1]
uint32
nextUnsignedRand();

// returns integer [-(2^30), 2^30 - 1]
int32
nextSignedRandInt();

// returns float [-(2^11), 2^11 - 2^(-12)]
float
nextSignedRandFlPt();

#endif
