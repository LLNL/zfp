#include <math.h>
#include "rand32.h"

#define SEED 5

// POSIX rand48
#define MULTIPLIER 0x5deece66d
#define INCREMENT 0xb
#define MODULO ((uint64)1 << 48)
#define MASK_31 (0x7fffffffu)

static uint64 X;

void
resetRandGen()
{
  X = SEED;
}

// returns integer [0, 2^31 - 1]
uint32
nextUnsignedRand()
{
  X = (MULTIPLIER*X + INCREMENT) % MODULO;
  return (uint32)((X >> 16) & MASK_31);
}

// returns integer [-(2^30), 2^30 - 1]
int32
nextSignedRandInt()
{
  return (int32)nextUnsignedRand() - 0x40000000;
}

// returns float [-(2^11), 2^11 - 2^(-12)]
float
nextSignedRandFlPt()
{
  // 23 bit signed number
  uint32 uVal = (nextUnsignedRand() >> 7) & 0x00ffffff;
  int32 sVal = (int32)uVal - 0x800000;
  return ldexpf((float)sVal, -12);
}
