#include <math.h>
#include "rand64.h"

#define SEED 5

// https://nuclear.llnl.gov/CNP/rng/rngman/node4.html
#define MULTIPLIER 2862933555777941757
#define INCREMENT 3037000493

#define MAX_RAND_63 (0x7fffffffffffffffuLL)

static uint64 X;

void
resetRandGen()
{
  X = SEED;
}

// returns integer [0, 2^63 - 1]
uint64
nextUnsignedRand()
{
  // (mod 2^64)
  X = MULTIPLIER*X + INCREMENT;
  return (uint64)(X & MAX_RAND_63);
}

// returns integer [-(2^62), 2^62 - 1]
int64
nextSignedRandInt()
{
  uint64 uDisplace = (uint64)1 << 62;
  return (int64)nextUnsignedRand() - (int64)uDisplace;
}

// returns double [-(2^26), 2^26 - 2^(-26)]
double
nextSignedRandFlPt()
{
  // 52 bit signed number
  uint64 uVal = (nextUnsignedRand() >> 11) & 0x1fffffffffffff;
  int64 sVal = (int64)uVal - 0x10000000000000;
  return ldexp((double)sVal, -26);
}

// returns integer [0, 2^32 - 1]
uint32
nextRand32()
{
  return (uint32)(nextUnsignedRand() >> 31);
}
