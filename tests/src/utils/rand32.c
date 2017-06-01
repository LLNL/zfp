#define SEED 5

// POSIX rand48
#define MULTIPLIER 0x5deece66d
#define INCREMENT 0xb
#define MODULO ((uint64)1 << 48)
#define MASK_31 (0x7fffffffu)

static uint64 X;

static void
resetRandGen()
{
  X = SEED;
}

// returns integer [0, 2^31 - 1]
static uint32
nextUnsignedRand()
{
  X = (MULTIPLIER*X + INCREMENT) % MODULO;
  return (uint32)((X >> 16) & MASK_31);
}

// returns integer [-(2^30), 2^30 - 1]
static int32
nextSignedRand()
{
  return (int32)nextUnsignedRand() - 0x40000000;
}
