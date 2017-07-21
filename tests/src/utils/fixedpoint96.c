#include <limits.h>
#include "fixedpoint96.h"

void
initFixedPt(int64 i, uint32 f, fixedPt* result)
{
  result->i = i;
  result->f = f;
}

// logical shift
static void
shiftRightSigned(int64 input, uint shiftAmount, int64* result)
{
  if (input < 0) {
    *result = ~(~input >> shiftAmount);
  } else {
    *result = input >> shiftAmount;
  }
}

// split 64 bit unsigned into two 32 bit unsigned parts
// both parts live in the lowest 32 bits of the uint64
static void
splitUnsigned(uint64 input, uint64* upper, uint64* lower)
{
  *upper = input >> 32;
  *lower = input - (*upper << 32);
}

// split 64 bit signed into two 32 bit parts
// both parts live in the lowest 32 bits of the 64 bit int
// upper keeps the sign
// lower is unsigned
static void
splitSigned(int64 input, int64* upper, uint64* lower)
{
  shiftRightSigned(input, 32, upper);
  *lower = (uint64)(input - (*upper << 32));
}

static void
addFractional(uint32 a, uint32 b, uint32* result, uint32* carry)
{
  uint64 a64 = (uint64)a;
  uint64 b64 = (uint64)b;

  uint64 carry64, result64;
  splitUnsigned(a64 + b64, &carry64, &result64);

  // carry is 0 or 1
  *carry = (uint32)carry64;
  *result = (uint32)result64;
}

// returns 1 if sum overflows
static int
addSignedIntegers(int64 a, int64 b, int64* result)
{
  if (b >= 0 && a > LLONG_MAX - b) {
    return 1;
  } else if (b < 0 && a < LLONG_MIN - b) {
    return 1;
  }

  *result = a + b;

  return 0;
}

int
roundFixedPt(fixedPt* fp, int64* result)
{
  return addSignedIntegers(fp->i, (int64)(fp->f >= 0x80000000), result);
}

// returns 0 if successful, 1 otherwise
int
add(fixedPt* a, fixedPt* b, fixedPt* result)
{
  uint32 carry;
  addFractional(a->f, b->f, &result->f, &carry);

  // detect overflow while trying each combination: 3 terms, 2 operations
  int64 val;

  // (a + carry) + b
  if (addSignedIntegers(a->i, (int64)carry, &val) == 0) {
    if (addSignedIntegers(val, b->i, &result->i) == 0) {
      return 0;
    }
  }

  // a + (carry + b)
  if (addSignedIntegers((int64)carry, b->i, &val) == 0) {
    if (addSignedIntegers(a->i, val, &result->i) == 0) {
      return 0;
    }
  }

  // (a + b) + carry
  if (addSignedIntegers(a->i, b->i, &val) == 0) {
    if (addSignedIntegers(val, (int64)carry, &result->i) == 0) {
      return 0;
    }
  }

  // unavoidable overflow
  return 1;
}

// always successful
// subtract borrow from a's next MSB [integer] part
static void
subtractFractional(uint32 a, uint32 b, uint32* result, int64* borrow)
{
  *result = a - b;

  *borrow = (a < b) ? 1 : 0;
}

// returns 1 if subtraction goes out of range
static int
subtractSignedIntegers(int64 a, int64 b, int64* result)
{
  if (b < 0 && a > LLONG_MAX + b) {
    return 1;
  } else if (b >= 0 && a < LLONG_MIN + b) {
    return 1;
  }

  *result = a - b;

  return 0;
}

// returns 1 if result would go out of range
int
subtract(fixedPt* a, fixedPt* b, fixedPt* result)
{
  int64 borrow;
  subtractFractional(a->f, b->f, &result->f, &borrow);

  // detect overflow while trying each combination: 3 terms, 2 operations
  int64 val;

  // (a - borrow) - b
  if (subtractSignedIntegers(a->i, borrow, &val) == 0) {
    if (subtractSignedIntegers(val, b->i, &result->i) == 0) {
      return 0;
    }
  }

  // a - (borrow + b)
  if (addSignedIntegers(borrow, b->i, &val) == 0) {
    if (subtractSignedIntegers(a->i, val, &result->i) == 0) {
      return 0;
    }
  }

  // (a - b) - borrow
  if (subtractSignedIntegers(a->i, b->i, &val) == 0) {
    if (subtractSignedIntegers(val, borrow, &result->i) == 0) {
      return 0;
    }
  }

  // unavoidable overflow
  return 1;
}

// returns 1 if integer part overflows
// fractional part is truncated
int
multiply(fixedPt* a, fixedPt* b, fixedPt* result)
{
  // split everything into 32 bit values, stored in 64 bit types
  // that way, multiplying 2 32 bit values will fit in 64 bits
  // also, uint64 to int64 casts will be safe
  uint64 af, bf, rf;
  af = (uint64)a->f;
  bf = (uint64)b->f;

  uint64 ai0, bi0, ri0;
  int64 ai1, bi1, ri1;
  splitSigned(a->i, &ai1, &ai0);
  splitSigned(b->i, &bi1, &bi0);

  // actual values:
  //   a = (2^32)*ai1 + ai0 + (2^-32)*af
  //   b = (2^32)*bi1 + bi0 + (2^-32)*bf
  //
  //   r = a*b =
  // A            (2^64) * ai1*bi1
  // B          + (2^32) * (ai1*bi0 + ai0*bi1)
  // C          + (ai0*bi0 + ai1*bf + af*bi1)
  // D          + (2^-32) * (ai0*bf + af*bi0)
  // E          + (2^-64) * af*bf
  //
  //
  //        (MSB)                         (LSB)
  //                    -----fixedPt-----
  //   a*b= _____|_____|_ri1_|_ri0_|_rf__|_____
  //        -----A-----
  //              -----B-----
  //                    -----C-----
  //                          -----D-----
  //                                -----E-----
  //  perform sum from LSB to MSB
  //    - store 32 bit result
  //    - carry overflow to next, more significant 32 bit chunk

  // naming
  // highA : 32 MSB of A stored in 32 LSB of highA
  // lowA : 32 LSB of A stored in 32 LSB of lowA

  // (2^-64) * E
  uint64 E = af * bf;
  uint64 highE = E >> 32;
  // omit lowE (truncated)

  // D = (2^-32) * (D1 + D2)
  uint64 highD1, lowD1, highD2, lowD2;
  splitUnsigned(ai0 * bf, &highD1, &lowD1);
  splitUnsigned(af * bi0, &highD2, &lowD2);

  // highD -> result LSB integer part
  // lowD -> result fractional part
  uint64 highD, uCarryD;
  splitUnsigned(lowD1 + lowD2 + highE, &uCarryD, &rf);
  splitUnsigned(highD1 + highD2 + uCarryD, &uCarryD, &highD);

  // C = C1 + C2 + C3
  uint64 highC1;
  int64 highC2, highC3;
  uint64 lowC1, lowC2, lowC3;
  // C1 is unsigned (uint32 * uint32 only fits in uint64)
  // uint32 * int32 fits in int64
  splitUnsigned(ai0 * bi0, &highC1, &lowC1);
  splitSigned(ai1 * (int64)bf, &highC2, &lowC2);
  splitSigned((int64)af * bi1, &highC3, &lowC3);

  // highC -> MSB integer part
  // lowC -> LSB integer part
  int64 sCarryC;
  uint64 highC;
  splitSigned((int64)lowC1 + (int64)lowC2 + (int64)lowC3 + (int64)highD, &sCarryC, &ri0);
  splitSigned((int64)highC1 + highC2 + highC3 + (int64)uCarryD + sCarryC, &sCarryC, &highC);

  // B = (2^32) * (B1 + B2)
  int64 highB1, highB2;
  uint64 lowB1, lowB2;
  splitSigned(ai1 * (int64)bi0, &highB1, &lowB1);
  splitSigned((int64)ai0 * bi1, &highB2, &lowB2);

  // lowB -> MSB integer part
  // highB -> more significant 32 bits than we can hold in fixedPt
  uint64 ri1Unsigned;
  int64 sCarryB;
  splitSigned((int64)lowB1 + (int64)lowB2 + (int64)highC, &sCarryB, &ri1Unsigned);
  int64 highB = highB1 + highB2 + sCarryC + sCarryB;

  // MSB of overall product keeps sign
  int64 A = ai1 * bi1 + highB;

  // check ri1Unsigned with A's sign is in range of int64
  ri1Unsigned <<= 32;
  uint64 leftmostBitSetVal = (uint64)1 << 63;
  if (A == -1) {
    // result < 0

    if (ri1Unsigned <= LLONG_MAX) {
      return 1;
    }

    // cast ri1Unsigned safely (set MSB to 1)
    ri1 = (int64)(ri1Unsigned - leftmostBitSetVal) - leftmostBitSetVal;

  } else if (A == 0){
    // result >= 0

    if (ri1Unsigned > LLONG_MAX) {
      return 1;
    }

    ri1 = (int64)ri1Unsigned;

  } else {
    return 1;
  }

  result->f = rf;
  result->i = ri1 + ri0;

  return 0;
}
