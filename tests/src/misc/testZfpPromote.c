#include "zfp.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

static void
given_int8_when_promoteToInt32_expect_demoteToInt8Matches(void **state)
{
  uint dims = 3;
  uint sz = 1u << (2 * dims);
  int8* iblock8 = (int8*)malloc(sizeof(int8)*sz);
  int8* oblock8 = (int8*)calloc(sz, sizeof(int8));
  int32* block32 = (int32*)malloc(sizeof(int32)*sz);

  assert_non_null(iblock8);
  assert_non_null(oblock8);
  assert_non_null(block32);

  uint i;
  for (i = 0; i < sz; i++)
    iblock8[i] = (int8)i;

  zfp_promote_int8_to_int32(block32, iblock8, dims);
  zfp_demote_int32_to_int8(oblock8, block32, dims);

  for (i = 0; i < sz; i++)
    assert_int_equal(iblock8[i], oblock8[i]);
}

static void
given_uint8_when_promoteToInt32_expect_demoteToUInt8Matches(void **state)
{
  uint dims = 3;
  uint sz = 1u << (2 * dims);
  uint8* iblock8 = (uint8*)malloc(sizeof(uint8)*sz);
  uint8* oblock8 = (uint8*)calloc(sz, sizeof(uint8));
  int32* block32 = (int32*)malloc(sizeof(int32)*sz);

  assert_non_null(iblock8);
  assert_non_null(oblock8);
  assert_non_null(block32);

  uint i;
  for (i = 0; i < sz; i++)
    iblock8[i] = (uint8)i;

  zfp_promote_uint8_to_int32(block32, iblock8, dims);
  zfp_demote_int32_to_uint8(oblock8, block32, dims);

  for (i = 0; i < sz; i++)
    assert_int_equal(iblock8[i], oblock8[i]);
}

static void
given_int16_when_promoteToInt32_expect_demoteToInt16Matches(void **state)
{
  uint dims = 3;
  uint sz = 1u << (2 * dims);
  int16* iblock16 = (int16*)malloc(sizeof(int16)*sz);
  int16* oblock16 = (int16*)calloc(sz, sizeof(int16));
  int32* block32 = (int32*)malloc(sizeof(int32)*sz);

  assert_non_null(iblock16);
  assert_non_null(oblock16);
  assert_non_null(block32);

  uint i;
  for (i = 0; i < sz; i++)
    iblock16[i] = (int16)i;

  zfp_promote_int16_to_int32(block32, iblock16, dims);
  zfp_demote_int32_to_int16(oblock16, block32, dims);

  for (i = 0; i < sz; i++)
    assert_int_equal(iblock16[i], oblock16[i]);
}

static void
given_uint16_when_promoteToInt32_expect_demoteToUInt16Matches(void **state)
{
  uint dims = 3;
  uint sz = 1u << (2 * dims);
  uint16* iblock16 = (uint16*)malloc(sizeof(uint16)*sz);
  uint16* oblock16 = (uint16*)calloc(sz, sizeof(uint16));
  int32* block32 = (int32*)malloc(sizeof(int32)*sz);

  assert_non_null(iblock16);
  assert_non_null(oblock16);
  assert_non_null(block32);

  uint i;
  for (i = 0; i < sz; i++)
    iblock16[i] = (uint16)i;

  zfp_promote_uint16_to_int32(block32, iblock16, dims);
  zfp_demote_int32_to_uint16(oblock16, block32, dims);

  for (i = 0; i < sz; i++)
    assert_int_equal(iblock16[i], oblock16[i]);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(given_int8_when_promoteToInt32_expect_demoteToInt8Matches),
    cmocka_unit_test(given_uint8_when_promoteToInt32_expect_demoteToUInt8Matches),
    cmocka_unit_test(given_int16_when_promoteToInt32_expect_demoteToInt16Matches),
    cmocka_unit_test(given_uint16_when_promoteToInt32_expect_demoteToUInt16Matches),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
