struct setupVars {
  zfp_field* field;
  SCALAR* data;
};

static int
setupBasic(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

#if DIMS == 1
  zfp_field* field = zfp_field_1d(NULL, ZFP_TYPE, NX);
#elif DIMS == 2
  zfp_field* field = zfp_field_2d(NULL, ZFP_TYPE, NX, NY);
#elif DIMS == 3
  zfp_field* field = zfp_field_3d(NULL, ZFP_TYPE, NX, NY, NZ);
#elif DIMS == 4
  zfp_field* field = zfp_field_4d(NULL, ZFP_TYPE, NX, NY, NZ, NW);
#endif

  bundle->field = field;
  bundle->data = NULL;

  *state = bundle;

  return 0;
}

static int
setupContiguous(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

#if DIMS == 1
  zfp_field* field = zfp_field_1d(NULL, ZFP_TYPE, NX);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR)*NX);
#elif DIMS == 2
  zfp_field* field = zfp_field_2d(NULL, ZFP_TYPE, NX, NY);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR)*NX*NY);
#elif DIMS == 3
  zfp_field* field = zfp_field_3d(NULL, ZFP_TYPE, NX, NY, NZ);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR)*NX*NY*NZ);
#elif DIMS == 4
  zfp_field* field = zfp_field_4d(NULL, ZFP_TYPE, NX, NY, NZ, NW);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR)*NX*NY*NZ*NW);
#endif
  assert_non_null(data);

  zfp_field_set_pointer(field, data);
  bundle->field = field;
  bundle->data = data;

  *state = bundle;

  return 0;
}

static int
setupStrided(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

#if DIMS == 1
  zfp_field* field = zfp_field_1d(NULL, ZFP_TYPE, NX);
  zfp_field_set_stride_1d(field, SX);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR) * ((SX*(NX-1)) + 1));
#elif DIMS == 2
  zfp_field* field = zfp_field_2d(NULL, ZFP_TYPE, NX, NY);
  zfp_field_set_stride_2d(field, SX, SY);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR) * ((SX*(NX-1)) + (SY*(NY-1)) + 1));
#elif DIMS == 3
  zfp_field* field = zfp_field_3d(NULL, ZFP_TYPE, NX, NY, NZ);
  zfp_field_set_stride_3d(field, SX, SY, SZ);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR) * ((SX*(NX-1)) + (SY*(NY-1)) + (SZ*(NZ-1)) + 1));
#elif DIMS == 4
  zfp_field* field = zfp_field_4d(NULL, ZFP_TYPE, NX, NY, NZ, NW);
  zfp_field_set_stride_4d(field, SX, SY, SZ, SW);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR) * ((SX*(NX-1)) + (SY*(NY-1)) + (SZ*(NZ-1)) + (SW*(NW-1)) + 1));
#endif
  assert_non_null(data);

  zfp_field_set_pointer(field, data);
  bundle->field = field;
  bundle->data = data;

  *state = bundle;

  return 0;
}

static int
setupNegativeStrided(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

#if DIMS == 1
  zfp_field* field = zfp_field_1d(NULL, ZFP_TYPE, NX);
  zfp_field_set_stride_1d(field, -SX);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR) * ((SX*(NX-1)) + 1));
#elif DIMS == 2
  zfp_field* field = zfp_field_2d(NULL, ZFP_TYPE, NX, NY);
  zfp_field_set_stride_2d(field, -SX, -SY);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR) * ((SX*(NX-1)) + (SY*(NY-1)) + 1));
#elif DIMS == 3
  zfp_field* field = zfp_field_3d(NULL, ZFP_TYPE, NX, NY, NZ);
  zfp_field_set_stride_3d(field, -SX, -SY, -SZ);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR) * ((SX*(NX-1)) + (SY*(NY-1)) + (SZ*(NZ-1)) + 1));
#elif DIMS == 4
  zfp_field* field = zfp_field_4d(NULL, ZFP_TYPE, NX, NY, NZ, NW);
  zfp_field_set_stride_4d(field, -SX, -SY, -SZ, -SW);
  SCALAR* data = (SCALAR*)malloc(sizeof(SCALAR) * ((SX*(NX-1)) + (SY*(NY-1)) + (SZ*(NZ-1)) + (SW*(NW-1)) + 1));
#endif
  assert_non_null(data);

  zfp_field_set_pointer(field, data);
  bundle->field = field;
  bundle->data = data;

  *state = bundle;

  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *bundle = *state;

  zfp_field_free(bundle->field);

  if (bundle->data != NULL)
    free(bundle->data);

  free(bundle);

  return 0;
}

static void
given_contiguousData_isContiguousReturnsTrue(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  assert_true(zfp_field_is_contiguous(field));
}

static void
given_noncontiguousData_isContiguousReturnsFalse(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  assert_false(zfp_field_is_contiguous(field));
}

static void
when_noFieldData_fieldBeginReturnsNull(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  assert_null(zfp_field_begin(field));
}

static void
when_contiguousData_fieldBeginsAtDataPointer(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  assert_true(zfp_field_begin(field) == zfp_field_pointer(field));
}

static void
when_noncontiguousDataWithNegativeStride_fieldBeginsAtCorrectLocation(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

#if DIMS == 1
  ptrdiff_t min = ((int)-SX * (ptrdiff_t)(NX - 1));
#elif DIMS == 2
  ptrdiff_t min = ((int)-SX * (ptrdiff_t)(NX - 1)) + ((int)-SY * (ptrdiff_t)(NY - 1));
#elif DIMS == 3
  ptrdiff_t min = ((int)-SX * (ptrdiff_t)(NX - 1)) + ((int)-SY * (ptrdiff_t)(NY - 1)) + ((int)-SZ * (ptrdiff_t)(NZ - 1));
#elif DIMS == 4
  ptrdiff_t min = ((int)-SX * (ptrdiff_t)(NX - 1)) + ((int)-SY * (ptrdiff_t)(NY - 1)) + ((int)-SZ * (ptrdiff_t)(NZ - 1)) + ((int)-SW * (ptrdiff_t)(NW - 1));
#endif
  void* begin = (void*)((uchar*)field->data + min * (ptrdiff_t)zfp_type_size(field->type));
  assert_true(zfp_field_begin(field) == begin);
}

static void
given_field_precisionCorrect(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  assert_true(zfp_field_precision(field) == sizeof(SCALAR) * CHAR_BIT);
}

static void
given_contiguousData_fieldSizeBytesCorrect(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

#if DIMS == 1
  assert_true(zfp_field_size_bytes(field) == NX * sizeof(SCALAR));
#elif DIMS == 2
  assert_true(zfp_field_size_bytes(field) == NX * NY * sizeof(SCALAR));
#elif DIMS == 3
  assert_true(zfp_field_size_bytes(field) == NX * NY * NZ * sizeof(SCALAR));
#elif DIMS == 4
  assert_true(zfp_field_size_bytes(field) == NX * NY * NZ * NW * sizeof(SCALAR));
#endif
}

static void
given_noncontiguousData_fieldSizeBytesCorrect(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

#if DIMS == 1
  assert_true(zfp_field_size_bytes(field) == ((SX*(NX-1) + 1) * sizeof(SCALAR)));
#elif DIMS == 2
  assert_true(zfp_field_size_bytes(field) == ((SX*(NX-1) + SY*(NY-1) + 1) * sizeof(SCALAR)));
#elif DIMS == 3
  assert_true(zfp_field_size_bytes(field) == ((SX*(NX-1) + SY*(NY-1) + SZ*(NZ-1) + 1) * sizeof(SCALAR)));
#elif DIMS == 4
  assert_true(zfp_field_size_bytes(field) == ((SX*(NX-1) + SY*(NY-1) + SZ*(NZ-1) + SW*(NW-1) + 1) * sizeof(SCALAR)));
#endif
}



int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(given_contiguousData_isContiguousReturnsTrue, setupContiguous, teardown),
    cmocka_unit_test_setup_teardown(given_noncontiguousData_isContiguousReturnsFalse, setupStrided, teardown),
    cmocka_unit_test_setup_teardown(when_noFieldData_fieldBeginReturnsNull, setupBasic, teardown),
    cmocka_unit_test_setup_teardown(when_contiguousData_fieldBeginsAtDataPointer, setupContiguous, teardown),
    cmocka_unit_test_setup_teardown(when_noncontiguousDataWithNegativeStride_fieldBeginsAtCorrectLocation, setupNegativeStrided, teardown),
    cmocka_unit_test_setup_teardown(given_field_precisionCorrect, setupBasic, teardown),
    cmocka_unit_test_setup_teardown(given_contiguousData_fieldSizeBytesCorrect, setupContiguous, teardown),
    cmocka_unit_test_setup_teardown(given_noncontiguousData_fieldSizeBytesCorrect, setupStrided, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
