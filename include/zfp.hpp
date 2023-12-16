#ifndef ZFP_HPP
#define ZFP_HPP

// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC and
// other zfp project contributors. See the top-level LICENSE file for details.
// SPDX-License-Identifier: BSD-3-Clause

#include "zfp.h"

// templated C++ wrappers around libzfp low-level C functions
namespace zfp {

// encoder declarations -------------------------------------------------------

template <typename Scalar, uint dims>
inline size_t
encode_block(zfp_stream* zfp, const Scalar* block);

template <typename Scalar>
inline size_t
encode_block_strided(zfp_stream* zfp, const Scalar* p, ptrdiff_t sx);

template <typename Scalar>
inline size_t
encode_block_strided(zfp_stream* zfp, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy);

template <typename Scalar>
inline size_t
encode_block_strided(zfp_stream* zfp, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz);

template <typename Scalar>
inline size_t
encode_block_strided(zfp_stream* zfp, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);

template <typename Scalar>
inline size_t
encode_partial_block_strided(zfp_stream* zfp, const Scalar* p, size_t nx, ptrdiff_t sx);

template <typename Scalar>
inline size_t
encode_partial_block_strided(zfp_stream* zfp, const Scalar* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy);

template <typename Scalar>
inline size_t
encode_partial_block_strided(zfp_stream* zfp, const Scalar* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz);

template <typename Scalar>
inline size_t
encode_partial_block_strided(zfp_stream* zfp, const Scalar* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);

// encoder specializations ----------------------------------------------------

template<>
inline size_t
encode_block<float, 1>(zfp_stream* zfp, const float* block) { return zfp_encode_block_float_1(zfp, block); }

template<>
inline size_t
encode_block<float, 2>(zfp_stream* zfp, const float* block) { return zfp_encode_block_float_2(zfp, block); }

template<>
inline size_t
encode_block<float, 3>(zfp_stream* zfp, const float* block) { return zfp_encode_block_float_3(zfp, block); }

template<>
inline size_t
encode_block<float, 4>(zfp_stream* zfp, const float* block) { return zfp_encode_block_float_4(zfp, block); }

template<>
inline size_t
encode_block<double, 1>(zfp_stream* zfp, const double* block) { return zfp_encode_block_double_1(zfp, block); }

template<>
inline size_t
encode_block<double, 2>(zfp_stream* zfp, const double* block) { return zfp_encode_block_double_2(zfp, block); }

template<>
inline size_t
encode_block<double, 3>(zfp_stream* zfp, const double* block) { return zfp_encode_block_double_3(zfp, block); }

template<>
inline size_t
encode_block<double, 4>(zfp_stream* zfp, const double* block) { return zfp_encode_block_double_4(zfp, block); }

template <>
inline size_t
encode_block_strided<float>(zfp_stream* zfp, const float* p, ptrdiff_t sx) { return zfp_encode_block_strided_float_1(zfp, p, sx); }

template <>
inline size_t
encode_block_strided<float>(zfp_stream* zfp, const float* p, ptrdiff_t sx, ptrdiff_t sy) { return zfp_encode_block_strided_float_2(zfp, p, sx, sy); }

template <>
inline size_t
encode_block_strided<float>(zfp_stream* zfp, const float* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) { return zfp_encode_block_strided_float_3(zfp, p, sx, sy, sz); }

template <>
inline size_t
encode_block_strided<float>(zfp_stream* zfp, const float* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) { return zfp_encode_block_strided_float_4(zfp, p, sx, sy, sz, sw); }

template <>
inline size_t
encode_block_strided<double>(zfp_stream* zfp, const double* p, ptrdiff_t sx) { return zfp_encode_block_strided_double_1(zfp, p, sx); }

template <>
inline size_t
encode_block_strided<double>(zfp_stream* zfp, const double* p, ptrdiff_t sx, ptrdiff_t sy) { return zfp_encode_block_strided_double_2(zfp, p, sx, sy); }

template <>
inline size_t
encode_block_strided<double>(zfp_stream* zfp, const double* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) { return zfp_encode_block_strided_double_3(zfp, p, sx, sy, sz); }

template <>
inline size_t
encode_block_strided<double>(zfp_stream* zfp, const double* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) { return zfp_encode_block_strided_double_4(zfp, p, sx, sy, sz, sw); }

template <>
inline size_t
encode_partial_block_strided<float>(zfp_stream* zfp, const float* p, size_t nx, ptrdiff_t sx)
{ return zfp_encode_partial_block_strided_float_1(zfp, p, nx, sx); }

template <>
inline size_t
encode_partial_block_strided<float>(zfp_stream* zfp, const float* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy) { return zfp_encode_partial_block_strided_float_2(zfp, p, nx, ny, sx, sy); }

template <>
inline size_t
encode_partial_block_strided<float>(zfp_stream* zfp, const float* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) { return zfp_encode_partial_block_strided_float_3(zfp, p, nx, ny, nz, sx, sy, sz); }

template <>
inline size_t
encode_partial_block_strided<float>(zfp_stream* zfp, const float* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) { return zfp_encode_partial_block_strided_float_4(zfp, p, nx, ny, nz, nw, sx, sy, sz, sw); }

template <>
inline size_t
encode_partial_block_strided<double>(zfp_stream* zfp, const double* p, size_t nx, ptrdiff_t sx)
{ return zfp_encode_partial_block_strided_double_1(zfp, p, nx, sx); }

template <>
inline size_t
encode_partial_block_strided<double>(zfp_stream* zfp, const double* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy) { return zfp_encode_partial_block_strided_double_2(zfp, p, nx, ny, sx, sy); }

template <>
inline size_t
encode_partial_block_strided<double>(zfp_stream* zfp, const double* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) { return zfp_encode_partial_block_strided_double_3(zfp, p, nx, ny, nz, sx, sy, sz); }

template <>
inline size_t
encode_partial_block_strided<double>(zfp_stream* zfp, const double* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) { return zfp_encode_partial_block_strided_double_4(zfp, p, nx, ny, nz, nw, sx, sy, sz, sw); }

// decoder declarations -------------------------------------------------------

template <typename Scalar, uint dims>
inline size_t
decode_block(zfp_stream* zfp, Scalar* block);

template <typename Scalar>
inline size_t
decode_block_strided(zfp_stream* zfp, Scalar* p, ptrdiff_t sx);

template <typename Scalar>
inline size_t
decode_block_strided(zfp_stream* zfp, Scalar* p, ptrdiff_t sx, ptrdiff_t sy);

template <typename Scalar>
inline size_t
decode_block_strided(zfp_stream* zfp, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz);

template <typename Scalar>
inline size_t
decode_block_strided(zfp_stream* zfp, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);

template <typename Scalar>
inline size_t
decode_partial_block_strided(zfp_stream* zfp, Scalar* p, size_t nx, ptrdiff_t sx);

template <typename Scalar>
inline size_t
decode_partial_block_strided(zfp_stream* zfp, Scalar* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy);

template <typename Scalar>
inline size_t
decode_partial_block_strided(zfp_stream* zfp, Scalar* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz);

template <typename Scalar>
inline size_t
decode_partial_block_strided(zfp_stream* zfp, Scalar* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);

// decoder specializations ----------------------------------------------------

template<>
inline size_t
decode_block<float, 1>(zfp_stream* zfp, float* block) { return zfp_decode_block_float_1(zfp, block); }

template<>
inline size_t
decode_block<float, 2>(zfp_stream* zfp, float* block) { return zfp_decode_block_float_2(zfp, block); }

template<>
inline size_t
decode_block<float, 3>(zfp_stream* zfp, float* block) { return zfp_decode_block_float_3(zfp, block); }

template<>
inline size_t
decode_block<float, 4>(zfp_stream* zfp, float* block) { return zfp_decode_block_float_4(zfp, block); }

template<>
inline size_t
decode_block<double, 1>(zfp_stream* zfp, double* block) { return zfp_decode_block_double_1(zfp, block); }

template<>
inline size_t
decode_block<double, 2>(zfp_stream* zfp, double* block) { return zfp_decode_block_double_2(zfp, block); }

template<>
inline size_t
decode_block<double, 3>(zfp_stream* zfp, double* block) { return zfp_decode_block_double_3(zfp, block); }

template<>
inline size_t
decode_block<double, 4>(zfp_stream* zfp, double* block) { return zfp_decode_block_double_4(zfp, block); }

template <>
inline size_t
decode_block_strided<float>(zfp_stream* zfp, float* p, ptrdiff_t sx) { return zfp_decode_block_strided_float_1(zfp, p, sx); }

template <>
inline size_t
decode_block_strided<float>(zfp_stream* zfp, float* p, ptrdiff_t sx, ptrdiff_t sy) { return zfp_decode_block_strided_float_2(zfp, p, sx, sy); }

template <>
inline size_t
decode_block_strided<float>(zfp_stream* zfp, float* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) { return zfp_decode_block_strided_float_3(zfp, p, sx, sy, sz); }

template <>
inline size_t
decode_block_strided<float>(zfp_stream* zfp, float* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) { return zfp_decode_block_strided_float_4(zfp, p, sx, sy, sz, sw); }

template <>
inline size_t
decode_block_strided<double>(zfp_stream* zfp, double* p, ptrdiff_t sx) { return zfp_decode_block_strided_double_1(zfp, p, sx); }

template <>
inline size_t
decode_block_strided<double>(zfp_stream* zfp, double* p, ptrdiff_t sx, ptrdiff_t sy) { return zfp_decode_block_strided_double_2(zfp, p, sx, sy); }

template <>
inline size_t
decode_block_strided<double>(zfp_stream* zfp, double* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) { return zfp_decode_block_strided_double_3(zfp, p, sx, sy, sz); }

template <>
inline size_t
decode_block_strided<double>(zfp_stream* zfp, double* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) { return zfp_decode_block_strided_double_4(zfp, p, sx, sy, sz, sw); }

template <>
inline size_t
decode_partial_block_strided<float>(zfp_stream* zfp, float* p, size_t nx, ptrdiff_t sx) { return zfp_decode_partial_block_strided_float_1(zfp, p, nx, sx); }

template <>
inline size_t
decode_partial_block_strided<float>(zfp_stream* zfp, float* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy) { return zfp_decode_partial_block_strided_float_2(zfp, p, nx, ny, sx, sy); }

template <>
inline size_t
decode_partial_block_strided<float>(zfp_stream* zfp, float* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) { return zfp_decode_partial_block_strided_float_3(zfp, p, nx, ny, nz, sx, sy, sz); }

template <>
inline size_t
decode_partial_block_strided<float>(zfp_stream* zfp, float* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) { return zfp_decode_partial_block_strided_float_4(zfp, p, nx, ny, nz, nw, sx, sy, sz, sw); }

template <>
inline size_t
decode_partial_block_strided<double>(zfp_stream* zfp, double* p, size_t nx, ptrdiff_t sx) { return zfp_decode_partial_block_strided_double_1(zfp, p, nx, sx); }

template <>
inline size_t
decode_partial_block_strided<double>(zfp_stream* zfp, double* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy) { return zfp_decode_partial_block_strided_double_2(zfp, p, nx, ny, sx, sy); }

template <>
inline size_t
decode_partial_block_strided<double>(zfp_stream* zfp, double* p, size_t nx, size_t ny, size_t nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) { return zfp_decode_partial_block_strided_double_3(zfp, p, nx, ny, nz, sx, sy, sz); }

template <>
inline size_t
decode_partial_block_strided<double>(zfp_stream* zfp, double* p, size_t nx, size_t ny, size_t nz, size_t nw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) { return zfp_decode_partial_block_strided_double_4(zfp, p, nx, ny, nz, nw, sx, sy, sz, sw); }

}

#endif
