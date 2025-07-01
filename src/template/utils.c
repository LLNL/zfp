#ifndef ZFP_UTILS_H
#define ZFP_UTILS_H

/* size / unit rounded up to the next integer */
#define count_up_(type) /* (type size, type unit) */\
{\
  return (size + unit - 1) / unit;\
}

/* smallest multiple of unit greater than or equal to size */
#define round_up_(type) /* (type size, type unit) */\
{\
  size += unit - 1;\
  size -= size % unit;\
  return size;\
}

/* template instantiations */
static uint _tdef1(count_up, uint, (uint size, uint unit))
static size_t _tdef1(count_up, size_t, (size_t size, size_t unit))
static uint64 _tdef1(count_up, uint64, (uint64 size, uint64 unit))
static bitstream_size _tdef1(count_up, bitstream_size, (bitstream_size size, bitstream_size unit))

static uint _tdef1(round_up, uint, (uint size, uint unit))
static size_t _tdef1(round_up, size_t, (size_t size, size_t unit))
static uint64 _tdef1(round_up, uint64, (uint64 size, uint64 unit))
static bitstream_size _tdef1(round_up, bitstream_size, (bitstream_size size, bitstream_size unit))

#endif
