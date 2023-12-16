#include "zfp.h"

#define DIV_ROUND_UP(x, y) (((x) + (y) - 1) / (y))
#define BITS_TO_BYTES(x) DIV_ROUND_UP(x, CHAR_BIT)

#define ZFP_HEADER_SIZE_BITS (ZFP_MAGIC_BITS + ZFP_META_BITS + ZFP_MODE_SHORT_BITS)
