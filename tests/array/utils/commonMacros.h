#include "zfp.h"

#define ROUND_UP_TO_MULTIPLE(x, y) (((x) + (y) - 1) / (y))
#define BITS_TO_BYTES(x) ROUND_UP_TO_MULTIPLE(x, CHAR_BIT)
#define BITS_TO_WORDS(x) ROUND_UP_TO_MULTIPLE(x, stream_word_bits)

#define ZFP_HEADER_SIZE_BITS (ZFP_MAGIC_BITS + ZFP_META_BITS + ZFP_MODE_SHORT_BITS)
