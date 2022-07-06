#ifndef ZFP_BITSTREAM_H
#define ZFP_BITSTREAM_H

#include <stddef.h>
#include "zfp/internal/zfp/types.h"
#include "zfp/internal/zfp/system.h"

/* forward declaration of opaque type */
typedef struct bitstream bitstream;

/* bit offset into stream where bits are read/written */
typedef uint64 bitstream_offset;

/* type for counting number of bits in a stream */
typedef bitstream_offset bitstream_size;

/* type for counting a small number of bits in a stream */
typedef size_t bitstream_count;

extern_ const size_t stream_word_bits; /* bit stream granularity */

#ifndef inline_
#ifdef __cplusplus
extern "C" {
#endif

/* allocate and initialize bit stream */
bitstream* stream_open(void* buffer, size_t bytes);

/* close and deallocate bit stream */
void stream_close(bitstream* stream);

/* make a copy of bit stream to shared memory buffer */
bitstream* stream_clone(const bitstream* stream);

/* word size in bits (equal to stream_word_bits) */
bitstream_count stream_alignment();

/* pointer to beginning of stream */
void* stream_data(const bitstream* stream);

/* current byte size of stream (if flushed) */
size_t stream_size(const bitstream* stream);

/* byte capacity of stream */
size_t stream_capacity(const bitstream* stream);

/* number of words per block */
size_t stream_stride_block(const bitstream* stream);

/* number of blocks between consecutive blocks */
ptrdiff_t stream_stride_delta(const bitstream* stream);

/* read single bit (0 or 1) */
uint stream_read_bit(bitstream* stream);

/* write single bit */
uint stream_write_bit(bitstream* stream, uint bit);

/* read 0 <= n <= 64 bits */
uint64 stream_read_bits(bitstream* stream, bitstream_count n);

/* write 0 <= n <= 64 low bits of value and return remaining bits */
uint64 stream_write_bits(bitstream* stream, uint64 value, bitstream_count n);

/* return bit offset to next bit to be read */
bitstream_offset stream_rtell(const bitstream* stream);

/* return bit offset to next bit to be written */
bitstream_offset stream_wtell(const bitstream* stream);

/* rewind stream to beginning */
void stream_rewind(bitstream* stream);

/* position stream for reading at given bit offset */
void stream_rseek(bitstream* stream, bitstream_offset offset);

/* position stream for writing at given bit offset */
void stream_wseek(bitstream* stream, bitstream_offset offset);

/* skip over the next n bits */
void stream_skip(bitstream* stream, bitstream_size n);

/* append n zero-bits to stream */
void stream_pad(bitstream* stream, bitstream_size n);

/* align stream on next word boundary */
bitstream_count stream_align(bitstream* stream);

/* flush out any remaining buffered bits */
bitstream_count stream_flush(bitstream* stream);

/* copy n bits from one bit stream to another */
void stream_copy(bitstream* dst, bitstream* src, bitstream_size n);

#ifdef BIT_STREAM_STRIDED
/* set block size in number of words and spacing in number of blocks */
int stream_set_stride(bitstream* stream, size_t block, ptrdiff_t delta);
#endif

#ifdef __cplusplus
}
#endif
#endif /* !inline_ */

#endif
