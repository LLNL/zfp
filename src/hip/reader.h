#ifndef ZFP_HIP_READER_H
#define ZFP_HIP_READER_H

namespace zfp {
namespace hip {
namespace internal {

class BlockReader {
private:
  // number of bits in a buffered word
  static constexpr size_t word_size = sizeof(Word) * CHAR_BIT;

  uint bits;               // number of buffered bits (0 <= bits < word_size)
  Word buffer;             // buffer for incoming bits (buffer < 2^bits)
  const Word* ptr;         // pointer to next word to be read
  const Word* const begin; // beginning of stream

  // read a single word from memory
  inline __device__
  Word read_word() { return *ptr++; }

public:
  typedef unsigned long long int Offset;

  __device__ BlockReader(const Word* data, Offset offset = 0) :
    begin(data)
  {
    rseek(offset);
  }

  // return bit offset to next bit to be read
  inline __device__
  Offset rtell() const { return word_size * (Offset)(ptr - begin) - bits; }

  // position stream for reading at given bit offset
  inline __device__
  void rseek(Offset offset)
  {
    uint n = (uint)(offset % word_size);
    ptr = begin + (size_t)(offset / word_size);
    if (n) {
      buffer = read_word() >> n;
      bits = word_size - n;
    }
    else {
      buffer = 0;
      bits = 0;
    }
  }

  // read single bit (0 or 1)
  inline __device__ 
  uint read_bit()
  {
    uint bit;
    if (!bits) {
      buffer = read_word();
      bits = word_size;
    }
    bits--;
    bit = (uint)buffer & 1u;
    buffer >>= 1;
    return bit;
  }

  // read 0 <= n <= 64 bits
  inline __device__ 
  uint64 read_bits(uint n)
  {
    uint64 value = buffer;
    if (bits < n) {
      // keep fetching word_size bits until enough bits are buffered
      do {
        // assert: 0 <= bits < n <= 64
        buffer = read_word();
        value += (uint64)buffer << bits;
        bits += word_size;
      } while (sizeof(buffer) < sizeof(value) && bits < n);
      // assert: 1 <= n <= bits < n + word_size
      bits -= n;
      if (!bits) {
        // value holds exactly n bits; no need for masking
        buffer = 0;
      }
      else {
        // assert: 1 <= bits < word_size
        buffer >>= word_size - bits;
        // assert: 1 <= n <= 64
        value &= ((uint64)2 << (n - 1)) - 1;
      }
    }
    else {
      // assert: 0 <= n <= bits < word_size <= 64 */
      bits -= n;
      buffer >>= n;
      value &= ((uint64)1 << n) - 1;
    }
    return value;
  }

  // skip over the next n bits (n >= 0)
  inline __device__
  void skip(Offset n) { rseek(rtell() + n); }

  // align stream on next word boundary
  inline __device__
  uint align()
  {
    uint count = bits;
    if (count)
      skip(count);
    return count;
  }
}; // BlockReader

} // namespace internal
} // namespace hip
} // namespace zfp

#endif
