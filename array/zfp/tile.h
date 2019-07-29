#ifndef ZFP_TILE_H
#define ZFP_TILE_H

#include <algorithm>
#include <cassert>
#include <climits>
#include "bitstream.h"

namespace zfp {

// tile comprising 2^12 = 4096 variable-length blocks
class Tile {
public:
  // destruct tile
  virtual ~Tile()
  {
    zfp::deallocate(data);
  }

  // byte size of specified tile storage classes
  size_t size(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_UNUSED)
      size += capacity() - bytes;
    if (mask & ZFP_DATA_META)
      size += sizeof(Tile) - sizeof(pos);
    if (mask & ZFP_DATA_PAYLOAD)
      size += bytes;
    if (mask & ZFP_DATA_INDEX)
      size += sizeof(pos);
    return size;
  }

protected:
  typedef uint16 offset; // storage offset of block in number of quanta
//#warning "need to rethink this"
  typedef uint64 word;

//#warning "quantum must be at least 16 bits to store offsets"
  // construct tile with smallest block size 'minbits' rounded up to whole words
  explicit Tile(uint minbits) :
    quantum(stream_word_bits * ((minbits + stream_word_bits - 1) / stream_word_bits) / CHAR_BIT),
    bytes(0),
    end(-1),
    data(0)
  {
    // initialize block offsets and free lists to null
    std::fill(pos, pos + blocks, +null);
    std::fill(head, head + sizes, +null);
  }

  // number of words in 'bits' bits
  static size_t word_size(size_t bits) { return bits / stream_word_bits; }

  // quantum size in bits, bytes, and words
  size_t quantum_bits() const { return quantum_bytes() * CHAR_BIT; }
  size_t quantum_bytes() const { return quantum; }
  size_t quantum_words() const { return quantum_bits() / stream_word_bits; }
//#warning "measure quantum in words to avoid division?"

  // return bit offset associated with offset p measured in quanta
  size_t offset_bits(offset p) const { return p * quantum_bits(); }
  size_t offset_bytes(offset p) const { return p * quantum_bytes(); }
  size_t offset_words(offset p) const { return p * quantum_words(); }

  // amount of compressed data allocated in bytes
  size_t capacity() const { return end < 0 ? 0u : quantum_bytes() << end; }

#ifdef DEBUG
  void print_lists() const
  {
    for (uint s = 0; s < sizes; s++) {
      uint n = 0;
      printf("%2u :", s);
      for (offset p = head[s]; p != null; p = get_next_slot(p), n++)
        printf(" %5u", p);
      printf(" null (%u)\n", n);
    }
  }
#endif

  // buddy slot to slot of size s at offset p
  static offset buddy_slot(uint s, offset p) { return p ^ (1u << s); }

  // parent slot to two buddy slots
  static offset parent_slot(uint p, uint q) { return p & q; }

#warning "what if word is a byte?"
  // return next free slot stored at offset p
  offset get_next_slot(offset p) const { return data[offset_words(p)]; }

  // set next free slot pointer at offset p
  void set_next_slot(offset p, offset next)
  {
    assert(next != p);
    data[offset_words(p)] = next;
  }

  // return first available free slot of size s or null if none is available
  offset get_slot(uint s)
  {
    offset p = head[s];
    if (p != null) {
      head[s] = get_next_slot(p);
      bytes += quantum_bytes() << s;
    }
    return p;
  }

  // insert slot at offset p of size s into free list s
  void put_slot(uint s, offset p, bool free)
  {
    offset b = buddy_slot(s, p);
    assert(p != null && b != null);
    if (head[s] == b) {
      // merge with buddy slot
      head[s] = get_next_slot(b);
      put_slot(s + 1, parent_slot(p, b), false);
    }
    else if (head[s] > p) {
      // insert p at head when head is null or larger than p
      set_next_slot(p, head[s]);
      head[s] = p;
    }
    else {
      // linearly scan for buddy or place to insert p
      offset q, r;
      for (q = head[s], r = get_next_slot(q); r != b && r < p; q = r, r = get_next_slot(r));
      if (r == b) {
        // merge with buddy slot
        set_next_slot(q, get_next_slot(r));
        put_slot(s + 1, parent_slot(p, b), false);
      }
      else {
        // insert p between q and r
        set_next_slot(q, p);
        set_next_slot(p, r);
      }
    }
    if (free)
      bytes -= quantum_bytes() << s;
  }

  // return storage size needed for 'words' words
  uint slot_size(size_t words) const
  {
    uint s = 0;
    for (size_t q = quantum_words(); q < words; q *= 2, s++);
    return s;
  }

  // allocate space for (at least) 'words' words
  offset allocate(uint words)
  {
    // determine slot size needed
    uint size = slot_size(words);
    for (;;) {
      // check free list
      offset p = get_slot(size);
      if (p != null)
        return p;
      // no free slot of the requested size; find larger slot to split
      for (uint s = size, t = size + 1; t < sizes; t++) {
        p = get_slot(t);
        if (p != null) {
          // partition slot into maximal pieces and put them on free lists
          for (offset q = p + (1u << s); s < t; q += 1u << s++)
            put_slot(s, q, true);
          return p;
        }
      }
      // no free slot was found; allocate more memory and try again
      size_t old_cap = capacity();
      uint s = size;
      if (end < 0) {
        p = 0;
        end = s;
      }
      else {
        p = 1u << end;
        s = end++;
      }
      assert(end < (int)sizes);
      // allocate more memory
      zfp::reallocate(data, capacity(), old_cap);
      put_slot(s, p, false);
    }
  }

  // deallocate space of size 'words' words at offset p
  void deallocate(offset p, uint words)
  {
    uint s = slot_size(words);
    put_slot(s, p, true);
#warning "attempt to free memory here"
  }

  static const uint blocks = 0x1000u;   // number of blocks per tile
  static const uint sizes = 17;         // number of distinct slot sizes
  static const offset null = 0xffffu;   // null offset
  static const offset cached = 0xfffeu; // decompressed and cached block
  static const uint max_size = 4;       // largest compressed block size

  const size_t quantum; // smallest slot size in bytes
  size_t bytes;         // amount of compressed data used in bytes
// rename to maxsize? (max_size exists above) or lgcap?
  int end;              // largest slot size (index) supported (initially -1)
  word* data;           // pointer to compressed data
  offset head[sizes];   // free lists for slots of increasing size
  offset pos[blocks];   // block positions in number of quanta
};

}

#endif
