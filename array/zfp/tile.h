#ifndef ZFP_TILE_H
#define ZFP_TILE_H

#include <algorithm>
#include <cassert>
#include <climits>
#include "bitstream.h"
//#include "../../src/inline/inline.h"
//#include "../../src/inline/bitstream.c"

namespace zfp {

class Tile {
public:
  // destruct tile
  virtual ~Tile()
  {
    zfp::deallocate(data);
  }

  // amount of compressed data allocated in bytes
  size_t capacity() const { return end < 0 ? 0u : quantum_bytes() << end; }

  // total amount of memory allocated for tile data structure
  size_t total_capacity() const
  { 
    size_t bytes = sizeof(Tile);
    bytes += capacity();
    return bytes;
  }

  // amount of compressed data stored in bytes
  size_t size() const { return bytes; }

protected:
  typedef uint16 offset; // storage offset of block in number of quanta
//#warning "need to rethink this"
  typedef uint64 word;

//#warning "quantum must be at least 16 bits to store offsets"
  // construct tile with smallest block size 'minbits'
  explicit Tile(uint minbits) :
    quantum(stream_word_bits * ((minbits + stream_word_bits - 1) / stream_word_bits) / CHAR_BIT),
    bytes(0),
    end(-1),
    data(0)
  {
    // initialize block offsets and free lists to null
    std::fill(pos, pos + blocks, null);
    std::fill(head, head + sizes, null);
  }

  // quantum size in bits, bytes, and words
  size_t quantum_bits() const { return quantum_bytes() * CHAR_BIT; }
  size_t quantum_bytes() const { return quantum; }
  size_t quantum_words() const { return quantum_bits() / stream_word_bits; }
//#warning "measure quantum in words to avoid division?"

  // return bit offset associated with offset p measured in quanta
  size_t offset_bits(offset p) const { return p * quantum_bits(); }
  size_t offset_bytes(offset p) const { return p * quantum_bytes(); }
  size_t offset_words(offset p) const { return p * quantum_words(); }

  // number of words in 'bits' bits
  static size_t word_size(size_t bits) { return bits / stream_word_bits; }

//#warning "what if word is a byte?"
  // return next free slot stored at offset p
  offset get_next_slot(offset p) { return data[offset_words(p)]; }

  // set next free slot pointer at offset p
  void set_next_slot(offset p, offset next) { data[offset_words(p)] = next; }

  // return first available free slot of size s or null if none unavailable
  offset dequeue_slot(uint s)
  {
    offset p = head[s];
    if (p != null) {
      head[s] = get_next_slot(p);
      bytes += quantum << s;
    }
    return p;
  }

  // insert slot at p of size s at head of free list s
  void enqueue_slot(uint s, offset p)
  {
    set_next_slot(p, head[s]);
    head[s] = p;
    bytes -= quantum << s;
//#warning "should attempt to merge slots and free memory here"
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
    for (;;) {
      // determine slot size needed
      uint s = slot_size(words);
      // check free list
      offset p = dequeue_slot(s);
      if (p != null)
        return p;
      // no free slot of the requested size; find larger slot to split
      for (uint t = s + 1; t < sizes; t++) {
        p = dequeue_slot(t);
        if (p != null) {
          // partition slot into maximal pieces and put them on free lists
          for (offset q = p + (1u << s); s < t; q += 1u << s++)
            enqueue_slot(s, q);
          return p;
        }
      }
      // no free slot was found; allocate more memory and try again
      if (end < 0) {
        p = 0;
        end = s;
      }
      else {
        p = 1u << end;
        s = end++;
      }
//#warning "assert"
      assert(end < (int)sizes);
      // allocate more memory
      zfp::reallocate(data, capacity());
      enqueue_slot(s, p);
    }
  }

  static const uint blocks = 0x1000u; // number of blocks per tile
  static const uint sizes = 16;       // number of distinct slot sizes
  static const offset null = 0xffffu; // null offset
  static const uint max_size = 4;     // largest compressed block size

  const size_t quantum; // smallest slot size in bytes
  size_t bytes;         // amount of compressed data used in bytes
  int end;              // largest slot size (index) supported (initially -1)
  word* data;           // pointer to compressed data
  offset head[sizes];   // free lists for slots of increasing size
  offset pos[blocks];   // block positions in number of quanta
};

}

#endif
