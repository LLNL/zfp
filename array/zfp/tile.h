#ifndef ZFP_TILE_H
#define ZFP_TILE_H

#include <algorithm>
#include <climits>
#include "zfp/exception.h"
#include "bitstream.h"

namespace zfp {
namespace internal {

// tile comprising 2^12 = 4096 variable-length 1D, 2D, 3D, or 4D blocks
template <typename Scalar, class Codec>
class Tile {
public:
  // block storage
  enum storage {
    storage_unused = -1, // outside array domain
    storage_empty =   0, // empty/all-zero (no compressed storage)
    storage_xs =      1, // extra small 
    storage_s =       2, // small = 2 * xs
    storage_m =       3, // medium = 4 * xs
    storage_l =       4, // large = 8 * xs
    storage_xl =      5, // extra large = 16 * xs
    storage_cached =  6  // uncompressed and cached
  };

  // destruct tile
  virtual ~Tile()
  {
    zfp::deallocate(data);
  }

  // byte size of specified tile storage classes
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
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

  // storage class for block with given id
  storage block_storage(Codec& codec, size_t id, uchar shape = 0) const
  {
    static Scalar block[4 * 4 * 4 * 4];
    offset p = pos[id];
    if (p == null)
      return storage_empty;
    if (p == cached)
      return storage_cached;
    if (p & 1u)
      return storage_xs;
    Codec c = codec;
    c.open(data, capacity());
    size_t bits = c.decode(offset_bits(p), shape, block);
    return storage(storage_xs + slot_size(word_size(bits)));
  }


#if DEBUG
  // print free lists
  void print_lists(FILE* file = stderr) const
  {
    for (uint s = 0; (int)s < end; s++) {
      uint n = 0;
      fprintf(file, "%2u :", s);
      for (offset p = head[s]; p != null; p = get_next_slot(p), n++)
        fprintf(file, " %5u", p);
      fprintf(file, " null (%u)\n", n);
    }
    fprintf(file, "\n");
  }
#endif

protected:
  typedef uint16 offset; // storage offset of block in number of quanta

// TODO: define stream_word type in bitstream.h
//   note: C++ compiled code must agree on word size with libzfp
#ifdef BIT_STREAM_WORD_TYPE
  // may be 8-, 16-, 32-, or 64-bit unsigned integer type
  typedef BIT_STREAM_WORD_TYPE word;
#else
  // use maximum word size by default for highest speed
  typedef uint64 word;
#endif

  // construct tile with smallest block size 'minbits' rounded up to whole
  // words at least 16 bits wide
  explicit Tile(size_t minbits) :
    quantum(zfp::count_up(minbits, std::max(sizeof(word), sizeof(offset)) * CHAR_BIT)),
    end(-1),
    bytes(0),
    data(0)
  {
    // initialize block offsets and free lists to null
    std::fill(pos, pos + blocks, +null);
    std::fill(head, head + slot_sizes, +null);
  }

  // number of words needed to store 'bits' bits
  static size_t word_size(size_t bits) { return zfp::count_up(bits, sizeof(word) * CHAR_BIT); }

  // quantum size in bits, bytes, and words
  size_t quantum_bits() const { return quantum_bytes() * CHAR_BIT; }
  size_t quantum_bytes() const { return quantum_words() * sizeof(word); }
  size_t quantum_words() const { return quantum; }

  // return bit offset associated with offset p measured in quanta
  size_t offset_bits(offset p) const { return p * quantum_bits(); }
  size_t offset_bytes(offset p) const { return p * quantum_bytes(); }
  size_t offset_words(offset p) const { return p * quantum_words(); }

  // maximum compressed block size
  size_t block_max_bits() const { return quantum_bits() << slot_max; }
  size_t block_max_bytes() const { return quantum_bytes() << slot_max; }
  size_t block_max_words() const { return quantum_words() << slot_max; }

  // amount of compressed data allocated in bytes
  size_t capacity() const { return end < 0 ? 0u : quantum_bytes() << end; }

  // buddy slot to slot of size s at offset p
  static offset buddy_slot(uint s, offset p) { return p ^ (1u << s); }

  // parent slot to two buddy slots
  static offset parent_slot(uint p, uint q) { return p & q; }

  // return next free slot stored at offset p
  offset get_next_slot(offset p) const { return data[offset_words(p)]; }

  // set next free slot pointer at offset p
  void set_next_slot(offset p, offset next)
  {
    if (next == p)
      throw zfp::exception("zfp internal error: free list loop");
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

  // insert slot at offset p of size s into sorted free list s
  void put_slot(uint s, offset p, bool free)
  {
#if DEBUG
    // sanity check that slot is not already on free list
    for (offset q = head[s]; q != null; q = get_next_slot(q))
      if (q == p)
        throw zfp::exception("slot is already free");
    fprintf(stderr, "put %u @ %u\n", s, p);
#endif
    // cached and null offsets are reserved
    if (p < cached) {
      offset b = buddy_slot(s, p);
      if (b == cached) {
        // special rule: slot at offset 'cached' is always free and mergeable
        put_slot(s + 1, p, false);
      }
      else if (head[s] == b) {
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
  }

  // return storage size needed for 'words' words
  uint slot_size(size_t words) const
  {
    uint s = 0;
    for (size_t q = quantum_words(); q < words; q *= 2, s++);
    if (s > slot_max)
{
fprintf(stderr, "slot size = %zu words\n", words);
      throw zfp::exception("zfp slot size request too large");
}
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
      for (uint s = size, t = size + 1; t < slot_sizes; t++) {
        p = get_slot(t);
        if (p != null) {
          // partition slot into maximal pieces and put them on free lists
          for (offset q = p + (1u << s); s < t; q += 1u << s++)
            put_slot(s, q, true);
          return p;
        }
      }
      // no free slot was found; allocate more memory and try again
      size_t cap = capacity();
      uint s = size;
      if (end < 0) {
        p = 0;
        end = s;
      }
      else {
        p = 1u << end;
        s = end++;
      }
      if (end >= (int)slot_sizes)
        throw zfp::exception("zfp internal error: tile storage overflow");
      // allocate more memory
      zfp::reallocate(data, capacity(), cap);
      put_slot(s, p, false);
    }
  }

  // deallocate space of size 'words' words at offset p
  void deallocate(offset p, uint words)
  {
    uint s = slot_size(words);
    put_slot(s, p, true);
    // attempt to free unused space
    if (!bytes) {
      // all blocks are empty; free all allocated storage
      head[end] = null;
      end = -1;
      zfp::deallocate(data);
      data = 0;
    }
    else if (end > 0 && head[end - 1] == (1u << (end - 1))) {
      // the top half of allocated space is unused; free it
      size_t cap = capacity();
      end--;
      head[end] = null;
      zfp::reallocate(data, capacity(), cap);
    }
  }

  // allocate space for and store buffered compressed block of given bit size
  void store_block(size_t id, const word* restrict_ buffer, size_t bits)
  {
    offset p = null;
    // if block is empty, no storage is needed; otherwise, find space for it
    if (bits > 1u) {
      // allocate memory for compressed block
      size_t words = std::min(word_size(bits), block_max_words());
      p = allocate(words);
      assert(p != null);
      assert(p != cached);
      // copy compressed data to persistent storage
      std::copy(buffer, buffer + words, data + offset_words(p));
    }
    pos[id] = p;
#if DEBUG
    print_lists();
#endif
  }

#ifdef DEBUG
  // display internal fragmentation of tile's memory pool
  void fragmentation(const Codec& codec, FILE* file = stderr) const
  {
    uchar* slot = new uchar[0x10000];
    std::fill(slot, slot + 0x10000, 0xff);
    size_t usage = 0;
    size_t zeros = 0;
    // process compressed blocks
    for (uint id = 0; id < blocks; id++) {
      offset p = pos[id];
      if (p == null)
        zeros++;
      else if (p != cached) {
        // determine slot size s
        uint s = block_storage(codec, id) - storage_xs;
        usage += quantum_bytes() << s;
        for (uint i = 0; i < (1u << s); i++)
          slot[p + i] = s;
      }
    }
    // process free slots
    for (uint s = 0; s < slot_sizes; s++) {
      for (offset p = head[s]; p != null; p = get_next_slot(p))
        for (uint i = 0; i < (1u << s); i++)
          slot[p + i] = 0x80 + s;
    }
    // print slot size information
    for (uint i = 0; i < 0x10000; i++)
      fprintf(file, "%02x ", slot[i]);
    fprintf(file, "\n");
    delete[] slot;
    fprintf(file, "bytes=%zu vs %zu\n", bytes, usage);
    fprintf(file, "zeros=%zu\n", zeros);
  }
#endif


  // constants
  static const uint blocks = 0x1000u;   // number of blocks per tile
  static const uint slot_sizes = 17;    // number of distinct slot sizes
  static const uint slot_max = 4;       // slot size of largest compressed block
  static const offset null = 0xffffu;   // null offset
  static const offset cached = 0xfffeu; // decompressed and cached block

  const size_t quantum;    // smallest slot size in words
  int end;                 // largest slot size (index) supported (initially -1)
  size_t bytes;            // amount of compressed data and padding in bytes
  word* restrict_ data;    // pointer to compressed data
  offset head[slot_sizes]; // free lists for slots of increasing size
  offset pos[blocks];      // block positions in number of quanta
};

} // internal
} // zfp

#endif
