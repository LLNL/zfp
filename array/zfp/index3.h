// uncompressed index (raw 64-bit offsets)
class VerbatimIndex {
public:
  // constructor for given nbumber of blocks
  VerbatimIndex(uint blocks) :
    data(0),
    block(0) 
  {
    resize(blocks);
  }

  // destructor
  ~VerbatimIndex() { delete[] data; }

  // reset index
  void clear() { block = 0; }

  void resize(uint blocks)
  {
    delete[] data;
    data = new uint64[blocks + 1];
    clear();
  }

  // push block bit size
  void push(size_t size)
  {
    if (block)
      data[block + 1] = data[block] + size;
    else {
      data[0] = 0;
      data[1] = size;
    }
    block++;
  }

  // flush any buffered data
  void flush() {}

  // bit offset of given block id
  size_t operator()(uint id) const { return data[id]; }

protected:
  uint64* data;
  uint block;
};

// hybrid index (raw offset every 8 blocks); templetize?
class Hybrid8Index {
public:
  // constructor for given nbumber of blocks
  Hybrid8Index(uint blocks) :
    data(0),
    block(0) 
  {
    resize(blocks);
  }

  // destructor
  ~Hybrid8Index() { delete[] data; }

  // reset index
  void clear()
  {
    block = 0;
    ptr = 0;
  }

  void resize(uint blocks)
  {
    delete[] data;
    data = new uint64[2 * ((blocks + 7) / 8)];
    clear();
  }

  // push block bit size
  void push(size_t size)
  {
    uint chunk = block / 8;
    buffer[block & 0x7u] = size;
    block++;
    if (!(block & 0x7u)) {
      // store all but low 8-bits of offset
      uint64 hi = (ptr >> 8) << 28;
      uint64 lo = (ptr & UINT64C(0xff)) << 56;
      for (uint k = 0; k < 7; k++) {
        // partition block size into 4 high and 8 low bits
        hi += (buffer[k] >> 8) << (4 * (6 - k));
        lo += (buffer[k] & 0xffu) << (8 * (6 - k));
        ptr += buffer[k];
      }
      ptr += buffer[7];
      data[2 * chunk + 0] = hi;
      data[2 * chunk + 1] = lo;
    }
  }

  // flush any buffered data
  void flush()
  {
    while (block & 0x7u)
      push(0);
  }

  // bit offset of given block id
  size_t operator()(uint id) const
  {
    uint chunk = id / 8;
    uint which = id % 8;
    return offset(data[2 * chunk + 0], data[2 * chunk + 1], which);
  }

protected:
  // kth offset in chunk, 0 <= k <= 7
  static uint64 offset(uint64 h, uint64 l, uint k)
  {
    uint64 base = h >> 32;
    h &= UINT64C(0xffffffff);
    uint64 hi = sum4(h >> (4 * (7 - k)));
    uint64 lo = sum8(l >> (8 * (7 - k)));
    return (base << 12) + (hi << 8) + lo;
  }

  // sum of eight packed 4-bit numbers
  static uint64 sum4(uint64 x)
  {
    uint64 y = x & UINT64C(0xf0f0f0f0);
    x -= y;
    x += y >> 4;
    x += x >> 16;
    x += x >> 8;
    return x & UINT64C(0xff);
  }

  // sum of eight packed 8-bit numbers
  static uint64 sum8(uint64 x)
  {
    uint64 y = x & UINT64C(0xff00ff00ff00ff00);
    x -= y;
    x += y >> 8;
    x += x >> 32;
    x += x >> 16;
    return x & UINT64C(0xffff);
  }

  uint64* data;
  uint block;
  uint64 ptr;
  size_t buffer[8];
};
