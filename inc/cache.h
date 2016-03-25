#ifndef CACHE_H
#define CACHE_H

#include "memory.h"

#ifdef CACHE_PROFILE
  // maintain stats on hit and miss rates
  #include <iostream>
#endif

typedef unsigned int uint;

// two-way skew-associative write-back cache
template <class Line>
class Cache {
public:
  // cache line index (zero is reserved for unused lines)
  typedef uint Index;

  // cache tag containing line meta data
  class Tag {
  public:
    Tag() : x(0) {}

    Tag(Index x, bool d) : x(2 * x + d) {}

    // cache line index
    Index index() const { return x >> 1; }

    // is line dirty?
    bool dirty() const { return x & 1; }

    // is line used?
    bool used() const { return x != 0; }

    // mark line as dirty
    void mark() { x |= 1u; }

    // mark line as unused
    void clear() { x = 0; }

  protected:
    Index x;
  };

  // allocate cache with at least minsize lines
  Cache(uint minsize)
  {
    for (mask = minsize ? minsize - 1 : 1; mask & (mask + 1); mask |= mask + 1);
    tag = (Tag*)allocate(((size_t)mask + 1) * sizeof(Tag), 0x100);
    line = (Line*)allocate(((size_t)mask + 1) * sizeof(Line), 0x100);
#ifdef CACHE_PROFILE
    std::cerr << "cache lines=" << mask + 1 << std::endl;
    phit[0] = shit[0] = miss[0] = back[0] = 0;
    phit[1] = shit[1] = miss[1] = back[1] = 0;
#endif
    clear();
  }

  ~Cache()
  {
    deallocate(tag);
    deallocate(line);
#ifdef CACHE_PROFILE
    std::cerr << "cache R1=" << phit[0] << " R2=" << shit[0] << " RM=" << miss[0] << " RB=" << back[0]
              <<      " W1=" << phit[1] << " W2=" << shit[1] << " WM=" << miss[1] << " WB=" << back[1] << std::endl;
#endif
  }

  // look up cache line #x and return pointer to it if in the cache;
  // otherwise return null
  const Line* lookup(Index x) const
  {
    uint i = primary(x);
    if (tag[i].index() == x)
      return line + i;
#ifndef CACHE_ONEWAY
    uint j = secondary(x);
    if (tag[j].index() == x)
      return line + j;
#endif
    return 0;
  }

  // look up cache line #x and set ptr to where x is or should be stored;
  // if the returned tag does not match x, then the caller must implement
  // write-back (if the line is in use) and then fetch the requested line
  Tag access(Line*& ptr, Index x, bool write)
  {
    uint i = primary(x);
    if (tag[i].index() == x) {
      ptr = line + i;
      if (write)
        tag[i].mark();
#ifdef CACHE_PROFILE
      phit[write]++;
#endif
      return tag[i];
    }
#ifndef CACHE_ONEWAY
    uint j = secondary(x);
    if (tag[j].index() == x) {
      ptr = line + j;
      if (write)
        tag[j].mark();
#ifdef CACHE_PROFILE
      shit[write]++;
#endif
      return tag[j];
    }
    // cache line not found; prefer primary and not dirty slots
    i = tag[j].used() && (!tag[i].dirty() || tag[j].dirty()) ? i : j;
#endif
    ptr = line + i;
    Tag t = tag[i];
    tag[i] = Tag(x, write);
#ifdef CACHE_PROFILE
    miss[write]++;
    if (tag[i].dirty())
      back[write]++;
#endif
    return t;
  }

  // clear cache without writing back
  void clear()
  {
    for (uint i = 0; i <= mask; i++)
      tag[i].clear();
  }

protected:
  uint primary(Index x) const { return x & mask; }
  uint secondary(Index x) const
  {
#ifdef CACHE_FAST_HASH
    // max entropy hash for 26- to 16-bit mapping (not full avalanche)
    x -= x <<  7;
    x ^= x >> 16;
    x -= x <<  3;
#else
    // Jenkins hash; see http://burtleburtle.net/bob/hash/integer.html
    x -= x <<  6;
    x ^= x >> 17;
    x -= x <<  9;
    x ^= x <<  4;
    x -= x <<  3;
    x ^= x << 10;
    x ^= x >> 15;
#endif
    return x & mask;
  }

  Index mask; // cache line mask
  Tag* tag;   // cache line tags
  Line* line; // actual decompressed cache lines
#ifdef CACHE_PROFILE
  unsigned long long phit[2]; // number of primary hits
  unsigned long long shit[2]; // number of secondary hits
  unsigned long long miss[2]; // number of misses
  unsigned long long back[2]; // number of write-backs
#endif
};

#endif
