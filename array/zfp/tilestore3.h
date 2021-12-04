#ifndef ZFP_TILE_STORE3_H
#define ZFP_TILE_STORE3_H

#include "zfp/tilestore.h"
#include "zfp/tile3.h"

namespace zfp {
namespace internal {

// compressed block store for 3D array
template <typename Scalar, class Codec>
class TileStore3 : public TileStore<Codec> {
public:
  typedef Tile3<Scalar, Codec> tile_type;
  using TileStore<Codec>::set_config;

  // default constructor
  TileStore3() :
    nx(0), ny(0), nz(0),
    bx(0), by(0), bz(0),
    tx(0), ty(0), tz(0),
    tile(0)
  {}

  // block store for array of size nx * ny * nz and given configuration
  TileStore3(size_t nx, size_t ny, size_t nz, const zfp_config& config) :
    tile(0)
  {
    set_size(nx, ny, nz);
    set_config(config);
  }

  ~TileStore3() { free(); }

  // perform a deep copy
  void deep_copy(const TileStore3& s)
  {
    free();
    TileStore<Codec>::deep_copy(s);
    nx = s.nx; ny = s.ny; nz = s.nz;
    bx = s.bx; by = s.by; bz = s.bz;
    tx = s.tx; ty = s.ty; tz = s.tz;
  }

  // resize array
  void resize(size_t nx, size_t ny, size_t nz)
  {
    free();
    set_size(nx, ny, nz);
    if (blocks())
      alloc();
  }

  // allocate memory for tiles
  virtual void alloc()
  {
    this->free();
    TileStore<Codec>::alloc();
    tile = new tile_type[tiles()];
    // TODO: support specifying tile quantum
  }

  // free single-block buffer
  virtual void free()
  {
    TileStore<Codec>::free();
    if (tile) {
      delete[] tile;
      tile = 0;
    }
  }

  // byte size of store data structure components indicated by mask
  virtual size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  { 
    size_t size = 0;
    size += TileStore<Codec>::size_bytes(mask);
    if (mask & ZFP_DATA_META)
      size += sizeof(*this) - sizeof(TileStore<Codec>);
    for (size_t t = 0; t < tiles(); t++)
      size += tile[t].size_bytes(mask);
    return size;
  }

  // conservative buffer size for a single compressed block
  virtual size_t buffer_size() const
  {
    zfp_field* field = zfp_field_3d(0, codec.type, 4, 4, 4);
    size_t size = codec.buffer_size(field);
    zfp_field_free(field);
    return size;
  }

  // number of elements per block
  virtual size_t block_size() const { return 4 * 4 * 4; }

  // total number of blocks
  virtual size_t blocks() const { return bx * by * bz; }

  // array size in blocks
  size_t block_size_x() const { return bx; }
  size_t block_size_y() const { return by; }
  size_t block_size_z() const { return bz; }

  // flat block index for element (i, j, k)
  size_t block_index(size_t i, size_t j, size_t k) const { return (i / 4) + bx * ((j / 4) + by * (k / 4)); }

  // encoding of block dimensions
  uint block_shape(size_t block_index) const
  {
    size_t i = 4 * (block_index % bx); block_index /= bx;
    size_t j = 4 * (block_index % by); block_index /= by;
    size_t k = 4 * block_index;
    uint mx = shape_code(i, nx);
    uint my = shape_code(j, ny);
    uint mz = shape_code(k, nz);
    return mx + 4 * (my + 4 * mz);
  }

  // total number of tiles
  virtual size_t tiles() const { return tx * ty * tz; }

  // encode contiguous block with given index
  size_t encode(size_t block_index, const Scalar* block)
  {
    return encode(block_index, block, 0, 0, 0);
  }

  // encode block with given index from strided array
  size_t encode(size_t block_index, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
  {
    // determine tile index and block index within tile
    size_t t = tile_id(block_index);
    size_t b = block_id(block_index);
    return tile[t].encode(codec, p, sx, sy, sz, b, block_shape(block_index));
  }

  // decode contiguous block with given index
  size_t decode(size_t block_index, Scalar* block, bool cache_block = true) const
  {
    return decode(block_index, block, 0, 0, 0, cache_block);
  }

  // decode block with given index to strided array
  size_t decode(size_t block_index, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, bool cache_block = true) const
  {
    size_t t = tile_id(block_index);
    size_t b = block_id(block_index);
    return tile[t].decode(codec, p, sx, sy, sz, b, block_shape(block_index), cache_block);
  }

protected:
  using TileStore<Codec>::alloc;
  using TileStore<Codec>::free;
  using TileStore<Codec>::shape_code;
  using TileStore<Codec>::codec;

  // tile-local block id associated with given global block index
  size_t block_id(size_t block_index) const
  {
    size_t x = block_index % bx; block_index /= bx;
    size_t y = block_index % by; block_index /= by;
    size_t z = block_index;
    size_t xx = x % tile_type::bx;
    size_t yy = y % tile_type::by;
    size_t zz = z % tile_type::bz;
    return xx + tile_type::bx * (yy + tile_type::by * zz);
  }

  // tile id associated with given block index
  size_t tile_id(size_t block_index) const
  {
    size_t x = block_index % bx; block_index /= bx;
    size_t y = block_index % by; block_index /= by;
    size_t z = block_index;
    size_t xx = x / tile_type::bx;
    size_t yy = y / tile_type::by;
    size_t zz = z / tile_type::bz;
    return xx + tx * (yy + ty * zz);
  }

  // set array dimensions
  void set_size(size_t nx, size_t ny, size_t nz)
  {
    if (nx == 0 || ny == 0 || nz == 0) {
      this->nx = this->ny = this->nz = 0;
      bx = by = bz = 0;
    }
    else {
      this->nx = nx; this->ny = ny; this->nz = nz;
      bx = (nx + 3) / 4; by = (ny + 3) / 4; bz = (nz + 3) / 4;
    }
    tx = zfp::count_up(bx, tile_type::bx);
    ty = zfp::count_up(by, tile_type::by);
    tz = zfp::count_up(bz, tile_type::bz);
  }

  size_t nx, ny, nz; // array dimensions
  size_t bx, by, bz; // array dimensions in number of blocks
  size_t tx, ty, tz; // array dimensions in number of tiles
  tile_type* tile;   // tiles of compressed blocks
};

} // internal
} // zfp

#endif
