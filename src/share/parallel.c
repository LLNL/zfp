#ifdef _OPENMP

/* block index at which chunk begins */
static size_t
chunk_offset(size_t blocks, size_t chunks, size_t chunk)
{
  return (size_t)(((uint64)blocks * (uint64)chunk) / chunks);
}

/* initialize per-thread bit streams for parallel compression */
static bitstream**
compress_init_par(zfp_stream* stream, const zfp_field* field, size_t chunks, size_t blocks)
{
  bitstream** bs;
  zfp_bool copy;
  size_t n = 4 * (blocks + chunks - 1) / chunks;
  size_t size;
  size_t chunk;

  /* determine maximum size buffer needed per thread */
  zfp_field f = *field;
  switch (zfp_field_dimensionality(field)) {
    case 1:
      f.nx = n;
      break;
    case 2:
      f.nx = 4;
      f.ny = n;
      break;
    case 3:
      f.nx = 4;
      f.ny = 4;
      f.nz = n;
      break;
    case 4:
      f.nx = 4;
      f.ny = 4;
      f.nz = 4;
      f.nw = n;
      break;
    default:
      return NULL;
  }
  size = zfp_stream_maximum_size(stream, &f);

  /* avoid copies in fixed-rate mode when each bitstream is word aligned */
  copy = (stream->minbits != stream->maxbits) ||
         (stream->maxbits % stream_word_bits != 0) ||
         (stream_wtell(stream->stream) % stream_word_bits != 0);

  /* set up buffer for each thread to compress to */
  bs = (bitstream**)malloc(chunks * sizeof(bitstream*));
  if (!bs)
    return NULL;
  for (chunk = 0; chunk < chunks; chunk++) {
    size_t block = chunk_offset(blocks, chunks, chunk);
    void* buffer = copy ? malloc(size) : (uchar*)stream_data(stream->stream) + stream_size(stream->stream) + block * (stream->maxbits / CHAR_BIT);
    if (!buffer)
      break;
    bs[chunk] = stream_open(buffer, size);
  }

  /* handle memory allocation failure */
  if (copy && chunk < chunks) {
    while (chunk--) {
      free(stream_data(bs[chunk]));
      stream_close(bs[chunk]);
    }
    free(bs);
    bs = NULL;
  }

  return bs;
}

/* flush and concatenate bit streams if needed */
static void
compress_finish_par(zfp_stream* stream, bitstream** src, size_t chunks)
{
  bitstream* dst = zfp_stream_bit_stream(stream);
  zfp_bool copy = (stream_data(dst) != stream_data(*src));
  bitstream_offset offset = stream_wtell(dst);
  size_t chunk;

  /* flush each stream and concatenate if necessary */
  for (chunk = 0; chunk < chunks; chunk++) {
    bitstream_size bits = stream_wtell(src[chunk]);
    offset += bits;
    stream_flush(src[chunk]);
    /* concatenate streams if they are not already contiguous */
    if (copy) {
      stream_rewind(src[chunk]);
      stream_copy(dst, src[chunk], bits);
      free(stream_data(src[chunk]));
    }
    stream_close(src[chunk]);
  }

  free(src);
  if (!copy)
    stream_wseek(dst, offset);
}

/* initialize per-thread bit streams for parallel decompression */
static bitstream**
decompress_init_par(zfp_stream* stream, uint chunks, uint blocks)
{
  void* buffer = stream_data(stream->stream);
  const size_t size = stream_size(stream->stream);
  zfp_mode mode = zfp_stream_compression_mode(stream);
  bitstream** bs;
  uint i;

  /* set up buffer for each thread to decompress from */
  bs = malloc(chunks * sizeof(bitstream*));
  if (!bs)
    return NULL;

  if (mode == zfp_mode_fixed_rate) {
    const size_t maxbits = stream->maxbits;
    for (i = 0; i < chunks; i++) {
      size_t offset = chunk_offset(blocks, chunks, i) * maxbits;
      bs[i] = stream_open(buffer, size);
      if (!bs[i]) {
        free(bs);
        return NULL;
      }
      /* point bit stream to the beginning of the chunk */
      stream_rseek(bs[i], offset);
    }
  }
  else {
    const zfp_index_type type = stream->index->type;
    if (type == zfp_index_offset) {
      const uint64* offset_table = stream->index->data;
      for (i = 0; i < chunks; i++) {
        bs[i] = stream_open(buffer, size);
        if (!bs[i]) {
          free(bs);
          return NULL;
        }
        /* point bit stream to the beginning of the chunk */
        stream_rseek(bs[i], (size_t)offset_table[i]);
      }
    }
    else {
      /* unsupported index type */
      free(bs);
      return NULL;
    }
  }
  return bs;
}

/* close all bit streams */
static size_t
decompress_finish_par(bitstream** bs, uint chunks)
{
  size_t max_offset = 0;
  uint i;
  for (i = 0; i < chunks; i++) {
    size_t offset = stream_rtell(bs[i]);
    if (max_offset < offset)
      max_offset = offset;
    stream_close(bs[i]);
  }
  free(bs);
  return max_offset;
}

#endif
