#ifdef _OPENMP

static bitstream**
compress_init_par(zfp_stream* stream, const zfp_field* field, uint chunks, uint blocks_per_chunk)
{
  bitstream** bs = malloc(chunks * sizeof(bitstream*));
  size_t size = zfp_stream_maximum_size(stream, field);
  uint i;

  /* avoid copies in fixed-rate mode when each bitstream is word aligned */
  int copy = 0;
  copy |= stream->minbits != stream->maxbits;
  copy |= (stream->maxbits % stream_word_bits) != 0;
  copy |= (stream_wtell(stream->stream) % stream_word_bits) != 0;

  /* set up buffer for each thread to compress to */
  for (i = 0; i < chunks; i++) {
    void* buffer = copy ? malloc(size) : (uchar*)stream_data(stream->stream) + stream_size(stream->stream) + i * blocks_per_chunk * stream->maxbits / CHAR_BIT;
    bs[i] = stream_open(buffer, size);
  }

  return bs;
}

static void
compress_finish_par(zfp_stream* stream, bitstream** src, uint chunks)
{
  bitstream* dst = zfp_stream_bit_stream(stream);
  int copy = (stream_data(dst) != stream_data(*src));
  size_t offset = stream_wtell(dst);
  uint i;
  for (i = 0; i < chunks; i++) {
    size_t bits = stream_wtell(src[i]);
    offset += bits;
    stream_flush(src[i]);
    /* concatenate streams if they are not already contiguous */
    if (copy) {
      stream_rewind(src[i]);
      stream_copy(dst, src[i], bits);
      free(stream_data(src[i]));
    }
    stream_close(src[i]);
  }
  free(src);
  if (!copy)
    stream_wseek(dst, offset);
}

#endif
