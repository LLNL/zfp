.. include:: defs.rst

.. cpp:namespace:: zfp

FAQ
===

The following is a list of answers to frequently asked questions.  For
questions not answered here or elsewhere in the documentation, please
`e-mail us <mailto:zfp@llnl.gov>`__.

Questions answered in this FAQ:

  0. :ref:`Do zfp arrays use C or Fortran order? <q-layout>`
  #. :ref:`Can zfp compress vector fields? <q-vfields>`
  #. :ref:`Should I declare a 2D array as zfp::array1d a(nx * ny, rate)? <q-array2d>`
  #. :ref:`How can I initialize a zfp compressed array from disk? <q-read>`
  #. :ref:`Can I use zfp to represent dense linear algebra matrices? <q-matrix>`
  #. :ref:`Can zfp compress logically regular but geometrically irregular data? <q-structured>`
  #. :ref:`Does zfp handle infinities, NaNs,and subnormal floating-point numbers? <q-valid>`
  #. :ref:`Can zfp handle data with some missing values? <q-missing>`
  #. :ref:`Can I use zfp to store integer data? <q-integer>`
  #. :ref:`Can I compress 32-bit integers using zfp? <q-int32>`
  #. :ref:`Why does zfp corrupt memory if my allocated buffer is too small? <q-overrun>`
  #. :ref:`Are zfp compressed streams portable across platforms? <q-portability>`
  #. :ref:`How can I achieve finer rate granularity? <q-granularity>`
  #. :ref:`Can I generate progressive zfp streams? <q-progressive>`
  #. :ref:`How do I initialize the decompressor? <q-init>`
  #. :ref:`Must I use the same parameters during compression and decompression? <q-same>`
  #. :ref:`Do strides have to match during compression and decompression? <q-strides>`
  #. :ref:`Why does zfp sometimes not respect my error tolerance? <q-tolerance>`
  #. :ref:`Why is the actual rate sometimes not what I requested? <q-rate>`
  #. :ref:`Can zfp perform compression in place? <q-inplace>`
  #. :ref:`Can zfp bound the point-wise relative error? <q-relerr>`
  #. :ref:`Does zfp support lossless compression? <q-lossless>`
  #. :ref:`Why is my actual, measured error so much smaller than the tolerance? <q-abserr>`
  #. :ref:`Are parallel compressed streams identical to serial streams? <q-parallel>`
  #. :ref:`Are zfp arrays and other data structures thread-safe? <q-thread-safety>`
  #. :ref:`Why does parallel compression performance not match my expectations? <q-omp-perf>`
  #. :ref:`Why are compressed arrays so slow? <q-1d-speed>`
  #. :ref:`Do compressed arrays use reference counting? <q-ref-count>`
  #. :ref:`How large a buffer is needed for compressed storage? <q-max-size>`
  #. :ref:`How can I print array values? <q-printf>`
  #. :ref:`What is known about zfp compression errors? <q-err-dist>`
  #. :ref:`Why are zfp blocks 4 * 4 * 4 values? <q-block-size>`
  #. :ref:`Can zfp (de)compress a single array in chunks? <q-chunked>`

-------------------------------------------------------------------------------

.. _q-layout:

Q0: *Do zfp arrays use C or Fortran order?*

*This is such an important question that we added it as question zero to our
FAQ, but do not let this C'ism fool you.*

A: |zfp| :ref:`compressed-array classes <arrays>` and uncompressed
:ref:`fields <field>` assume that the leftmost index varies fastest, which
often is referred to as Fortran order.  By convention, |zfp| uses *x* (or *i*)
to refer to the leftmost index, then *y* (or *j*), and so on.

.. warning::
  It is critical that the order of dimensions is specified correctly to
  achieve good compression and accuracy.  If the order of dimensions is
  transposed, |zfp| will still compress the data, but with no indication
  that the order was wrong.  Compression ratio and/or accuracy will likely
  suffer significantly, however.  Please see
  :ref:`this section <p-dimensions>` for further discussion.

In C order, the rightmost index varies fastest (e.g., *x* in
:code:`arr[z][y][x]`), meaning that if we increment the rightmost index we
move to the next consecutive address in memory.  If an uncompressed array,
:code:`arr`, is stored in C order, we would for compatibility with |zfp|
let *x* be the rightmost index in :code:`arr` but the leftmost index in the
compressed |zfp| array, :code:`zarr`, e.g.,::

  const size_t nx = 5;
  const size_t ny = 3;
  const size_t nz = 2;
  float arr[nz][ny][nx] = { ... };
  zfp::array3<float> zarr(nx, ny, nz, rate, &a[0][0][0]);

Then :code:`arr[z][y][x]` and :code:`zarr(x, y, z)` refer to the same element,
as do :code:`(&arr[0][0][0])[sx * x + sy * y + sz * z]` and
:code:`zarr[sx * x + sy * y + sz * z]`, where
::

  ptrdiff_t sx = &arr[0][0][1] - &arr[0][0][0]; // sx = 1
  ptrdiff_t sy = &arr[0][1][0] - &arr[0][0][0]; // sy = nx = 5
  ptrdiff_t sz = &arr[1][0][0] - &arr[0][0][0]; // sz = nx * ny = 15

Here *sx*, *sy*, and *sz* are the *strides* along the three dimensions,
with *sx* < *sy* < *sz*.

Of course, C vs. Fortran ordering matters only for multidimensional arrays
and when the array dimensions (*nx*, *ny*, *nz*) are not all equal.

Note that |zfp| :ref:`fields <field>` also support strides, which can be
used to represent more general layouts than C and Fortran order, including
non-contiguous storage, reversed dimensions via negative strides, and
other advanced layouts.  With the default strides, however, it is correct
to think of |zfp| as using Fortran order.

For uncompressed data stored in C order, one easily translates to |zfp|
Fortran order by reversing the order of dimensions or by specifying
appropriate :ref:`strides <field>`.  We further note that |zfp| provides
:ref:`nested views <nested_view>` of arrays that support C indexing syntax,
e.g., :code:`view[z][y][x]` corresponds to :code:`arr(x, y, z)`.

.. note::
  The |zfp| :ref:`NumPy interface <zfpy>` uses the strides of the NumPy array
  to infer the correct layout.  Although NumPy arrays use C order by default,
  |zfp| handles such arrays correctly regardless of their memory layout.  The
  actual order of dimensions for compressed storage are, however, reversed so
  that NumPy arrays in C order are traversed sequentially during compression.

Why does |zfp| use Fortran order when C is today a far more common language?
This choice is somewhat arbitrary yet has strong proponents in either camp,
similar to the preference between :ref:`little and big endian <q-portability>`
byte order.  We believe that a single 2D array storing an (*x*, *y*) image is
most naturally extended to a sequence of *nt* time-varying images by
*appending* (not prepending) a time dimension *t* as (*x*, *y*, *t*).  This
is the convention used in mathematics, e.g., we use (*x*, *y*) coordinates in
2D and (*x*, *y*, *z*) coordinates in 3D.  Using Fortran order, each time
slice, *t*, is still a 2D contiguous image, while C order
(:code:`arr[x][y][t]`) would suggest that appending the *t* dimension now
gives us *nx* 2D arrays indexed by (*y*, *t*), even though without the *t*
dimension the images would be indexed by (*x*, *y*).

-------------------------------------------------------------------------------

.. _q-vfields:

Q1: *Can zfp compress vector fields?*

I have a 2D vector field
::

  double velocity[ny][nx][2];

of dimensions *nx* |times| *ny*.  Can I use a 3D |zfp| array to store this as::

  array3d velocity(2, nx, ny, rate);

A: Although this could be done, zfp assumes that consecutive values are
related.  The two velocity components (*vx*, *vy*) are almost assuredly
independent and would not be correlated.  This will severely hurt the
compression rate or quality.  Instead, consider storing *vx* and *vy* as
two separate 2D scalar arrays::

  array2d vx(nx, ny, rate);
  array2d vy(nx, ny, rate);

or as
::

  array2d velocity[2] = {array2d(nx, ny, rate), array2d(nx, ny, rate)};

-------------------------------------------------------------------------------

.. _q-array2d:

Q2: *Should I declare a 2D array as zfp::array1d a(nx * ny, rate)?*

I have a 2D scalar field of dimensions *nx* |times| *ny* that I allocate as
::

  double* a = new double[nx * ny];

and index as
::

  a[x + nx * y]

Should I use a corresponding zfp array
::

  array1d a(nx * ny, rate);

to store my data in compressed form?

A: Although this is certainly possible, if the scalar field exhibits
coherence in both spatial dimensions, then far better results can be
achieved by using a 2D array::

  array2d a(nx, ny, rate);

Although both compressed arrays can be indexed as above, the 2D array can
exploit smoothness in both dimensions and improve the quality dramatically
for the same rate.

Since |zfp| 0.5.2, proxy pointers are also available that act much like
the flat :code:`double*`.

-------------------------------------------------------------------------------

.. _q-read:

Q3: *How can I initialize a zfp compressed array from disk?*

I have a large, uncompressed, 3D data set::

  double a[nz][ny][nx];

stored on disk that I would like to read into a compressed array.  This data
set will not fit in memory uncompressed.  What is the best way of doing this?

A: Using a |zfp| array::

  array3d a(nx, ny, nz, rate);

the most straightforward (but perhaps not best) way is to read one
floating-point value at a time and copy it into the array::

  for (size_t z = 0; z < nz; z++)
    for (size_t y = 0; y < ny; y++)
      for (size_t x = 0; x < nx; x++) {
        double f;
        if (fread(&f, sizeof(f), 1, file) == 1)
          a(x, y, z) = f;
        else {
          // handle I/O error
        }
      }

Note, however, that if the array cache is not large enough, then this may
compress blocks before they have been completely filled.  Therefore it is
recommended that the cache holds at least one complete layer of blocks,
i.e., (*nx* / 4) |times| (*ny* / 4) blocks in the example above.

To avoid inadvertent evictions of partially initialized blocks, it is better
to buffer four layers of *nx* |times| *ny* values each at a time, when
practical, and to completely initialize one block after another, which is
facilitated using |zfp|'s iterators::

  double* buffer = new double[nx * ny * 4];
  int zmin = -4;
  for (zfp::array3d::iterator it = a.begin(); it != a.end(); it++) {
    int x = it.i();
    int y = it.j();
    int z = it.k();
    if (z > zmin + 3) {
      // read another layer of blocks
      if (fread(buffer, sizeof(*buffer), nx * ny * 4, file) != nx * ny * 4) {
        // handle I/O error
      }
      zmin += 4;
    }
    a(x, y, z) = buffer[x + nx * (y + ny * (z - zmin))];
  }

Iterators have been available since |zfp| 0.5.2.

-------------------------------------------------------------------------------

.. _q-matrix:

Q4: *Can I use zfp to represent dense linear algebra matrices?*

A: Yes, but your mileage may vary.  Dense matrices, unlike smooth scalar
fields, rarely exhibit correlation between adjacent rows and columns.  Thus,
the quality or compression ratio may suffer.

For examples of dense linear solvers that use |zfp| for matrix storage,
see `STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/>`__
and `ButterflyPACK <https://portal.nersc.gov/project/sparse/butterflypack/>`__.

-------------------------------------------------------------------------------

.. _q-structured:

Q5: *Can zfp compress logically regular but geometrically irregular data?*

My data is logically structured but irregularly sampled, e.g., it is
rectilinear, curvilinear, or Lagrangian, or uses an irregular spacing of
quadrature points.  Can I still use zfp to compress it?

A: Yes, as long as the data is (or can be) represented as a logical
multidimensional array, though your mileage may vary.  |zfp| has been designed
for uniformly sampled data, and compression will in general suffer the more
irregular the sampling is.

-------------------------------------------------------------------------------

.. _q-valid:

Q6: *Does zfp handle infinities, NaNs,and subnormal floating-point numbers?*

A: Yes, but only in :ref:`reversible mode <mode-reversible>`.

|zfp|'s lossy compression modes currently support only finite
floating-point values.  If a block contains a NaN or an infinity, undefined
behavior is invoked due to the C math function :c:func:`frexp` being
undefined for non-numbers.  Subnormal numbers are, however, handled correctly.

-------------------------------------------------------------------------------

.. _q-missing:

Q7: *Can zfp handle data with some missing values?*

My data has some missing values that are flagged by very large numbers, e.g.
1e30.  Is that OK?

A: Although all finite numbers are "correctly" handled, such large sentinel
values are likely to pollute nearby values, because all values within a block
are expressed with respect to a common largest exponent.  The presence of
very large values may result in complete loss of precision of nearby, valid
numbers.  Currently no solution to this problem is available, but future
versions of |zfp| will likely support a bit mask to tag values that should be
excluded from compression.

-------------------------------------------------------------------------------

.. _q-integer:

Q8: *Can I use zfp to store integer data?*

Can I use zfp to store integer data such as 8-bit quantized images or 16-bit
digital elevation models?

A: Yes (as of version 0.4.0), but the data has to be promoted to 32-bit signed
integers first.  This should be done one block at a time using an appropriate
:code:`zfp_promote_*_to_int32` function call (see :ref:`ll-utilities`).  Future
versions of |zfp| may provide a high-level interface that automatically
performs promotion and demotion.

Note that the promotion functions shift the low-precision integers into the
most significant bits of 31-bit (not 32-bit) integers and also convert unsigned
to signed integers.  Do use these functions rather than simply casting 8-bit
integers to 32 bits to avoid wasting compressed bits to encode leading zeros.
Moreover, in fixed-precision mode, set the precision relative to the precision
of the (unpromoted) source data.

As of version 0.5.1, integer data is supported both by the low-level API and
high-level calls :c:func:`zfp_compress` and :c:func:`zfp_decompress`.

-------------------------------------------------------------------------------

.. _q-int32:

Q9: *Can I compress 32-bit integers using zfp?*

I have some 32-bit integer data.  Can I compress it using |zfp|'s 32-bit
integer support?

A: Yes, this can safely be done in :ref:`reversible mode <mode-reversible>`.

In other (lossy) modes, the answer depends.
|zfp| compression of 32-bit and 64-bit integers requires that each
integer *f* have magnitude \|\ *f*\ \| < 2\ :sup:`30` and
\|\ *f*\ \| < 2\ :sup:`62`, respectively.  To handle signed integers that
span the entire range |minus|\ 2\ :sup:`31` |leq| x < 2\ :sup:`31`, or
unsigned integers 0 |leq| *x* < 2\ :sup:`32`, the data has to be promoted to
64 bits first.

As with floating-point data, the integers should ideally represent a
quantized continuous function rather than, say, categorical data or set of
indices.  Depending on compression settings and data range, the integers may
or may not be losslessly compressed.  If fixed-precision mode is used, the
integers may be stored at less precision than requested.
See :ref:`Q21 <q-lossless>` for more details on precision and lossless
compression.

-------------------------------------------------------------------------------

.. _q-overrun:

Q10: *Why does zfp corrupt memory if my allocated buffer is too small?*

Why does |zfp| corrupt memory rather than return an error code if not enough
memory is allocated for the compressed data?

A: This is for performance reasons.  |zfp| was primarily designed for fast
random access to fixed-rate compressed arrays, where checking for buffer
overruns is unnecessary.  Adding a test for every compressed byte output
would significantly compromise performance.

One way around this problem (when not in fixed-rate mode) is to use the
:c:data:`maxbits` parameter in conjunction with the maximum precision or
maximum absolute error parameters to limit the size of compressed blocks.
Finally, the function :c:func:`zfp_stream_maximum_size` returns a conservative
buffer size that is guaranteed to be large enough to hold the compressed data
and the optional header.

-------------------------------------------------------------------------------

.. index::
   single: Endianness
.. _q-portability:

Q11: *Are zfp compressed streams portable across platforms?*

Are |zfp| compressed streams portable across platforms?  Are there, for
example, endianness issues?

A: Yes, |zfp| can write portable compressed streams.  To ensure portability
across different endian platforms, the bit stream must however be written
in increments of single bytes on big endian processors (e.g., PowerPC, SPARC),
which is achieved by compiling |zfp| with an 8-bit (single-byte) word size::

  -DBIT_STREAM_WORD_TYPE=uint8

See :c:macro:`BIT_STREAM_WORD_TYPE`.  Note that on little endian processors
(e.g., Intel x86-64 and AMD64), the word size does not affect the bit stream
produced, and thus the default word size may be used.  By default, |zfp| uses
a word size of 64 bits, which results in the coarsest rate granularity but
fastest (de)compression.  If cross-platform portability is not needed, then
the maximum word size is recommended (but see also :ref:`Q12 <q-granularity>`).

When using 8-bit words, |zfp| produces a compressed stream that is byte order
independent, i.e., the exact same compressed sequence of bytes is generated
on little and big endian platforms.  When decompressing such streams,
floating-point and integer values are recovered in the native byte order of
the machine performing decompression.  The decompressed values can be used
immediately without the need for byte swapping and without having to worry
about the byte order of the computer that generated the compressed stream.

Finally, |zfp| assumes that the floating-point format conforms to IEEE 754.
Issues may arise on architectures that do not support IEEE floating point.

-------------------------------------------------------------------------------

.. _q-granularity:

Q12: *How can I achieve finer rate granularity?*

A: For *d*-dimensional data, |zfp| supports a rate granularity of 1 / |4powd|
bits, i.e., the rate can be specified in increments of a fraction of a bit.
Such fine rate selection is always available for sequential compression
(e.g., when calling :c:func:`zfp_compress`).

Unlike in sequential compression, |zfp|'s
:ref:`read-write compressed-array classes <array_classes>` require
random-access writes, which are supported only at the granularity of whole
words.  By default, a word is 64 bits, which gives a rate granularity of
64 / |4powd| in *d* dimensions, i.e., 16 bits in 1D, 4 bits in 2D, 1 bit
in 3D, and 0.25 bits in 4D.
:ref:`Read-only compressed arrays <carray_classes>` support the same fine
granularity as sequential compression.

To achieve finer granularity, build |zfp| with a smaller (but as large as
possible) stream word size, e.g.::

  -DBIT_STREAM_WORD_TYPE=uint8

gives the finest possible granularity, but at the expense of (de)compression
speed.  See :c:macro:`BIT_STREAM_WORD_TYPE`.

-------------------------------------------------------------------------------

.. _q-progressive:

Q13: *Can I generate progressive zfp streams?*

A: Yes, but it requires some coding effort.  There is currently no high-level
support for progressive |zfp| streams.  To implement progressive fixed-rate
streams, the fixed-length bit streams should be interleaved among the blocks
that make up an array.  For instance, if a 3D array uses 1024 bits per block,
then those 1024 bits could be broken down into, say, 16 pieces of 64 bits
each, resulting in 16 discrete quality settings.  By storing the blocks
interleaved such that the first 64 bits of all blocks are contiguous,
followed by the next 64 bits of all blocks, etc., one can achieve progressive
decompression by setting the :c:member:`zfp_stream.maxbits` parameter (see
:c:func:`zfp_stream_set_params`) to the number of bits per block received so
far.

To enable interleaving of blocks, |zfp| must first be compiled with::

  -DBIT_STREAM_STRIDED

to enable strided bit stream access.  In the example above, if the stream
word size is 64 bits and there are *n* blocks, then::

  stream_set_stride(stream, m, n);

implies that after every *m* 64-bit words have been decoded, the bit stream
is advanced by *m* |times| *n* words to the next set of m 64-bit words
associated with the block.

-------------------------------------------------------------------------------

.. _q-init:

Q14: *How do I initialize the decompressor?*

A: The :c:type:`zfp_stream` and :c:type:`zfp_field` objects usually need to
be initialized with the same values as they had during compression (but see
:ref:`Q15 <q-same>` for exceptions).
These objects hold the compression mode and parameters, and field data like
the scalar type and dimensions.  By default, these parameters are not stored
with the compressed stream (the "codestream") and prior to |zfp| 0.5.0 had to
be maintained separately by the application.

Since version 0.5.0, functions exist for reading and writing a 12- to 19-byte
header that encodes compression and field parameters.  For applications that
wish to embed only the compression parameters, e.g., when the field dimensions
are already known, there are separate functions that encode and decode this
information independently.

-------------------------------------------------------------------------------

.. _q-same:

Q15: *Must I use the same parameters during compression and decompression?*

A: Usually, but there are exceptions.  When decompressing one block at a time
using the :ref:`low-level API <ll-api>`, it is possible to use more tightly
constrained :c:type:`zfp_stream` parameters during decompression than were
used during compression.  For instance, one may use a smaller
:c:member:`zfp_stream.maxbits`, smaller :c:member:`zfp_stream.maxprec`, or
larger :c:member:`zfp_stream.minexp` during decompression to process fewer
compressed bits than are stored, and to decompress the array more quickly at
a lower precision.  This may be useful in situations where the precision and
accuracy requirements are not known a priori, thus forcing conservative
settings during compression, or when the compressed stream is used for
multiple purposes.  For instance, visualization usually has less stringent
precision requirements than quantitative data analysis.  This feature of
decompressing to a lower precision is particularly useful when the stream is
stored progressively (see :ref:`Q13 <q-progressive>`).

Note, however, that when doing so, the caller must manually fast-forward
the stream (using :c:func:`stream_rseek`) to the beginning of the next block
before decompressing it, which may require extra bookkeeping.

Also note that one may not use less constrained parameters during
decompression, e.g., one cannot ask for more than
:c:member:`zfp_stream.maxprec` bits of precision when decompressing.
Furthermore, the parameters must agree between compression and decompression
when calling the high-level API function :c:func:`zfp_decompress`.

With regards to the :c:type:`zfp_field` struct passed to
:c:func:`zfp_compress` and :c:func:`zfp_decompress`, field dimensions must
generally match between compression and decompression, though see
:ref:`Q32 <q-chunked>` on chunked (de)compression.  Strides, however, need
not match; see :ref:`Q16 <q-strides>`.  Additionally, the scalar type,
:c:type:`zfp_type`, must match.  For example, float arrays currently have a
compressed representation different from compressed double arrays due to
differences in exponent width.  It is not possible to compress a double array
and then decompress (demote) the result to floats, for instance.  Future
versions of the |zfp| codec may use a unified representation that does allow
this.

By default, compression parameters and array metadata are not stored in the
compressed stream, as often such information is recorded separately.
However, the user may optionally record both compression parameters and array
metadata in a header at the beginning of the compressed stream; see
:c:func:`zfp_write_header`, :c:func:`zfp_read_header`, and further discussion
:ref:`here <field-match>`.

-------------------------------------------------------------------------------

.. _q-strides:

Q16: *Do strides have to match during compression and decompression?*

A: No.  For instance, a 2D vector field::

  float in[ny][nx][2];

could be compressed as two scalar fields with strides *sx* = 2,
*sy* = 2 |times| *nx*, and with pointers :code:`&in[0][0][0]` and
:code:`&in[0][0][1]` to the first value of each scalar field.  These two
scalar fields can later be decompressed as non-interleaved fields::

  float out[2][ny][nx];

using strides *sx* = 1, *sy* = *nx* and pointers :code:`&out[0][0][0]`
and :code:`&out[1][0][0]`.

Another use case is when a compressed array is to be decompressed into a
larger surrounding array.  For example, a 3D subarray with dimensions
*mx* |times| *my* |times| *mz* may be decompressed into a larger array
with dimensions *nx* |times| *ny* |times| *nz*, with *mx* |leq| *nx*,
*my* |leq| *ny*, *mz* |leq| *nz*.  This can be achieved by setting the strides
to *sx* = 1, *sy* = *nx*, *sz* = *nx* |times| *ny* upon decompression (using
:c:func:`zfp_field_set_stride_3d`), while specifying *mx*, *my*, and *mz* as
the field dimensions (using :c:func:`zfp_field_3d` or
:c:func:`zfp_field_set_size_3d`).  In this case, one may also wish to offset
the decompressed subarray to (*ox*, *oy*, *oz*) within the larger array
using::

  float* data = new float[nx * ny * nz];
  pointer = data + ox * sx + oy * sy + oz * sz

where *data* specifies the beginning of the larger array.  *pointer* rather
than *data* would then be passed to :c:func:`zfp_field_3d` or
:c:func:`zfp_field_set_pointer` before calling :c:func:`zfp_decompress`.

.. note::
  Strides are a property of the in-memory layout of an uncompressed array
  and have no meaning with respect to the compressed bit stream representation
  of the array.  As such, |zfp| provides no mechanism for storing information
  about the original strides in the compressed stream.  If strides are to be
  retained upon decompression, then the user needs to record them as auxiliary
  metadata and initialize :c:type:`zfp_field` with them.

-------------------------------------------------------------------------------

.. _q-tolerance:

Q17: *Why does zfp sometimes not respect my error tolerance?*

A: First, |zfp| does not support
:ref:`fixed-accuracy mode <mode-fixed-accuracy>` for integer data and
will ignore any tolerance requested via :c:func:`zfp_stream_set_accuracy`
or associated :ref:`expert mode <mode-expert>` parameter settings.  So this
FAQ pertains to floating-point data only.

The short answer is that, given finite precision, the |zfp| and IEEE
floating-point number systems represent distinct subsets of the reals
(or, in case of |zfp|, blocks of reals).  Although these subsets have
significant overlap, they are not equal.  Consequently, there are some
combinations of floating-point values that |zfp| cannot represent exactly;
conversely, there are some |zfp| blocks that cannot be represented exactly
as IEEE floating point.  If the user-specified tolerance is smaller than the
difference between the IEEE floating-point representation to be compressed
and its closest |zfp| representation, then the tolerance necessarily will
be violated (except in :ref:`reversible mode <mode-reversible>`).  In
practice, absolute tolerances have to be extremely small relative to the
numbers being compressed for this issue to occur, however.

Note that this issue is not particular to |zfp| but occurs in the conversion
between any two number systems of equal precision; we may just as well fault
IEEE floating point for not being able to represent all |zfp| blocks
accurately enough!  By analogy, not all 32-bit integers can be represented
exactly in 32-bit floating point.  The integer 123456789 is one example; the
closest float is 123456792.  And, obviously, not all floats (e.g., 0.5) can
be represented exactly as integers.

To further demonstrate this point, let us consider a concrete example.  |zfp|
does not store each floating-point scalar value independently but represents
a group of values (4, 16, 64, or 256 values, depending on dimensionality) as
linear combinations like averages by evaluating arithmetic expressions.
Just like in uncompressed IEEE floating-point arithmetic, both representation
error and roundoff error in the least significant bit(s) often occur.

To illustrate this, consider compressing the following 1D array of four
floats
::

  float f[4] = { 1, 1e-1, 1e-2, 1e-3 };

using the |zfp| command-line tool::

  zfp -f -1 4 -a 0 -i input.dat -o output.dat

In spite of an error tolerance of zero, the reconstructed values are::

  float g[4] = { 1, 1e-1, 9.999998e-03, 9.999946e-04 };

with a (computed) maximum error of 5.472e-9.  Because f[3] = 1e-3 can only
be approximately represented in radix-2 floating-point, the actual error
is even smaller: 5.424e-9.  This reconstruction error is primarily due to
|zfp|'s block-floating-point representation, which expresses the four values
in a block relative to a single, common binary exponent.  Such exponent
alignment occurs also in regular IEEE floating-point operations like addition.
For instance,
::

  float x = (f[0] + f[3]) - 1;

should of course result in :code:`x = f[3] = 1e-3`, but due to exponent
alignment a few of the least significant bits of f[3] are lost in the
rounded result of the addition, giving :code:`x = 1.0000467e-3` and a
roundoff error of 4.668e-8.  Similarly,
::

  float sum = f[0] + f[1] + f[2] + f[3];

should return :code:`sum = 1.111`, but is computed as 1.1110000610.  Moreover,
the value 1.111 cannot even be represented exactly in (radix-2) floating-point;
the closest float is 1.1109999.  Thus the computed error
::

  float error = sum - 1.111f;

which itself has some roundoff error, is 1.192e-7.

*Phew*!  Note how the error introduced by |zfp| (5.472e-9) is in fact one to
two orders of magnitude smaller than the roundoff errors (4.668e-8 and
1.192e-7) introduced by IEEE floating point in these computations.  This lower
error is in part due to |zfp|'s use of 30-bit significands compared to IEEE's
24-bit single-precision significands.  Note that data sets with a large dynamic
range, e.g., where adjacent values differ a lot in magnitude, are more
susceptible to representation errors.

The moral of the story is that error tolerances smaller than machine epsilon
(relative to the data range) cannot always be satisfied by |zfp|.  Nor are such
tolerances necessarily meaningful for representing floating-point data that
originated in floating-point arithmetic expressions, since accumulated
roundoff errors are likely to swamp compression errors.  Because such
roundoff errors occur frequently in floating-point arithmetic, insisting on
lossless compression on the grounds of accuracy is tenuous at best.

-------------------------------------------------------------------------------

.. _q-rate:

Q18: *Why is the actual rate sometimes not what I requested?*

A: In principle, |zfp| allows specifying the size of a compressed block in
increments of single bits, thus allowing very fine-grained tuning of the
bit rate.  There are, however, cases when the desired rate does not exactly
agree with the effective rate, and users are encouraged to check the return
value of :c:func:`zfp_stream_set_rate`, which gives the actual rate.

There are several reasons why the requested rate may not be honored.  First,
the rate is specified in bits/value, while |zfp| always represents a block
of |4powd| values in *d* dimensions, i.e., using
*N* = |4powd| |times| *rate* bits.  *N* must be an integer number of bits,
which constrains the actual rate to be a multiple of 1 / |4powd|.  The actual
rate is computed by rounding |4powd| times the desired rate.

Second, if the array dimensions are not multiples of four, then |zfp| pads the
dimensions to the next higher multiple of four.  Thus, the total number of
bits for a 2D array of dimensions *nx* |times| *ny* is computed in terms of
the number of blocks *bx* |times| *by*::

  bitsize = (4 * bx) * (4 * by) * rate

where *nx* |leq| 4 |times| bx < *nx* + 4 and
*ny* |leq| 4 |times| *by* < *ny* + 4.  When amortizing bitsize over the
*nx* |times| *ny* values, a slightly higher rate than requested may result.

Third, to support updating compressed blocks, as is needed by |zfp|'s
compressed array classes, the user may request write random access to the
fixed-rate stream.  To support this, each block must be aligned on a stream
word boundary (see :ref:`Q12 <q-granularity>`), and therefore the rate when
write random access is requested must be a multiple of *wordsize* / |4powd|
bits.  By default *wordsize* = 64 bits.  Even when write random access is
not requested, the compressed stream is written in units of *wordsize*.
Hence, once the stream is flushed, either by a :c:func:`zfp_compress` or
:c:func:`zfp_stream_flush` call, to output any buffered bits, its size
will be a multiple of *wordsize* bits.

Fourth, for floating-point data, each block must hold at least the common
exponent and one additional bit, which places a lower bound on the rate.

Finally, the user may optionally include a header with each array.  Although
the header is small, it must be accounted for in the rate.  The function
:c:func:`zfp_stream_maximum_size` conservatively includes space for a header,
for instance.

Aside from these caveats, |zfp| is guaranteed to meet the exact rate specified.

-------------------------------------------------------------------------------

.. _q-inplace:

Q19: *Can zfp perform compression in place?*

A: Because the compressed data tends to be far smaller than the uncompressed
data, it is natural to ask if the compressed stream can overwrite the
uncompressed array to avoid having to allocate separate storage for the
compressed stream.  |zfp| does allow for the possibility of such in-place
compression, but with several caveats and restrictions:

  1. A bitstream must be created whose buffer points to the beginning of
     uncompressed (and to be compressed) storage.

  2. The array must be compressed using |zfp|'s low-level API.  In particular,
     the data must already be partitioned and organized into contiguous blocks
     so that all values of a block can be pulled out once and then replaced
     with the corresponding shorter compressed representation.

  3. No one compressed block can occupy more space than its corresponding
     uncompressed block so that the not-yet compressed data is not overwritten.
     This is usually easily accomplished in fixed-rate mode, although the
     expert interface also allows guarding against this in all modes using the
     :c:member:`zfp_stream.maxbits` parameter.  This parameter should be set to
     :code:`maxbits = 4^d * sizeof(type) * 8`, where *d* is the array
     dimensionality (1, 2, 3, or 4) and where *type* is the scalar type of the
     uncompressed data.

  4. No header information may be stored in the compressed stream.

In-place decompression can also be achieved, but in addition to the above
constraints requires even more care:

  1. The data must be decompressed in reverse block order, so that the last
     block is decompressed first to the end of the block array.  This requires
     the user to maintain a pointer to uncompressed storage and to seek via
     :c:func:`stream_rseek` to the proper location in the compressed stream
     where the block is stored.

  2. The space allocated to the compressed stream must be large enough to
     also hold the uncompressed data.

An :ref:`example <ex-inplace>` is provided that shows how in-place compression
can be done.

-------------------------------------------------------------------------------

.. _q-relerr:

Q20: *Can zfp bound the point-wise relative error?*

A: Yes, but with some caveats.  First, we define the relative error in a value
*f* approximated by *g* as \|\ *f* - *g*\ \| / \|\ *f*\ \|, which converges to
\|\ log(*f* / *g*)\ \| = \|\ log(*f*) - \ log(*g*)\| as *g* approaches *f*,
where log(*f*) denotes the natural logarithm of *f*.
Below, we discuss three strategies for relative error control that may be
applicable depending on the properties of the underlying floating-point data.

If all floating-point values to be compressed are normalized, i.e., with no
nonzero subnormal values smaller in magnitude than
2\ :sup:`-126` |approx| 10\ :sup:`-38` (for floats) or
2\ :sup:`-1022` |approx| 10\ :sup:`-308` (for doubles), then the relative error
can be bounded using |zfp|'s :ref:`expert mode <mode-expert>` settings by
invoking :ref:`reversible mode <mode-reversible>`.  This is achieved by
truncating (zeroing) some number of least significant bits of all
floating-point values and then losslessly compressing the result.  The
*q* least significant bits of *n*-bit floating-point numbers (*n* = 32
for floats and *n* = 64 for doubles) are truncated by |zfp| by specifying a
maximum precision of *p* = *n* |minus| *q*.  The resulting point-wise relative
error is then at most 2\ :sup:`q - 23` (for floats) or 2\ :sup:`q - 52`
(for doubles).

.. note::
  For large enough *q*, floating-point exponent bits will be discarded,
  in which case the bound no longer holds, but then the relative error
  is already above 100%.  Also, as mentioned, the bound does not hold
  for subnormals; however, such values are likely too small for relative
  errors to be meaningful.

To bound the relative error, set the expert mode parameters to::

  minbits = 0
  maxbits = 0
  maxprec = p
  minexp = ZFP_MIN_EXP - 1 = -1075

For example, using the |zfpcmd| command-line tool, set the parameters using
:option:`-c` :code:`0 0 p -1075`.

Note that while the above approach respects the error bound when the
above conditions are met, it uses |zfp| for a purpose it was not designed
for, and the compression ratio may not be competitive with those obtained
using compressors designed to bound the relative error.

Other forms of relative error control can be achieved using |zfp|'s lossy
compression modes.  In :ref:`fixed-accuracy mode <mode-fixed-accuracy>`,
the *absolute error* \|\ *f* - *g*\ \| is bounded by a user-specified error
tolerance.  For a field whose values are all positive (or all negative), we
may pre-transform values by taking the natural logarithm, replacing
each value *f* with log(*f*) before compression, and then exponentiating
values after decompression.  This ensures that
\|\ log(*f*) - log(*g*)\ \| = \|\ log(*f* / *g*)\ \| is bounded.  (Note,
however, that many implementations of the math library make no guarantees
on the accuracy of the logarithm function.)  For fields whose values are
signed, an approximate bound can be achieved by using
log(*f*) |approx| asinh(*f* / 2), where asinh is the inverse of the
hyperbolic sine function, which is defined for both positive and negative
numbers.  One benefit of this approach is that it de-emphasizes the
importance of relative errors for small values that straddle zero, where
relative errors rarely make sense, e.g., because of round-off and other
errors already present in the data.

Finally, in :ref:`fixed-precision mode <mode-fixed-precision>`, the
precision of |zfp| transform coefficients is fixed, resulting in an error
that is no more than a constant factor of the largest (in magnitude)
value, *fmax*, within the same |zfp| block.  This can be thought of as a
weaker version of relative error, where the error is measured relative
to values in a local neighborhood.

In fixed-precision mode, |zfp| cannot bound the point-wise relative error
due to its use of a block-floating-point representation, in which all
values within a block are represented in relation to a single common
exponent.  For a high enough dynamic range within a block, there may
simply not be enough precision available to guard against loss.  For
instance, a block containing the values 2\ :sup:`0` = 1 and 2\ :sup:`-n`
would require a precision of *n* + 3 bits to represent losslessly, and
|zfp| uses at most 64-bit integers to represent values.  Thus, if
*n* |geq| 62, then 2\ :sup:`-n` is replaced with 0, which is a 100%
relative error.  Note that such loss also occurs when, for instance,
2\ :sup:`0` and 2\ :sup:`-n` are added using floating-point arithmetic
(see also :ref:`Q17 <q-tolerance>`).

As alluded to, it is possible to bound the error relative to the largest
value, *fmax*, within a block, which if the magnitude of values does not
change too rapidly may serve as a reasonable proxy for point-wise relative
errors.

One might then ask if using |zfp|'s fixed-precision mode with *p* bits of
precision ensures that the block-wise relative error is at most
2\ :sup:`-p` |times| *fmax*.  This is, unfortunately, not the case, because
the requested precision, *p*, is ensured only for the transform coefficients.
During the inverse transform of these quantized coefficients the quantization
error may amplify.  That being said, it is possible to derive a bound on the
error in terms of *p* that would allow choosing an appropriate precision.
Such a bound is derived below.

Let
::

  emax = floor(log2(fmax))

be the largest base-2 exponent within a block.  For transform coefficient
precision, *p*, one can show that the maximum absolute error, *err*, is
bounded by::

  err <= k(d) * (2^emax / 2^p) <= k(d) * (fmax / 2^p)

Here *k*\ (*d*) is a constant that depends on the data dimensionality *d*::

  k(d) = 20 * (15/4)^(d-1)

so that in 1D, 2D, 3D, and 4D we have::

  k(1) = 20
  k(2) = 125
  k(3) = 1125/4
  k(4) = 16876/16

Thus, to guarantee *n* bits of accuracy in the decompressed data, we need
to choose a higher precision, *p*, for the transform coefficients::

  p(n, d) = n + ceil(log2(k(d))) = n + 2 * d + 3

so that
::

  p(n, 1) = n + 5
  p(n, 2) = n + 7
  p(n, 3) = n + 9
  p(n, 4) = n + 11

This *p* value should be used in the call to
:c:func:`zfp_stream_set_precision`.

Note, again, that some values in the block may have leading zeros when
expressed relative to 2\ :sup:`emax`, and these leading zeros are counted
toward the *n*-bit precision.  Using decimal to illustrate this, suppose
we used 4-digit precision for a 1D block containing these four values::

  -1.41421e+1 ~ -1.414e+1 = -1414 * (10^1 / 1000)
  +2.71828e-1 ~ +0.027e+1 =   +27 * (10^1 / 1000)
  +3.14159e-6 ~ +0.000e+1 =     0 * (10^1 / 1000)
  +1.00000e+0 ~ +0.100e+1 =  +100 * (10^1 / 1000)

with the values in the middle column aligned to the common base-10 exponent
+1, and with the values on the right expressed as scaled integers.  These
are all represented using four digits of precision, but some of those digits
are leading zeros.

-------------------------------------------------------------------------------

.. _q-lossless:

Q21: *Does zfp support lossless compression?*

A: Yes.  As of |zfp| |revrelease|, bit-for-bit lossless compression is
supported via the :ref:`reversible compression mode <mode-reversible>`.
This mode supports both integer and floating-point data.

In addition, it is sometimes possible to ensure lossless compression using
|zfp|'s fixed-precision and fixed-accuracy modes.  For integer data, |zfp|
can with few exceptions ensure lossless compression in
:ref:`fixed-precision mode <mode-fixed-precision>`.
For a given *n*-bit integer type (*n* = 32 or *n* = 64), consider compressing
*p*-bit signed integer data, with the sign bit counting toward the precision.
In other words, there are exactly 2\ :sup:`p` possible signed integers.  If
the integers are unsigned, then subtract 2\ :sup:`p-1` first so that they
range from |minus|\ 2\ :sup:`p-1` to 2\ :sup:`p-1` - 1.

Lossless integer compression in fixed-precision mode is achieved by first
promoting the *p*-bit integers to *n* - 1 bits (see :ref:`Q8 <q-integer>`)
such that all integer values fall in
[|minus|\ 2\ :sup:`30`, +2\ :sup:`30`), when *n* = 32, or in
[|minus|\ 2\ :sup:`62`, +2\ :sup:`62`), when *n* = 64.  In other words, the
*p*-bit integers first need to be shifted left by *n* - *p* - 1 bits.  After
promotion, the data should be compressed in zfp's fixed-precision mode using::

  q = p + 4 * d + 1

bits of precision to ensure no loss, where *d* is the data dimensionality
(1 |leq| d |leq| 4).  Consequently, the *p*-bit data can be losslessly
compressed as long as *p* |leq| *n* - 4 |times| *d* - 1.  The table below
lists the maximum precision *p* that can be losslessly compressed using 32-
and 64-bit integer types.

  = ==== ====
  d n=32 n=64
  = ==== ====
  1 27   59
  2 23   55
  3 19   51
  4 15   47
  = ==== ====

Although lossless compression is possible as long as the precision constraint
is met, the precision needed to guarantee no loss is generally much higher
than the precision intrinsic in the uncompressed data.  Therefore, we
recommend using the :ref:`reversible mode <mode-reversible>` when lossless
compression is desired.

The minimum precision, *q*, given above is often larger than what
is necessary in practice.  There are worst-case inputs that do require such
large *q* values, but they are quite rare.

The reason for expanded precision, i.e., why *q* > *p*, is that |zfp|'s
decorrelating transform computes averages of integers, and this transform is
applied *d* times in *d* dimensions.  Each average of two *p*-bit numbers
requires *p* + 1 bits to avoid loss, and each transform can be thought of
involving up to four such averaging operations.

For floating-point data, fully lossless compression with |zfp| usually
requires :ref:`reversible mode <mode-reversible>`, as the other compression
modes are unlikely to guarantee bit-for-bit exact reconstructions.  However,
if the dynamic range is low or varies slowly such that values
within a |4powd| block have the same or similar exponent, then the
precision gained by discarding the 8 or 11 bits of the common floating-point
exponents can offset the precision lost in the decorrelating transform.  For
instance, if all values in a block have the same exponent, then lossless
compression is obtained using
*q* = 26 + 4 |times| *d* |leq| 32 bits of precision for single-precision data
and *q* = 55 + 4 |times| *d* |leq| 64 bits of precision for double-precision
data.  Of course, the constraint imposed by the available integer precision
*n* implies that lossless compression of such data is possible only in 1D for
single-precision data and only in 1D and 2D for double-precision data.
Finally, to preserve special values such as negative zero, plus and minus
infinity, and NaNs, reversible mode is needed.

-------------------------------------------------------------------------------

.. _q-abserr:

Q22: *Why is my actual, measured error so much smaller than the tolerance?*

A: For two reasons.  The way |zfp| bounds the absolute error in
:ref:`fixed-accuracy mode <mode-fixed-accuracy>` is by keeping all transform
coefficient bits whose place value exceeds the tolerance while discarding the
less significant bits.  Each such bit has a place value that is a power of
two, and therefore the tolerance must first be rounded down to the next
smaller power of two, which itself will introduce some slack.  This possibly
lower, effective tolerance is returned by the
:c:func:`zfp_stream_set_accuracy` call.

Second, the quantized coefficients are then put through an inverse transform.
This linear transform will combine signed quantization errors that, in the
worst case, may cause them to add up and increase the error, even though the
average (RMS) error remains the same, i.e., some errors cancel while others
compound.  For *d*-dimensional data, *d* such inverse transforms are applied,
with the possibility of errors cascading across transforms.  To account for
the worst possible case, zfp has to conservatively lower its internal error
tolerance further, once for each of the *d* transform passes.

Unless the data is highly oscillatory or noisy, the error is not likely to
be magnified much, leaving an observed error in the decompressed data that
is much lower than the prescribed tolerance.  In practice, the observed
maximum error tends to be about 4-8 times lower than the error tolerance
for 3D data, while the difference is smaller for 2D and 1D data.

We recommend experimenting with tolerances and evaluating what error levels
are appropriate for each application, e.g., by starting with a low,
conservative tolerance and successively doubling it.  The distribution of
errors produced by |zfp| is approximately Gaussian (see
:ref:`Q30 <q-err-dist>`), so even if the maximum error may seem large at
an individual grid point, most errors tend to be much smaller and tightly
clustered around zero.

-------------------------------------------------------------------------------

.. _q-parallel:

Q23: *Are parallel compressed streams identical to serial streams?*

A: Yes, it matters not what execution policy is used; the final compressed
stream produced by :c:func:`zfp_compress` depends only on the uncompressed
data and compression settings.

To support future parallel decompression, in particular variable-rate
streams, it will be necessary to also store an index of where (at what
bit offset) each compressed block is stored in the stream.  Extensions to the
current |zfp| format are being considered to support parallel decompression.

Regardless, the execution policy and parameters such as number of threads
do not need to be the same for compression and decompression.

-------------------------------------------------------------------------------

.. _q-thread-safety:

Q24: *Are zfp's compressed arrays and other data structures thread-safe?*

A: Yes, compressed arrays can be made thread-safe; no, data structures
like :c:type:`zfp_stream` and :c:type:`bitstream` are not necessarily
thread-safe.  As of |zfp| |viewsrelease|, thread-safe read and write access
to compressed arrays via OpenMP threads is provided through the use of
:ref:`private views <private_immutable_view>`, although these come with
certain restrictions and requirements such as the need for the user to
enforce cache coherence.  Please see the documentation on
:ref:`views <views>` for further details.

As far as C objects, |zfp|'s parallel OpenMP compressor assigns one
:c:type:`zfp_stream` per thread, each of which uses its own private
:c:type:`bitstream`.  Users who wish to make parallel calls to |zfp|'s
:ref:`low-level functions <ll-api>` are advised to consult the source
files :file:`ompcompress.c` and :file:`parallel.c`.

Finally, the |zfp| API is thread-safe as long as multiple threads do not
simultaneously call API functions and pass the same :c:type:`zfp_stream`
or :c:type:`bitstream` object.

-------------------------------------------------------------------------------

.. _q-omp-perf:

Q25: *Why does parallel compression performance not match my expectations?*

A: |zfp| partitions arrays into chunks and assigns each chunk to an OpenMP
thread.  A chunk is a sequence of consecutive *d*-dimensional blocks, each
composed of |4powd| values.  If there are fewer chunks than threads, then
full processor utilization will not be achieved.

The number of chunks is by default set to the number of threads, but can
be modified by the user via :c:func:`zfp_stream_set_omp_chunk_size`.
One reason for using more chunks than threads is to provide for better
load balance.  If compression ratios vary significantly across the array,
then threads that process easy-to-compress blocks may finish well ahead
of threads in charge of difficult-to-compress blocks.  By breaking chunks
into smaller units, OpenMP is given the opportunity to balance the load
better (though the effect of using smaller chunks depends on OpenMP
thread scheduling).  If chunks are too small, however, then the overhead
of allocating and initializing chunks and assigning threads to them may
dominate.  Experimentation with chunk size may improve performance, though
chunks ought to be at least several hundred blocks each.

In variable-rate mode, compressed chunk sizes are not known ahead of time.
Therefore the compressed chunks must be concatenated into a single stream
following compression.  This task is performed sequentially on a single
thread, and will inevitably limit parallel efficiency.

Other reasons for poor parallel performance include compressing arrays
that are too small to offset the overhead of thread creation and
synchronization.  Arrays should ideally consist of thousands of blocks
to offset the overhead of setting up parallel compression.

-------------------------------------------------------------------------------

.. _q-1d-speed:

Q26: *Why are compressed arrays so slow?*

A: This is likely due to the use of a very small cache.  Prior to |zfp|
|csizerelease|, all arrays used two 'layers' of blocks as default cache
size, which is reasonable for 2D and higher-dimensional arrays (as long
as they are not too 'skinny').  In 1D, however, this implies that the
cache holds only two blocks, which is likely to cause excessive thrashing.

As of version |csizerelease|, the default cache size is roughly proportional
to the square root of the total number of array elements, regardless of
array dimensionality.  While this tends to reduce thrashing, we suggest
experimenting with larger cache sizes of at least a few kilobytes to ensure
acceptable performance.

Note that compressed arrays constructed with the
:ref:`default constructor <array_ctor_default>` will
have an initial cache size of only one block.  Therefore, users should call
:cpp:func:`array::set_cache_size` after :ref:`resizing <array_resize>`
such arrays to ensure a large enough cache.

Depending on factors such as rate, cache size, array access pattern,
array access primitive (e.g., indices vs. iterators), and arithmetic
intensity, we usually observe an application slow-down of 1-10x when
switching from uncompressed to compressed arrays.

-------------------------------------------------------------------------------

.. _q-ref-count:

Q27: *Do compressed arrays use reference counting?*

A: It is possible to reference compressed-array elements via proxy
:ref:`references <references>` and :ref:`pointers <pointers>`, through
:ref:`iterators <iterators>`, and through :ref:`views <views>`.  Such
indirect references are valid only during the lifetime of the underlying
array.  No reference counting and garbage collection is used to keep the
array alive if there are external references to it.  Such references
become invalid once the array is destructed, and dereferencing them will
likely lead to segmentation faults.

-------------------------------------------------------------------------------

.. _q-max-size:

Q28: *How large a buffer is needed for compressed storage?*

A: :c:func:`zfp_compress` requires that memory has already been allocated to
hold the compressed data.  But often the compressed size is data dependent
and not known a priori.  The function :c:func:`zfp_stream_maximum_size`
returns a buffer size that is guaranteed to be large enough.  This function,
which should be called *after* setting the desired compression mode and
parameters, computes the largest possible compressed data size based on the
current compression settings and array size.  Note that by the pigeonhole
principle, any (lossless) compressor must expand at least one input, so this
buffer size may be larger than the size of the uncompressed input data.
:c:func:`zfp_compress` returns the actual number of bytes of compressed
storage.

When compressing individual blocks using the :ref:`low-level API <ll-api>`,
it is useful to know the maximum number of bits that a compressed block
can occupy.  In addition to the :c:macro:`ZFP_MAX_BITS` macro, the following
table lists the maximum block size (in bits) for each scalar type, whether
:ref:`reversible mode <mode-reversible>` is used, and block dimensionality.
Note that these sizes are upper bounds that are independent of compression
parameter settings, which may further constrain the storage size.

  +--------+---------+-------+-------+-------+-------+
  | type   | rev.    |   1D  |   2D  |   3D  |   4D  |
  +========+=========+=======+=======+=======+=======+
  |        |         |   131 |   527 |  2111 |  8447 |
  | int32  +---------+-------+-------+-------+-------+
  |        | |check| |   136 |   532 |  2116 |  8452 |
  +--------+---------+-------+-------+-------+-------+
  |        |         |   140 |   536 |  2120 |  8456 |
  | float  +---------+-------+-------+-------+-------+
  |        | |check| |   146 |   542 |  2126 |  8462 |
  +--------+---------+-------+-------+-------+-------+
  |        |         |   259 |  1039 |  4159 | 16639 |
  | int64  +---------+-------+-------+-------+-------+
  |        | |check| |   265 |  1045 |  4165 | 16645 |
  +--------+---------+-------+-------+-------+-------+
  |        |         |   271 |  1051 |  4171 | 16651 |
  | double +---------+-------+-------+-------+-------+
  |        | |check| |   278 |  1058 |  4178 | 16658 |
  +--------+---------+-------+-------+-------+-------+

The function :c:func:`zfp_block_maximum_size` returns the block sizes encoded
in this table.

-------------------------------------------------------------------------------

.. _q-printf:

Q29: *How can I print array values?*

Consider the following seemingly reasonable piece of code::

  #include <cstdio>
  #include "zfp/array1.hpp"

  int main()
  {
    zfp::array1<double> a(100, 16.0);
    printf("%f\n", a[0]); // does not compile
    return 0;
  }

The compiler will complain about :code:`a[0]` being a non-POD object.  This
is because :code:`a[0]` is a :ref:`proxy reference <references>` object
rather than a :code:`double`.  To make this work, :code:`a[0]` must be
explicitly converted to :code:`double`, e.g., using a cast::

    printf("%f\n", (double)a[0]);

For similar reasons, one may not use :code:`scanf` to initialize the value
of :code:`a[0]` because :code:`&a[0]` is a :ref:`proxy pointer <pointers>`
object, not a :code:`double*`.  Rather, one must use a temporary variable,
e.g.
::

  double t;
  scanf("%lf", &t);
  a[0] = t;

Note that using :code:`iostream`, expressions like
::

  std::cout << a[0] << std::endl;

do work, but
::

  std::cin >> a[0];

does not.

-------------------------------------------------------------------------------

.. _q-err-dist:

Q30: *What is known about zfp compression errors?*

A: Significant effort has been spent on characterizing compression errors
resulting from |zfp|, as detailed in the following publications:

#. P. Lindstrom,
   "`Error Distributions of Lossy Floating-Point Compressors <https://www.osti.gov/servlets/purl/1526183>`__,"
   JSM 2017 Proceedings.
#. J. Diffenderfer, A. Fox, J. Hittinger, G. Sanders, P. Lindstrom,
   "`Error Analysis of ZFP Compression for Floating-Point Data <https://doi.org/10.1137/18M1168832>`__,"
   SIAM Journal on Scientific Computing, 2019.
#. D. Hammerling, A. Baker, A. Pinard, P. Lindstrom,
   "`A Collaborative Effort to Improve Lossy Compression Methods for Climate Data <https://doi.org/10.1109/DRBSD-549595.2019.00008>`__,"
   5th International Workshop on Data Analysis and Reduction for Big Scientific Data, 2019.
#. A. Fox, J. Diffenderfer, J. Hittinger, G. Sanders, P. Lindstrom.
   "`Stability Analysis of Inline ZFP Compression for Floating-Point Data in Iterative Methods <https://doi.org/10.1137/19M126904X>`__,"
   SIAM Journal on Scientific Computing, 2020.
#. P. Lindstrom, J. Hittinger, J. Diffenderfer, A. Fox, D. Osei-Kuffuor, J. Banks.
   "`ZFP: A Compressed Array Representation for Numerical Computations <https://doi.org/10.1177/10943420241284023>`__,"
   International Journal of High-Performance Computing Applications, 2024.
#. A. Fox, P. Lindstrom.
   "`Statistical Analysis of ZFP: Understanding Bias <https://doi.org/10.48550/arXiv.2407.01826>`__,"
   LLNL-JRNL-858256, Lawrence Livermore National Laboratory, 2024.

In short, |zfp| compression errors are roughly normally distributed as a
consequence of the central limit theorem, and can be bounded.  Because the
error distribution is normal and because the worst-case error is often much
larger than errors observed in practice, it is common that measured errors
are far smaller than the absolute error tolerance specified in
:ref:`fixed-accuracy mode <mode-fixed-accuracy>`
(see :ref:`Q22 <q-abserr>`).

It is known that |zfp| errors can be slightly biased and correlated (see
:numref:`zfp-rounding` and papers #3 and #6 above).  Recent work has been done
to combat such issues by supporting optional :ref:`rounding modes <rounding>`.

.. _zfp-rounding:
.. figure:: zfp-rounding.pdf
  :figwidth: 90 %
  :align: center
  :alt: "zfp rounding modes"

  |zfp| errors are normally distributed.  This figure illustrates the
  agreement between theoretical (lines) and observed (dots) error
  distributions (*X*, *Y*, *Z*, *W*) for 1D blocks.  Without proper rounding
  (left), errors are biased and depend on the relative location within a |zfp|
  block, resulting in errors not centered on zero.  With proper rounding
  (right), errors are both smaller and unbiased.

It is also known how |zfp| compression errors behave as a function of grid
spacing, *h*.  In particular, regardless of dimensionality, the compression
error *decreases* with finer grids (smaller *h*) for a given rate (i.e.,
fixed compressed storage size).  The |zfp| compression error decay is fast
enough that the corresponding error in partial derivative estimates based on
finite differences, which *increases* with smaller *h* when using conventional
floating point, instead *decreases* with finer grids when using |zfp|.  See
paper #5 for details.

-------------------------------------------------------------------------------

.. _q-block-size:

Q31: *Why are zfp blocks 4* |times| *4* |times| *4 values?*

One might ask why |zfp| uses *d*-dimensional blocks of |4powd| values and not
some other, perhaps configurable block size, *n*\ :sup:`d`.  There are several
reasons why *n* = 4 was chosen:

* For good performance, *n* should be an integer power of two so that indexing
  can be done efficiently using bit masks and shifts rather than more
  expensive division and modulo operations.  As nontrivial compression demands
  *n* > 1, possible choices for *n* are 2, 4, 8, 16, ...

* When *n* = 2, blocks are too small to exhibit significant redundancy; there
  simply is insufficient spatial correlation to exploit for sufficient data
  reduction.  Additionally, excessive software cache thrashing would likely
  occur for stencil computations, as even the smallest centered difference
  stencil would span more than one block.  Finally, per-block overhead in
  storage (e.g., shared exponent, bit offset) and computation (e.g., software
  cache lookup) could be amortized over only few values.  Such small blocks
  were immediately dismissed.

* When *n* = 8, blocks are too large, for several reasons:

  * Each uncompressed block occupies a large number of hardware cache lines.
    For example, a single 3D block of double-precision values would occupy
    4,096 bytes, which would represent a significant fraction of L1 cache.
    |zfp| reduces data movement in computations by ensuring that repeated
    accesses are to cached data rather than to main memory.

  * A generalization of the |zfp| :ref:`decorrelating transform <algorithm>`
    to *n* = 8 would require many more operations as well as "arbitrary"
    numerical constants in need of expensive multiplication instead of cheap
    bit shifts.  The number of operations in this more general case scales as
    *d* |times| *n*\ :sup:`d+1`.  For *d* = 4, *n* = 8, this implies
    2\ :sup:`17` = 131,072 multiplications and 114,688 additions per block.
    Contrast this with the algorithm optimized for *n* = 4, which uses only
    1,536 bit shifts and 2,560 additions or subtractions per 4D block.

  * The additional computational cost would also significantly increase the
    latency of decoding a single block or filling a pipeline of concurrently
    (de)compressed blocks, as in existing |zfp| hardware implementations.

  * The computational and cache storage overhead of accessing a single value
    in a block would be very large: 4,096 values in *d* = 4 dimensions would
    have to be decoded even if only one value were requested.

  * "Skinny" arrays would have to be padded to multiples of *n* = 8, which
    could introduce an unacceptable storage overhead.  For instance, a
    30 |times| 20 |times| 3 array of 1,800 values would be padded to
    32 |times| 24 |times| 8 = 6,144 values, an increase of about 3.4 times.
    In contrast, when *n* = 4, only 32 |times| 20 |times| 4 = 2,560 values
    would be needed, representing a 60% overhead.

  * The opportunity for data parallelism would be reduced by a factor of
    2\ :sup:`d` compared to using *n* = 4.  The finer granularity and larger
    number of blocks provided by *n* = 4 helps with load balancing and maps
    well to today's GPUs that can concurrently process thousands of blocks.

  * With blocks comprising as many as 8\ :sup:`4` = 4,096 values, register
    spillage would be substantial in GPU kernels for compression and
    decompression.

The choice *n* = 4 seems to be a sweet spot that well balances all of the
above factors.  Additionally, *n* = 4 has these benefits:

  * *n* = 4 admits a very simple lifted implementation of the decorrelating
    transform that can be performed using only integer addition, subtraction,
    and bit shifts.

  * *n* = 4 allows taking advantage of AVX/SSE instructions designed for
    vectors of length four, both in the (de)compression algorithm and
    application code.

  * For 2D and 3D data, a block is 16 and 64 values, respectively, which
    either equals or is close to the warp size on current GPU hardware.  This
    allows multiple cooperating threads to execute the same instruction on one
    value in 1-4 blocks (either during (de)compression or in the numerical
    application code).

  * Using a rate of 16 bits/value (a common choice for numerical computations),
    a compressed 3D block occupies 128 bytes, or 1-2 hardware cache lines on
    contemporary computers.  Hence, a fair number of *compressed* blocks can
    also fit in hardware cache.

-------------------------------------------------------------------------------

.. _q-chunked:

Q32: *Can zfp (de)compress a single array in chunks?*

Yes, but there are restrictions.

First, one can trivially partition any array into subarrays and (de)compress
those independently using separate matching :c:func:`zfp_compress` and
:c:func:`zfp_decompress` calls for each chunk.  Via subarray dimensions,
strides, and pointers into the larger array, one can thus (de)compress the
full array in pieces; see also :ref:`Q16 <q-strides>`.  This approach to
chunked (de)compression incurs no constraints on compression mode, compression
parameters, or array dimensions, though producer and consumer must agree on
chunk size.  This type of chunking is employed by the |zfp| HDF5 filter
`H5Z-ZFP <https://github.com/LLNL/H5Z-ZFP>`__ for I/O.

A more restricted form of chunked (de)compression is to produce (compress) or
consume (decompress) a single compressed stream for the whole array in chunks
in a manner compatible with producing/consuming the entire stream all at once.
Such chunked (de)compression divides the array into slabs along the slowest
varying dimension (e.g., along *z* for 3D arrays), (de)compresses one slab at
a time, and produces or consumes consecutive pieces of the sequential
compressed stream.  This approach, too, is possible, though only when these
requirements are met:

* The size of each chunk (except the last) must be a whole multiple of four
  along the slowest varying dimension; other dimensions are not subject to this
  constraint.  For example, a 3D array with *nz* = 120 can be (de)compressed
  in two or three equal-size chunks, but not four, since 120/2 = 60, and
  120/3 = 40 are both divisible by four, but 120/4 = 30 is not.  Other viable
  chunk sizes are 120/5 = 24, 120/6 = 20, 120/10 = 12, 120/15 = 8, and
  120/30 = 4.  Note that other chunk sizes may be possible by relaxing the
  constraint that they all be equal, as exploited by the
  :ref:`chunk <ex-chunk>` code example, e.g., *nz* = 120 can be partitioned
  into three chunks of size 32 and one of size 24.

  The reason for this requirement is that |zfp| always pads each compressed
  (sub)array to fill out whole blocks of size 4 in each dimension, and such
  interior padding would not occur if the whole array were compressed as a
  single chunk.

* The length of the compressed substream for each chunk must be a multiple of
  the :ref:`word size <word-size>`.  The reason for this is that each
  :c:func:`zfp_compress` and :c:func:`zfp_decompress` call aligns the stream
  on a word boundary upon completion.  One may avoid this requirement by using
  the low-level API, which does not automatically perform such alignment.

.. note::

  When using the :ref:`high-level API <hl-api>`, the requirement on stream
  alignment essentially limits chunked (de)compression to
  :ref:`fixed-rate mode <mode-fixed-rate>`, as it is the only one that can
  guarantee that the size of each compressed chunk is a multiple of the word
  size.  To support other compression modes, use the
  :ref:`low-level API <ll-api>`.

Chunked (de)compression requires the user to set the :c:type:`zfp_field`
dimensions to match the current chunk size and to set the
:ref:`field pointer <zfp_field_set>` to the beginning of each uncompressed
chunk before (de)compressing it.  The user may also have to position the
compressed stream so that it points to the beginning of each compressed
chunk.  See the :ref:`code example <ex-chunk>` for how one may implement
chunked (de)compression.

Note that the chunk size used for compression need not match the size used for
decompression; e.g., the array may be compressed in a single sweep but
decompressed in chunks, or vice versa.  Any combination of chunk sizes that
respect the above constraints is valid.

Chunked (de)compression makes it possible to perform, for example, windowed
streaming computations on smaller subsets of the decompressed array at a time,
i.e., without having to allocate enough space to hold the entire uncompressed
array.  It also can be useful for overlapping or interleaving computation with
(de)compression in a producer/consumer model.
