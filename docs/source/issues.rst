.. include:: defs.rst

.. _issues:

Troubleshooting
===============

This section is intended for troubleshooting problems with |zfp|, in case
any arise, and primarily focuses on how to correctly make use of |zfp|.  If
the decompressed data looks nothing like the original data, or if the
compression ratios obtained seem not so impressive, then it is very likely
that array dimensions or compression parameters have not been set correctly,
in which case this troubleshooting guide could help.

The problems addressed in this section include:

  #. :ref:`Is the data dimensionality correct? <p-dimensionality>`
  #. :ref:`Do the compressor and decompressor agree on the dimensionality? <p-agree>`
  #. :ref:`Have the "smooth" dimensions been identified? <p-smooth>`
  #. :ref:`Are the array dimensions correct? <p-dimensions>`
  #. :ref:`Are the array dimensions large enough? <p-large>`
  #. :ref:`Is the data logically structured? <p-structured>`
  #. :ref:`Is the data set embedded in a regular grid? <p-embedded>`
  #. :ref:`Have fill values, NaNs, and infinities been removed? <p-invalid>`
  #. :ref:`Is the byte order correct? <p-endian>`
  #. :ref:`Is the floating-point precision correct? <p-float-precision>`
  #. :ref:`Is the integer precision correct? <p-int-precision>`
  #. :ref:`Is the data provided to the zfp executable a raw binary array? <p-binary>`
  #. :ref:`Has the appropriate compression mode been set? <p-mode>`

-------------------------------------------------------------------------------

.. _p-dimensionality:

P1: *Is the data dimensionality correct?*

This is one of the most common problems.  First, make sure that |zfp| is given
the correct dimensionality of the data.  For instance, an audio stream is a
1D array, an image is a 2D array, and a volume grid is a 3D array, and a
time-varying volume is a 4D array.  Sometimes a data set is a discrete
collection of lower-dimensional objects.  For instance, a stack of unrelated
images (of the same size) could be represented in C as a 3D array::

  imstack[count][ny][nx]

but since in this case the images are unrelated, no correlation would be
expected along the third dimension---the underlying dimensionality of the data
is here two.  In this case, the images could be compressed one at a time, or
they could be compressed together by treating the array dimensions as::

  imstack[count * ny][nx]

Note that |zfp| partitions *d*-dimensional arrays into blocks of |4powd|
values.  If *ny* above is not a multiple of four, then some blocks of |4by4|
pixels will contain pixels from different images, which could hurt compression
and/or quality.  Still, this way of creating a single image by stacking multiple
images is far preferable over linearizing each image into a 1D signal, and
then compressing the images as::

  imstack[count][ny * nx]

This loses the correlation along the *y* dimension and further introduces
discontinuities unless *nx* is a multiple of four.

Similarly to the example above, a 2D vector field
::

  vfield[ny][nx][2]

could be declared as a 3D array, but the *x*- and *y*-components of the
2D vectors are likely entirely unrelated.  In this case, each component
needs to be compressed independently, either by rearranging the data
as two scalar fields::

  vfield[2][ny][nx]

or by using strides (see also FAQ :ref:`#1 <q-vfields>`).  Note that in all
these cases |zfp| will still compress the data, but if the dimensionality is
not correct then the compression ratio will suffer.

-------------------------------------------------------------------------------

.. _p-agree:

P2: *Do the compressor and decompressor agree on the dimensionality?*

Consider compressing a 3D array::

  double a[1][1][100]

with *nx* = 100, *ny* = 1, *nz* = 1, then decompressing the result to a 1D
array::

  double b[100]

with *nx* = 100.  Although the arrays *a* and *b* occupy the same amount of
memory and are in C laid out similarly, these arrays are not equivalent to
|zfp| because their dimensionalities differ.  |zfp| uses different CODECs
to (de)compress 1D, 2D, 3D, and 4D arrays, and the 1D decompressor expects a
compressed bit stream that corresponds to a 1D array.

What happens in practice in this case is that the array *a* is compressed
using |zfp|'s 3D CODEC, which first pads the array to
::

  double padded[4][4][100]

When this array is correctly decompressed using the 3D CODEC, the padded
values are generated but discarded.  |zfp|'s 1D decompressor, on the other
hand, expects 100 values, not 100 |times| 4 |times| 4 = 1600 values, and
therefore likely returns garbage.

-------------------------------------------------------------------------------

.. _p-smooth:

P3: *Have the "smooth" dimensions been identified?*

Closely related to :ref:`P1 <p-dimensionality>` above, some fields simply do
not vary smoothly along all dimensions, and |zfp| can do a good job
compressing only those dimensions that exhibit some coherence.  For instance,
consider a table of stock prices indexed by date and stock::

  price[stocks][dates]

One could be tempted to compress this as a 2D array, but there is likely
little to no correlation in prices between different stocks.  Each such
time series should be compressed independently as a 1D signal.

What about time-varying images like a video sequence?  In this case, it is
likely that there is correlation over time, and that the value of a single
pixel varies smoothly in time.  It is also likely that each image exhibits
smoothness along its two spatial dimensions.  So this can be treated as a
single, 3D data set.

How about time-varying volumes, such as
::

  field[nt][nz][ny][nx]

As of version |4drelease|, |zfp| supports compression of 4D arrays.  Since
all dimensions in this example are likely to be correlated, the 4D array
can be compressed directly.  Alternatively, the data could be organized by
the three "smoothest" dimensions and compressed as a 3D array.  Given the
organization above, the array could be treated as 3D::

  field[nt * nz][ny][nx]

Again, do **not** compress this as a 3D array with the *innermost*
dimensions unfolded::

  field[nt][nz][ny * nx]

-------------------------------------------------------------------------------

.. _p-dimensions:

P4: *Are the array dimensions correct?*

This is another common problem that seems obvious, but often the dimensions
are accidentally transposed.  Assuming that the smooth dimensions have been
identified, it is important that the dimensions are listed in the correct
order.  For instance, if the data (in C notation) is organized as::

  field[d1][d2][d3]

then the data is organized in memory (or on disk) with the d3 dimension varying
fastest, and hence *nx* = *d3*, *ny* = *d2*, *nz* = *d1* using the |zfp| naming
conventions for the dimensions, e.g., the :ref:`zfp executable <zfpcmd>` should
be invoked with::

  zfp -3 d3 d2 d1

in this case.  Things will go horribly wrong if |zfp| in this case is called
with *nx* = *d1*, *ny* = *d2*, *nz* = *d3*.  The entire data set will still
compress and decompress, but compression ratio and quality will likely suffer
greatly.  See :ref:`this FAQ <q-layout>` for more details.

-------------------------------------------------------------------------------

.. _p-large:

P5: *Are the array dimensions large enough?*

|zfp| partitions *d*-dimensional data sets into blocks of |4powd| values, e.g.,
in 3D a block consists of |4by4by4| values.  If the dimensions are not
multiples of four, then |zfp| will "pad" the array to the next larger multiple
of four.  Such padding can hurt compression.  In particular, if one or more of
the array dimensions are small, then the overhead of such padding could be
significant.

Consider compressing a collection of 1000 small 3D arrays::

  field[1000][5][14][2]

|zfp| would first logically pad this to a larger array::

  field[1000][8][16][4]

which is (8 |times| 16 |times| 4) / (5 |times| 14 |times| 2) ~ 3.66 times
larger.  Although such padding often compresses well, this still represents
a significant overhead.

If a large array has been partitioned into smaller pieces, it may be best to
reassemble the larger array.  Or, when possible, ensure that the sub-arrays
have dimensions that are multiples of four.

-------------------------------------------------------------------------------

.. _p-structured:

P6: *Is the data logically structured?*

|zfp| was designed for logically structured data, i.e., Cartesian grids.  It
works much like an image compressor does, which assumes that the data set is a
structured array of pixels, and it assumes that values vary reasonably smoothly
on average, just like natural images tend to contain large regions of uniform
color or smooth color gradients, like a blue sky, smoothly varying skin tones
of a human's face, etc.  Many data sets are not represented on a regular grid.
For instance, an array of particle *xyz* positions::

  points[count][3]

is a 2D array, but does not vary smoothly in either dimension.  Furthermore,
such unstructured data sets need not be organized in any particular order;
the particles could be listed in any arbitrary order.  One could attempt to
sort the particles, for example by the *x* coordinate, to promote smoothness,
but this would still leave the other two dimensions non-smooth.

Sometimes the underlying dimensions are not even known, and only the total
number of floating-point values is known.  For example, suppose we only knew
that the data set contained *n* = *count* |times| 3 values.  One might be
tempted to compress this using |zfp|'s 1-dimensional compressor, but once
again this would not work well.  Such abuse of |zfp| is much akin to trying
to compress an image using an audio compressor like mp3, or like compressing
an *n*-sample piece of music as an *n*-by-one sized image using an image
compressor like JPEG.  The results would likely not be very good.

Some data sets are logically structured but geometrically irregular.  Examples
include fields stored on Lagrangian meshes that have been warped, or on
spectral element grids, which use a non-uniform grid spacing.  |zfp| assumes
that the data has been regularly sampled in each dimension, and the more the
geometry of the sampling deviates from uniform, the worse compression gets.
Note that rectilinear grids with different but uniform grid spacing in each
dimension are fine.  If your application uses very non-uniform sampling, then
resampling onto a uniform grid (if possible) may be advisable.

Other data sets are "block structured" and consist of piecewise structured
grids that are "glued" together.  Rather than treating such data as
unstructured 1D streams, consider partitioning the data set into independent
(possibly overlapping) regular grids.

-------------------------------------------------------------------------------

.. _p-embedded:

P7: *Is the data set embedded in a regular grid?*

Some applications represent irregular geometry on a Cartesian grid, and leave
portions of the domain unspecified.  Consider, for instance, sampling the
density of the Earth onto a Cartesian grid.  Here the density for grid points
outside the Earth is unspecified.

In this case, |zfp| does best by initializing the "background field" to all
zeros.  In |zfp|'s :ref:`fixed-accuracy mode <mode-fixed-accuracy>`, any
"empty" block that consists of all zeros is represented using a single bit,
and therefore the overhead of representing empty space can be kept low.

-------------------------------------------------------------------------------

.. _p-invalid:

P8: *Have fill values, NaNs, and infinities been removed?*

It is common to signal unspecified values using what is commonly called a
"fill value," which is a special constant value that tends to be far out of
range of normal values.  For instance, in climate modeling the ocean
temperature over land is meaningless, and it is common to use a very large
temperature value such as 1e30 to signal that the temperature is undefined
for such grid points.

Very large fill values do not play well with |zfp|, because they both introduce
artificial discontinuities and pollute nearby values by expressing them all
with respect to the common largest exponent within their block.  Assuming
a fill value of 1e30, the value pi in the same block would be represented as::

  0.00000000000000000000000000000314159... * 1e30

Given finite precision, the small fraction would likely be replaced with zero,
resulting in complete loss of the actual value being stored.

Other applications use NaNs (special not-a-number values) or infinities as
fill values.  These are even more problematic, because they do not have a
defined exponent.  |zfp| relies on the C function :c:func:`frexp` to compute
the exponent of the largest (in magnitude) value within a block, but produces
unspecified behavior if that value is not finite.  

|zfp| currently has no independent mechanism for handling fill values.  Ideally
such special values would be signalled separately, e.g., using a bit mask, 
and then replaced with zeros to ensure that they both compress well and do
not pollute actual data.

-------------------------------------------------------------------------------

.. _p-endian:

P9: *Is the byte order correct?*

|zfp| generally works with the native byte order (e.g., little or big endian)
of the machine it is compiled on.  One needs only be concerned with byte order
when reading raw, binary data into the |zfp| executable, when exchanging
compressed files across platforms, and when varying the bit stream word size
on big endian machines (not common).  For instance, to compress a binary
double-precision floating-point file stored in big endian byte order on a
little endian machine, byte swapping must first be done.  For example, on
Linux and macOS, 8-byte doubles can be byte swapped using::

  objcopy -I binary -O binary --reverse-bytes=8 big.bin little.bin

See also FAQ :ref:`#11 <q-portability>` for more discussion of byte order.

-------------------------------------------------------------------------------

.. _p-float-precision:

P10: *Is the floating-point precision correct?*

Another obvious problem: Please make sure that |zfp| is told whether the data
to compress is an array of single- (32-bit) or double-precision (64-bit)
values, e.g., by specifying the :option:`-f` or :option:`-d` options to the
:program:`zfp` executable or by passing the appropriate :c:type:`zfp_type`
to the C functions.

-------------------------------------------------------------------------------

.. _p-int-precision:

P11: *Is the integer precision correct?*

|zfp| currently supports compression of 31- or 63-bit signed integers.  Shorter
integers (e.g., bytes, shorts) can be compressed but must first be promoted
to one of the longer types.  This should always be done using |zfp|'s functions
for :ref:`promotion and demotion <ll-utilities>`, which both perform bit
shifting and biasing to handle both signed and unsigned types.  It is not
sufficient to simply cast short integers to longer integers.  See also FAQs
:ref:`#8 <q-integer>` and :ref:`#9 <q-int32>`.

-------------------------------------------------------------------------------

.. _p-binary:

P12: *Is the data provided to the zfp executable a raw binary array?*

|zfp| expects that the input file is a raw binary array of integers or
floating-point values in the IEEE format, e.g., written to file using
:c:func:`fwrite`.  Do not hand |zfp| a text file containing ASCII
floating-point numbers.  Strip the file of any header information.
Languages like Fortran tend to store with the array its size.  No such
metadata may be embedded in the file.

-------------------------------------------------------------------------------

.. _p-mode:

P13: *Has the appropriate compression mode been set?*

|zfp| provides three different lossy
:ref:`modes of compression <modes>` that trade storage and accuracy,
plus one :ref:`lossless mode <mode-reversible>`.  In
fixed-rate mode, the user specifies the exact number of bits (often in
increments of a fraction of a bit) of compressed storage per value (but see
FAQ :ref:`#18 <q-rate>` for caveats).  From the user's perspective, this
seems a very desirable feature, since it provides for a direct mechanism for
specifying how much storage to use.  However, there is often a large quality
penalty associated with the fixed-rate mode, because each block of |4powd|
values is allocated the same number of bits.  In practice, the information
content over the data set varies significantly, which means that
easy-to-compress regions are assigned too many bits, while too few bits are
available to faithfully represent the more challenging-to-compress regions.
Although one of the unique features of |zfp|, its fixed-rate mode should
primarily be used only when random access to the data is needed.

|zfp| also provides a fixed-precision mode, where the user specifies how many
uncompressed significant bits to use to represent the floating-point fraction.
This precision may not be exactly what people might normally think of.  For
instance, the C float type is commonly referred to as 32-bit precision.
However, the sign bit and exponent account for nine of those bits and do
not contribute to the number of significant bits of precision.  Furthermore,
for normal numbers, IEEE uses a hidden implicit one bit, so most float values
actually have 24 bits of precision.  Furthermore, |zfp| uses a
block-floating-point representation with a single exponent per block,
which may cause some small values to have several leading zero bits and
therefore less precision than requested.  Thus, the effective precision
returned by |zfp| in its fixed-precision mode may in fact vary.  In practice,
the precision requested is only an upper bound, though typically at least one
value within a block has the requested precision.

|zfp| supports a fixed-accuracy mode, which except in rare
circumstances (see FAQ :ref:`#17 <q-tolerance>`) ensures that the absolute
error is bounded, i.e., the difference between any decompressed and original
value is at most the tolerance specified by the user (but usually several
times smaller).  Whenever possible, we recommend using this compression mode,
which depending on how easy the data is to compress results in the smallest
compressed stream that respects the error tolerance.

As of |zfp| |revrelease|, reversible (lossless) compression is available.
The amount of lossless reduction of floating-point data is usually quite
limited, however, especially for double-precision data.  Unless a bit-for-bit
exact reconstruction is needed, we strongly advocate the use of lossy
compression.

Finally, there is also an expert mode that allows the user to combine the
constraints of fixed rate, precision, and accuracy.  See the section on
:ref:`compression modes <modes>` for more details.
