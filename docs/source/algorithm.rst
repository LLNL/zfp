.. include:: defs.rst

.. _algorithm:

Algorithm
=========

|zfp| uses two different algorithms to support :ref:`lossy <algorithm-lossy>`
and :ref:`lossless <algorithm-lossless>` compression.  These algorithms are
described in detail below.

.. _algorithm-lossy:

Lossy Compression
-----------------

The |zfp| lossy compression scheme is based on the idea of breaking a
*d*-dimensional array into independent blocks of |4powd| values each,
e.g., |4by4by4| values in three dimensions.  Each block is
compressed/decompressed entirely independently from all other blocks.  In
this sense, |zfp| is similar to current hardware texture compression schemes
for image coding implemented on graphics cards and mobile devices.

The lossy compression scheme implemented in this version of |zfp| has evolved
from the method described in the :ref:`original paper <tvcg-paper>`, and can
conceptually be thought of as consisting of eight sequential steps (in
practice some steps are consolidated or exist only for illustrative
purposes):

  1. The *d*-dimensional array is partitioned into blocks of dimensions
     |4powd|.  If the array dimensions are not multiples of four, then
     blocks near the boundary are padded to the next multiple of four.  This
     padding is invisible to the application.

  2. The independent floating-point values in a block are converted to what
     is known as a block-floating-point representation, which uses a single,
     common floating-point exponent for all |4powd| values.  The effect of
     this conversion is to turn each floating-point value into a 31- or 63-bit
     signed integer.  If the values in the block are all zero or are smaller
     in magnitude than the fixed-accuracy tolerance (see below), then only a
     single bit is stored with the block to indicate that it is "empty" and
     expands to all zeros.  Note that the block-floating-point conversion and
     empty-block encoding are not performed if the input data is represented
     as integers rather than floating-point numbers.

  3. The integers are decorrelated using a custom, high-speed, near orthogonal
     transform similar to the discrete cosine transform used in JPEG image
     coding.  The transform exploits separability and is implemented
     efficiently in-place using the lifting scheme, requiring only
     2.5 *d* integer additions and 1.5 *d* bit shifts by one per integer in
     *d* dimensions.  If the data is "smooth," then this transform will turn
     most integers into small signed values clustered around zero.

  4. The signed integer coefficients are reordered in a manner similar to
     JPEG zig-zag ordering so that statistically they appear in a roughly
     monotonically decreasing order.  Coefficients corresponding to low
     frequencies tend to have larger magnitude and are listed first.  In 3D,
     coefficients corresponding to frequencies *i*, *j*, *k* in the three
     dimensions are ordered by *i* + *j* + *k* first and then by
     *i*\ :sup:`2` + *j*\ :sup:`2` + *k*\ :sup:`2`.

  5. The two's complement signed integers are converted to their negabinary
     (base negative two) representation using one addition and one bit-wise
     exclusive or per integer.  Because negabinary has no single dedicated
     sign bit, these integers are subsequently treated as unsigned.  Unlike
     sign-magnitude representations, the leftmost one-bit in negabinary
     simultaneously encodes the sign and approximate magnitude of a number.
     Moreover, unlike two's complement, numbers small in magnitude have many
     leading zeros in negabinary regardless of sign, which facilitates
     encoding.

  6. The bits that represent the list of |4powd| integers are transposed so
     that instead of being ordered by coefficient they are ordered by bit
     plane, from most to least significant bit.  Viewing each bit plane as
     an unsigned integer, with the lowest bit corresponding to the lowest
     frequency coefficient, the anticipation is that the first several of
     these transposed integers are small, because the coefficients are
     assumed to be ordered by magnitude.

  7. The transform coefficients are compressed losslessly using embedded
     coding by exploiting the property that the coefficients tend to have many
     leading zeros that need not be encoded explicitly.  Each bit plane is
     encoded in two parts, from lowest to highest bit.  First, the *n* lowest
     bits are emitted verbatim, where *n* is the smallest number such that
     the |4powd| |minus| *n* highest bits in all previous bit planes are all
     zero.  Initially, *n* = 0.  Then, a variable-length representation of the
     remaining |4powd| |minus| *n* bits, *x*, is encoded.  For such an integer
     *x*, a single bit is emitted to indicate if *x* = 0, in which case we are
     done with the current bit plane.  If not, then bits of *x* are emitted,
     starting from the lowest bit, until a one-bit is emitted.  This triggers
     another test whether this is the highest set bit of *x*, and the result
     of this test is output as a single bit.  If not, then the procedure
     repeats until all *m* of *x*'s value bits have been output, where
     2\ :sup:`m-1` |leq| *x* < 2\ :sup:`m`.  This can be thought of as a
     run-length encoding of the zeros of *x*, where the run lengths are
     expressed in unary.  The total number of value bits, *n*, in this bit
     plane is then incremented by *m* before being passed to the next bit
     plane, which is encoded by first emitting its *n* lowest bits.  The
     assumption is that these bits correspond to *n* coefficients whose most
     significant bits have already been output, i.e., these *n* bits are
     essentially random and not compressible.  Following this, the remaining
     |4powd| |minus| *n* bits of the bit plane are run-length encoded as
     described above, which potentially results in *n* being increased.

     As an example, *x* = 000001001101000 with *m* = 10 is encoded as
     **0**\ 100\ **1**\ 1\ **1**\ 10\ **1**\ 1000\ **1**, where the bits in
     boldface indicate "group tests" that determine if the remainder of *x*
     (to the left) contains any one-bits.  Again, this variable-length code
     is generated and parsed from right to left.

  8. The embedded coder emits one bit at a time, with each successive bit
     potentially improving the accuracy of the approximation.  The early
     bits are most important and have the greatest impact on accuracy,
     with the last few bits providing very small changes.  The resulting
     compressed bit stream can be truncated at any point and still allow for
     a valid approximate reconstruction of the original block of values.
     The final step truncates the bit stream in one of three ways: to a fixed
     number of bits (the fixed-rate mode); after some fixed number of bit
     planes have been encoded (the fixed-precision mode); or until a lowest
     bit plane number has been encoded, as expressed in relation to the common
     floating-point exponent within the block (the fixed-accuracy mode).

Various parameters are exposed for controlling the quality and compressed
size of a block, and can be specified by the user at a very fine
granularity.  These parameters are discussed :ref:`here <modes>`.

.. _algorithm-lossless:

Lossless Compression
--------------------

The reversible (lossless) compression algorithm shares most steps with
the lossy algorithm.  The main differences are steps 2, 3, and 8, which are
the only sources of error.  Since step 2 may introduce loss in the conversion
to |zfp|'s block-floating-point representation, the reversible algorithm adds
a test to see if this conversion is lossless.  It does so by converting the
values back to the source format and testing the result for bitwise equality
with the uncompressed data.  If this test passes, then a modified
decorrelating transform is performed in step 3 that uses reversible integer
subtraction operations only.  Finally, step 8 is modified so that no one-bits
are truncated in the variable-length bit stream.  However, all least
significant bit planes with all-zero bits are truncated, and the number of
encoded bit planes is recorded in step 7.  As with lossy compression, a
floating-point block consisting of all ("positive") zeros is represented as
a single bit, making it possible to efficiently encode sparse data.

If the block-floating-point transform is not lossless, then the reversible
compression algorithm falls back on a simpler scheme that reinterprets
floating-point values as integers via *type punning*.  This lossless
conversion from floating-point to integer data replaces step 2, and the
algorithm proceeds from there with the modified step 3.  Moreover, this
conversion ensures that special values like infinities, NaNs, and negative
zero are preserved.

The lossless algorithm handles integer data also, for which step 2 is omitted.
