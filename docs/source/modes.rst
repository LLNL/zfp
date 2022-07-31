.. include:: defs.rst

.. index::
   single: Compression mode
.. _modes:

Compression Modes
=================

|zfp| accepts one or more parameters for specifying how the data is to be
compressed to meet various constraints on accuracy or size.  At a high
level, there are five different compression modes that are mutually
exclusive:
:ref:`expert <mode-expert>`,
:ref:`fixed-rate <mode-fixed-rate>`,
:ref:`fixed-precision <mode-fixed-precision>`,
:ref:`fixed-accuracy <mode-fixed-accuracy>`, and
:ref:`reversible <mode-reversible>` mode.
The user has to select one of these modes and its corresponding parameters.
In streaming I/O applications, the
:ref:`fixed-accuracy mode <mode-fixed-accuracy>` is preferred, as
it provides the highest quality (in the absolute error sense) per bit of
compressed storage.

The :c:type:`zfp_stream` struct encapsulates the compression parameters and
other information about the compressed stream.  Its members should not be
manipulated directly.  Instead, use the access functions (see the
:ref:`C API <hl-api>` section) for setting and querying them.  One can
verify the active compression mode on a :c:type:`zfp_stream` through
:c:func:`zfp_stream_compression_mode`.  The members that govern the
compression parameters are described below.

.. _mode-expert:
.. index::
   single: Compression mode; Expert mode

Expert Mode
-----------

The most general mode is the 'expert mode,' which takes four integer
parameters.  Although most users will not directly select this mode,
we discuss it first since the other modes can be expressed in terms of
setting expert mode parameters.

The four parameters denote constraints that are applied to each block
in the :ref:`compression algorithm <algorithm-lossy>`.
Compression is terminated as soon as one of these constraints is not met,
which has the effect of truncating the compressed bit stream that encodes
the block.  The four constraints are as follows:

.. c:member:: uint zfp_stream.minbits

  The minimum number of compressed bits used to represent a block.  Usually
  this parameter equals one bit, unless each and every block is to be stored
  using a fixed number of bits to facilitate random access, in which case it
  should be set to the same value as :c:member:`zfp_stream.maxbits`.

.. c:member:: uint zfp_stream.maxbits

  The maximum number of bits used to represent a block.  This parameter
  sets a hard upper bound on compressed block size and governs the rate
  in :ref:`fixed-rate mode <mode-fixed-rate>`.  It may also be used as an
  upper storage limit to guard against buffer overruns in combination with
  the accuracy constraints given by :c:member:`zfp_stream.maxprec` and
  :c:member:`zfp_stream.minexp`.

.. c:member:: uint zfp_stream.maxprec

  The maximum number of bit planes encoded.  This parameter governs the number
  of most significant uncompressed bits encoded per transform coefficient.
  It does not directly correspond to the number of uncompressed mantissa bits
  for the floating-point or integer values being compressed, but is closely
  :ref:`related <q-relerr>`.  This is the parameter that specifies the
  precision in :ref:`fixed-precision mode <mode-fixed-precision>`, and it
  provides a mechanism for controlling the *relative error*.  Note that this
  parameter selects how many bits planes to encode regardless of the magnitude
  of the common floating-point exponent within the block.

.. c:member:: int zfp_stream.minexp

  The smallest absolute bit plane number encoded (applies to floating-point
  data only; this parameter is ignored for integer data).  The place value of
  each transform coefficient bit depends on the common floating-point exponent,
  *e*, that scales the integer coefficients.  If the most significant
  coefficient bit has place value 2\ :sup:`e`, then the number of bit planes
  encoded is (one plus) the difference between *e* and
  :c:member:`zfp_stream.minexp`.  As an analogy, consider representing
  currency in decimal.  Setting :c:member:`zfp_stream.minexp` to -2 would,
  if generalized to base 10, ensure that amounts are represented to cent
  accuracy, i.e., in units of 10\ :sup:`-2` = $0.01.  This parameter governs
  the *absolute error* in :ref:`fixed-accuracy mode <mode-fixed-accuracy>`.
  Note that to achieve a certain accuracy in the decompressed values, the
  :c:member:`zfp_stream.minexp` value has to be conservatively lowered since
  |zfp|'s inverse transform may magnify the error (see also
  FAQs :ref:`#20-22 <q-relerr>`).

Care must be taken to allow all constraints to be met, as encoding
terminates as soon as a single constraint is violated (except
:c:member:`zfp_stream.minbits`, which is satisfied at the end of encoding by
padding zeros).

.. warning::

  For floating-point data, the :c:member:`zfp_stream.maxbits` parameter must
  be large enough to allow the common block exponent and any control bits to
  be encoded.  This implies *maxbits* |geq| 9 for single-precision data and
  *maxbits* |geq| 12 for double-precision data.  Choosing a smaller value is
  of no use as it would prevent any fraction (value) bits from being encoded,
  resulting in an all-zero decompressed block.  More importantly, such a
  constraint will not be respected by |zfp| for performance reasons, which
  if not accounted for could potentially lead to buffer overruns.

As mentioned above, other combinations of constraints can be used.
For example, to ensure that the compressed stream is not larger than
the uncompressed one, or that it fits within the amount of memory
allocated, one may in conjunction with other constraints set
::

  maxbits = 4^d * CHAR_BIT * sizeof(Type)

where Type is either float or double.  The ``minbits`` parameter is useful
only in fixed-rate mode; when ``minbits`` = ``maxbits``, zero-bits are
padded to blocks that compress to fewer than ``maxbits`` bits.

The effects of the above four parameters are best explained in terms of the
three main compression modes supported by |zfp|, described below.

.. _mode-fixed-rate:
.. index::
   single: Compression mode; Fixed-rate mode
   single: Rate

Fixed-Rate Mode
---------------

In fixed-rate mode, each *d*-dimensional compressed block of |4powd| values
is stored using a fixed number of bits given by the parameter
:c:member:`zfp_stream.maxbits`.  This number of compressed bits per
*block* is amortized over the |4powd| values to give a *rate* in
bits per *value*::

  rate = maxbits / 4^d

This rate is specified in the :ref:`zfp executable <zfpcmd>` via the
:option:`-r` option, and programmatically via :c:func:`zfp_stream_set_rate`,
as a floating-point value.  Fixed-rate mode can also be achieved via the
expert mode interface by setting
::

  minbits = maxbits = (1 << (2 * d)) * rate
  maxprec = ZFP_MAX_PREC
  minexp = ZFP_MIN_EXP

Note that each block stores a bit to indicate whether the block is empty,
plus a common exponent.  Hence :c:member:`zfp_stream.maxbits` must be at
least 9 for single precision and 12 for double precision.

Fixed-rate mode is needed to support random access to blocks, and also is
the mode used in the implementation of |zfp|'s
:ref:`compressed arrays <arrays>`.  Fixed-rate mode also ensures a
predictable memory/storage footprint, but usually results in far worse
accuracy per bit than the variable-rate fixed-precision and fixed-accuracy
modes.

.. note::
  Use fixed-rate mode only if you have to bound the compressed size
  or need read and write random access to blocks.

.. _mode-fixed-precision:
.. index::
   single: Compression mode; Fixed-precision mode

Fixed-Precision Mode
--------------------

In fixed-precision mode, the number of bits used to encode a block may
vary, but the number of bit planes (i.e., the precision) encoded for the
transform coefficients is fixed.  To achieve the desired precision,
use option :option:`-p` with the :ref:`zfp executable <zfpcmd>` or call
:c:func:`zfp_stream_set_precision`.  In expert mode, fixed precision is
achieved by specifying the precision in :c:member:`zfp_stream.maxprec`
and fully relaxing the size constraints, i.e.,
::

  minbits = ZFP_MIN_BITS
  maxbits = ZFP_MAX_BITS
  maxprec = precision
  minexp = ZFP_MIN_EXP

Fixed-precision mode is preferable when relative rather than absolute
errors matter.

.. _mode-fixed-accuracy:
.. index::
   single: Compression mode; Fixed-accuracy mode

Fixed-Accuracy Mode
-------------------

In fixed-accuracy mode, all transform coefficient bit planes up to a
minimum bit plane number are encoded.  (The actual minimum bit plane
is not necessarily :c:member:`zfp_stream.minexp`, but depends on the
dimensionality, *d*, of the data.  The reason for this is that the inverse
transform incurs range expansion, and the amount of expansion depends on
the number of dimensions.)  Thus, :c:member:`zfp_stream.minexp` should
be interpreted as the base-2 logarithm of an absolute error tolerance.
In other words, given an uncompressed value, *f*, and a reconstructed
value, *g*, the absolute difference \| *f* |minus| *g* \| is at most
2\ :sup:`minexp`.
(Note that it is not possible to guarantee error tolerances smaller than
machine epsilon relative to the largest value within a block.)  This error
tolerance is not always tight (especially for 3D and 4D arrays), but can
conservatively be set so that even for worst-case inputs the error
tolerance is respected.  To achieve fixed accuracy to within 'tolerance',
use option :option:`-a` with the :ref:`zfp executable <zfpcmd>` or call
:c:func:`zfp_stream_set_accuracy`.  The corresponding expert mode
parameters are::

  minbits = ZFP_MIN_BITS
  maxbits = ZFP_MAX_BITS
  maxprec = ZFP_MAX_PREC
  minexp = floor(log2(tolerance))

As in fixed-precision mode, the number of bits used per block is not
fixed but is dictated by the data.  Use *tolerance* = 0 to achieve
near-lossless compression (see :ref:`mode-reversible` for guaranteed
lossless compression).  Fixed-accuracy mode gives the highest quality
(in terms of absolute error) for a given compression rate, and is
preferable when random access is not needed.

.. note::
  Fixed-accuracy mode is available for floating-point (not integer) data
  only.

.. index::
   single: Compression mode; Reversible mode
   single: Lossless compression
.. _mode-reversible:

Reversible Mode
---------------

As of |zfp| |revrelease|, reversible (lossless) compression is supported.
As with the other compression modes, each block is compressed and decompressed
independently, but reversible mode uses a different compression algorithm
that ensures a bit-for-bit identical reconstruction of integer and
floating-point data.  For IEEE-754 floating-point data, reversible mode
preserves special values such as subnormals, infinities, NaNs, and
positive and negative zero.

The expert mode parameters corresponding to reversible mode are::

  minbits = ZFP_MIN_BITS
  maxbits = ZFP_MAX_BITS
  maxprec = ZFP_MAX_PREC
  minexp < ZFP_MIN_EXP

Reversible mode is enabled via :c:func:`zfp_stream_set_reversible` and through
the :option:`-R` command-line option in the :ref:`zfp executable <zfpcmd>`.
It is supported by both the low- and high-level interfaces and by the serial
and OpenMP execution policies, but it is not yet implemented in CUDA.
