.. include:: defs.rst

Code Examples
=============

The :file:`examples` directory includes ten programs that make use of the
compressor.

.. _ex-simple:

Simple Compressor
-----------------

The :program:`simple` program is a minimal example that shows how to call
the compressor and decompressor on a double-precision 3D array.  Without
the :code:`-d` option, it will compress the array and write the compressed
stream to standard output.  With the :code:`-d` option, it will instead
read the compressed stream from standard input and decompress the array::

    simple > compressed.zfp
    simple -d < compressed.zfp

For a more elaborate use of the compressor, see the
:ref:`zfp utility <zfpcmd>`.

.. _ex-array:

Compressed-Array C++ Classes
----------------------------

The :program:`array` program shows how to declare, write to, and read from
|zfp|'s compressed-array C++ objects (in this case, 2D double-precision
arrays), which is essentially as straightforward as working with STL vectors.
This example initializes a 2D array with a linear ramp of 12 |times| 8 = 96
values using only four bits of storage per value, which using uncompressed
storage would not be enough to distinguish more than 16 different values.
For more advanced compressed-array features, see the
:ref:`tutorial <tut-arrays>`.

.. _ex-diffusion:

Diffusion Solver
----------------

The :program:`diffusion` example is a simple forward Euler solver for the
heat equation on a 2D regular grid, and is intended to show how to declare
and work with |zfp|'s compressed arrays, as well as give an idea of how
changing the compression parameters and cache size affects the error in the
solution and solution time.  The usage is::

    diffusion [options]
      -a <tolerance> : absolute error tolerance (requires -c)
      -b <blocks> : cache size in number of 4x4 blocks
      -c : use read-only arrays (needed for -a, -p, -R)
      -d : use double-precision tiled arrays
      -f : use single-precision tiled arrays
      -h : use half-precision tiled arrays
      -i : traverse arrays using iterators instead of integer indices
      -j : use OpenMP parallel execution (requires -r)
      -n <nx> <ny> : grid dimensions
      -p <precision> : precision in uncompressed bits/value (requires -c)
      -r <rate> : rate in compressed bits/value
      -R : reversible mode (requires -c)
      -t <nt> : number of time steps

Here *rate* specifies the exact number of compressed bits to store per
double-precision floating-point value; *nx* and *ny* specify the grid size
(default = 128 |times| 128); *nt* specifies the number of time steps to take
(the default is to run until time *t* = 1); and *blocks* is the number of
uncompressed blocks to cache (default = *nx* / 2).  The :code:`-i` option
enables array traversal via iterators instead of indices.

The :code:`-j` option enables OpenMP parallel execution, which makes use
of both mutable and immutable :ref:`private views <private_immutable_view>`
for thread-safe array access.  Note that this example has not been
optimized for parallel performance, but rather serves to show how to
work with |zfp|'s compressed arrays in a multithreaded setting.

This example also illustrates how :ref:`read-only arrays <carray_classes>`
(:code:`-c`) may be used in conjunction with fixed-rate (:code:`-r`),
fixed-precision (:code:`-p`), fixed-accuracy (:code:`-a`),
or reversible (:code:`-R`) mode.

The output lists for each time step the current rate of the state array and
in parentheses any additional storage, e.g., for the block
:ref:`cache <caching>` and :ref:`index <index>` data structures, both in bits
per array element.  Running diffusion with the following arguments::

    diffusion -r 8
    diffusion -r 12
    diffusion -r 16
    diffusion -r 24
    diffusion

should result in this final output::

    sum=0.995170 error=4.044954e-07
    sum=0.998151 error=1.237837e-07
    sum=0.998345 error=1.212734e-07
    sum=0.998346 error=1.212716e-07
    sum=0.998346 error=1.212716e-07

For speed and quality comparison, the solver solves the same problem using
uncompressed double-precision row-major arrays when compression parameters
are omitted.  If one of :code:`-h`, :code:`-f`, :code:`-d` is specified,
uncompressed tiled arrays are used.  These arrays are based on the |zfp|
array classes but make use of the :ref:`generic codec <codec>`, which
stores blocks as uncompressed scalars of the specified type (:code:`half`,
:code:`float`, or :code:`double`) while utilizing a double-precision block
cache (like |zfp|'s compressed arrays).

The :program:`diffusionC` program is the same example written entirely
in C using the |cfp| :ref:`wrappers <cfp>` around the C++ compressed array
classes.

.. _ex-speed:

Speed Benchmark
---------------

The :program:`speed` program takes two optional parameters::

    speed [rate] [blocks]

It measures the throughput of compression and decompression of 3D
double-precision data (in megabytes of uncompressed data per second).
By default, a rate of 1 bit/value and two million blocks are
processed.

.. _ex-pgm:

PGM Image Compression
---------------------

The :program:`pgm` program illustrates how |zfp| can be used to compress
grayscale images in the
`pgm format <http://netpbm.sourceforge.net/doc/pgm.html>`_.  The usage is::

    pgm <param> <input.pgm >output.pgm

If :code:`param` is positive, it is interpreted as the rate in bits per pixel,
which ensures that each block of |4by4| pixels is compressed to a fixed
number of bits, as in texture compression codecs.
If :code:`param` is negative, then fixed-precision mode is used with precision
:code:`-param`, which tends to give higher quality for the same rate.  This
use of |zfp| is not intended to compete with existing texture and image
compression formats, but exists merely to demonstrate how to compress 8-bit
integer data with |zfp|.  See FAQs :ref:`#20 <q-relerr>` and
:ref:`#21 <q-lossless>` for information on the effects of setting the
precision.

.. _ex-ppm:

PPM Image Compression
---------------------

The :program:`ppm` program is analogous to the :program:`pgm` example, but
has been designed for compressing color images in the
`ppm format <http://netpbm.sourceforge.net/doc/ppm.html>`_.  Rather than
compressing RGB channels independently, ppm exploits common strategies for
color image compression such as color channel decorrelation and chroma
subsampling.

The usage is essentially the same as for :ref:`pgm <ex-pgm>`::

    ppm <param> <input.ppm >output.ppm

where a positive :code:`param` specifies the rate in bits per pixel; when
negative, it specifies the precision (number of bit planes to encode) in
fixed-precision mode.

.. _ex-inplace:

In-place Compression
--------------------

The :program:`inplace` example shows how one might use zfp to perform in-place
compression and decompression when memory is at a premium.  Here the
floating-point array is overwritten with compressed data, which is later
decompressed back in place.  This example also shows how to make use of
some of the low-level features of zfp, such as its low-level, block-based
compression API and bit stream functions that perform seeks on the bit
stream.  The program takes one optional argument::

    inplace [tolerance]

which specifies the fixed-accuracy absolute tolerance to use during
compression.  Please see FAQ :ref:`#19 <q-inplace>` for more on the
limitations of in-place compression.

.. _ex-iterators:

Iterators
---------

The :program:`iterator` example illustrates how to use |zfp|'s
compressed-array iterators and pointers for traversing arrays.  For
instance, it gives an example of sorting a 1D compressed array
using :cpp:func:`std::sort`.  This example takes no command-line
options.

The :program:`iteratorC` example illustrates the equivalent |cfp|
iterator operations.  It closely follows the usage shown in the 
:program:`iterator` example with some minor differences. It 
likewise takes no command-line options.
