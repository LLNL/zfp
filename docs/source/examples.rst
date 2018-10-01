.. include:: defs.rst

Code Examples
=============

The :file:`examples` directory includes five programs that make use of the
compressor.

.. _ex-simple:

Simple Compressor
-----------------

The :program:`simple` program is a minimal example that shows how to call
the compressor and decompressor on a double-precision 3D array.  Without
the :code:`-d` option, it will compress the array and write the compressed
stream to standard output.  With the :code:`-d` option, it will instead
read the compressed stream from standard input and decompress the
array::

    simple > compressed.zfp
    simple -d < compressed.zfp

For a more elaborate use of the compressor, see the
:ref:`zfp utility <zfpcmd>`.

.. _ex-diffusion:

Diffusion Solver
----------------

The :program:`diffusion` example is a simple forward Euler solver for the
heat equation on a 2D regular grid, and is intended to show how to declare
and work with |zfp|'s compressed arrays, as well as give an idea of how
changing the compression rate and cache size affects the error in the
solution and solution time.  The usage is::

    diffusion [-i] [-n nx ny] [-p] [-t nt] [-r rate] [-c blocks]

where *rate* specifies the exact number of compressed bits to store per
double-precision floating-point value (default = 64); *nx* and *ny*
specify the grid size (default = 100 |times| 100); *nt* specifies the number
of time steps to take (the default is to run until time *t* = 1); and *blocks*
is the number of uncompressed blocks to cache (default = *nx* / 2).  The
:code:`-i` option enables array traversal via iterators instead of indices.

The :code:`-p` option enables OpenMP parallel execution, which makes use
of both mutable and immutable :ref:`private views <private_immutable_view>`
for thread-safe array access.  Note that this example has not been
optimized for parallel performance, but rather serves to show how to
work with |zfp|'s compressed arrays in a multithreaded setting.

Running diffusion with the following arguments::

    diffusion -r 8
    diffusion -r 12
    diffusion -r 20
    diffusion -r 64

should result in this output::

    rate=8 sum=0.996442 error=4.813938e-07
    rate=12 sum=0.998338 error=1.967777e-07
    rate=20 sum=0.998326 error=1.967952e-07
    rate=64 sum=0.998326 error=1.967957e-07

For speed and quality comparison, the solver solves the same problem using
uncompressed double-precision arrays when :code:`-r` is omitted.

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

The :program:`pgm` program illustrates how zfp can be used to compress
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
compression.  Please see :ref:`FAQ #19 <q-inplace>` for more on the
limitations of in-place compression.

.. _ex-iterators:

Iterators
---------

The :program:`iterator` example illustrates how to use |zfp|'s
compressed-array iterators and pointers for traversing arrays.  For
instance, it gives an example of sorting a 1D compressed array
using :cpp:func:`std::sort`.  This example takes no command-line
options.
