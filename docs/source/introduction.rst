.. include:: defs.rst
.. _introduction:

Introduction
============

|zfp| is an open source library for compressed numerical arrays that
support high throughput read and write random access.  |zfp| also supports
streaming compression of integer and floating-point data, e.g., for
applications that read and write large data sets to and from disk.
|zfp| is primarily written in C and C++ but also includes Python and
Fortran bindings.

|zfp| was developed at
`Lawrence Livermore National Laboratory <https://www.llnl.gov>`_ and is
loosely based on the :ref:`algorithm <algorithm>` described in the following
paper:

.. _paper:

  | Peter Lindstrom
  | "`Fixed-Rate Compressed Floating-Point Arrays <https://www.researchgate.net/publication/264417607_Fixed-Rate_Compressed_Floating-Point_Arrays>`_"
  | IEEE Transactions on Visualization and Computer Graphics
  | 20(12):2674-2683, December 2014
  | `doi:10.1109/TVCG.2014.2346458 <http://doi.org/10.1109/TVCG.2014.2346458>`_

|zfp| was originally designed for floating-point arrays only, but has been
extended to also support integer data, and could for instance be used to
compress images and quantized volumetric data.  To achieve high compression
ratios, |zfp| generally uses lossy but optionally error-bounded compression.
Bit-for-bit lossless compression is also possible through one of |zfp|'s
:ref:`compression modes <modes>`.

|zfp| works best for 2D and 3D arrays that exhibit spatial correlation, such as
continuous fields from physics simulations, images, regularly sampled terrain
surfaces, etc.  Although |zfp| also provides a 1D array class that can be used
for 1D signals such as audio, or even unstructured floating-point streams,
the compression scheme has not been well optimized for this use case, and
rate and quality may not be competitive with floating-point compressors
designed specifically for 1D streams.  As of version |4drelease|, |zfp| also
supports compression of 4D arrays.

|zfp| is freely available as open source under a :ref:`BSD license <license>`.
For more information on |zfp| and comparisons with other compressors, please
see the |zfp|
`website <https://computation.llnl.gov/projects/floating-point-compression>`_.
For bug reports, please consult the
`GitHub issue tracker <https://github.com/LLNL/zfp/issues>`_.
For questions, comments, and requests, please contact
`Peter Lindstrom <mailto:pl@llnl.gov>`__.
