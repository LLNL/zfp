.. include:: defs.rst
.. _introduction:

Introduction
============

|zfp| is an open source C/C++ library for compressed numerical arrays that
support high throughput read and write random access.  |zfp| also supports
streaming compression of integer and floating-point data, e.g., for
applications that read and write large data sets to and from disk.

|zfp| was written by `Peter Lindstrom <https://people.llnl.gov/pl>`_
at `Lawrence Livermore National Laboratory <https://www.llnl.gov>`_, and
is loosely based on the :ref:`algorithm <algorithm>` described in the
following paper:

.. _paper:

  | Peter Lindstrom
  | "`Fixed-Rate Compressed Floating-Point Arrays <https://www.researchgate.net/publication/264417607_Fixed-Rate_Compressed_Floating-Point_Arrays>`_"
  | IEEE Transactions on Visualization and Computer Graphics
  | 20(12):2674-2683, December 2014
  | `doi:10.1109/TVCG.2014.2346458 <http://doi.org/10.1109/TVCG.2014.2346458>`_

|zfp| was originally designed for floating-point data only, but has been
extended to also support integer data, and could for instance be used to
compress images and quantized volumetric data.  To achieve high compression
ratios, |zfp| uses lossy but optionally error-bounded compression.  Although
bit-for-bit lossless compression of floating-point data is not always
possible, |zfp| is usually accurate to within machine epsilon in near-lossless
mode.

|zfp| works best for 2D and 3D arrays that exhibit spatial coherence, such as
continuous fields from physics simulations, images, regularly sampled terrain
surfaces, etc.  Although |zfp| also provides a 1D array class that can be used
for 1D signals such as audio, or even unstructured floating-point streams,
the compression scheme has not been well optimized for this use case, and
rate and quality may not be competitive with floating-point compressors
designed specifically for 1D streams.

|zfp| is freely available as open source under a :ref:`BSD license <license>`.
For more information on |zfp| and comparisons with other compressors, please
see the |zfp|
`website <https://computation.llnl.gov/projects/floating-point-compression>`_.
For questions, comments, requests, and bug reports, please contact
`Peter Lindstrom <mailto:pl@llnl.gov>`__.
