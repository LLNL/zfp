.. include:: defs.rst
.. _overview:

Overview
========

|zfp| is a compressor for integer and floating-point data stored in
multidimensional arrays.  The compressor is primarily *lossy*, meaning
that the numerical values are usually only approximately represented,
though the user may specify error tolerances to limit the amount of loss.
Fully :ref:`lossless compression <q-lossless>`, where values are
represented exactly, is also possible.

The |zfp| software primarily consists of three components: a C library
for compressing whole arrays (or smaller pieces of arrays); C++ classes
that implement random-accessible compressed arrays; and a command-line
compression tool and other code examples.  Additional language bindings
are also available for the first two components.

|zfp| has also been incorporated into several independently developed
plugins for interfacing |zfp| with popular I/O libraries and visualization
tools such as
`ADIOS <https://www.olcf.ornl.gov/center-projects/adios/>`_,
`HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_,
`Intel IPP <https://software.intel.com/en-us/intel-ipp>`_,
`TTK <https://topology-tool-kit.github.io>`_, and
`VTK-m <http://m.vtk.org>`_.

The typical user will interact with |zfp| via one or more of those
components, specifically

* Via the :ref:`C API <hl-api>` when doing I/O in an application or
  otherwise performing data (de)compression online.  High-speed,
  parallel compression is supported via OpenMP and CUDA.

* Via |zfp|'s in-memory :ref:`compressed-array classes <arrays>` when
  performing computations on very large arrays that demand random access to
  array elements, e.g., in visualization, data analysis, or even in numerical
  simulation.  These classes can often substitute C/C++ arrays and STL
  vectors in applications with minimal code changes.

* Via the |zfp| :ref:`command-line tool <zfpcmd>` when compressing
  binary files offline.

* Via one of the I/O libraries or visualization tools that support |zfp|,
  e.g.,

  * `ADIOS plugin <https://github.com/suchyta1/AtoZ>`_
  * `HDF5 plugin <https://github.com/LLNL/H5Z-ZFP>`_
  * `Intel IPP plugin <https://software.intel.com/en-us/ipp-dev-reference-zfp-compression-functions>`_
  * `TTK plugin <https://topology-tool-kit.github.io/installation.html>`_
  * `VTK plugin <https://gitlab.kitware.com/vtk/vtk/tree/master/ThirdParty/zfp>`_
  * `VTK-m plugin <http://m.vtk.org/documentation/namespacevtkm_1_1worklet_1_1zfp.html>`_

In all cases, it is important to know how to use |zfp|'s
:ref:`compression modes <modes>` as well as what the
:ref:`limitations <limitations>` of |zfp| are.  Although it is not critical
to understand the
:ref:`compression algorithm <algorithm>` itself, having some familiarity with
its major components may help understand what to expect and how |zfp|'s
parameters influence the result.

|zfp| compresses *d*-dimensional (1D, 2D, 3D, and 4D) arrays of integer or
floating-point values by partitioning the array into cubical blocks of |4powd|
values, i.e., 4, 16, 64, or 256 values for 1D, 2D, 3D, and 4D arrays,
respectively.  Each such block is (de)compressed independently into a fixed-
or variable-length bit string, and these bit strings are concatenated into a
single stream of bits.

|zfp| usually truncates each bit string to a fixed number of bits to meet
a storage budget or to some variable length needed to meet a given error
tolerance, as dictated by the compressibility of the data.
The bit string representing any given block may be truncated at any point and
still yield a valid approximation.  The early bits are most important; later
bits progressively refine the approximation, similar to how the last few bits
in a floating-point number have less significance than the first several bits
and can often be discarded (zeroed) with limited impact on accuracy.

The next several sections cover information on the |zfp| algorithm and its
parameters; the C API; the compressed array classes; the Python and Fortran
APIs; examples of how to perform compression and work with the classes; how
to use the binary file compressor; and code examples that further illustrate
how to use |zfp|.  The documentation concludes with frequently asked questions
and troubleshooting, as well as current limitations and future development
directions.

For questions not answered here, please contact
`Peter Lindstrom <mailto:pl@llnl.gov>`_.
