.. include:: defs.rst
.. _introduction:

Introduction
============

|zfp| is an open-source library for representing multidimensional numerical
arrays in compressed form to reduce storage and bandwidth requirements.
|zfp| consists of four main components:

* An **efficient number format** for representing small, fixed-size *blocks*
  of real values.  The |zfp| format usually provides higher accuracy per bit
  stored than conventional number formats like IEEE 754 floating point.

* A set of :ref:`classes <arrays>` that implement storage and manipulation
  of a **multidimensional array data type**.  |zfp| arrays support high-speed
  read and write random access to individual array elements and are a
  drop-in replacement for :code:`std::vector` and native C/C++ arrays.
  |zfp| arrays provide accessors like :ref:`proxy pointers <pointers>`,
  :ref:`iterators <iterators>`, and :ref:`views <views>`.  |zfp| arrays
  allow specifying an exact memory footprint or an error tolerance.

* A :ref:`C library <hl-api>` for **streaming compression** of partial or
  whole arrays of integers or floating-point numbers, e.g., for applications
  that read and write large data sets to and from disk.  This library
  supports fast, parallel (de)compression via OpenMP and CUDA.

* A **command-line executable** for :ref:`compressing binary files <zfpcmd>`
  of integer or floating-point arrays, e.g., as a substitute for
  general-purpose compressors like :code:`gzip`.

As a compressor, |zfp| is primarily *lossy*, meaning that the numerical
values are usually only approximately represented, though the user may
specify error tolerances to limit the amount of loss.  Fully
:ref:`lossless compression <q-lossless>`, where values are represented
exactly, is also supported.

|zfp| is primarily written in C and C++ but also includes
:ref:`Python <zfpy>` and :ref:`Fortran <zforp>` bindings.
|zfp| is being developed at
`Lawrence Livermore National Laboratory <https://www.llnl.gov>`__
and is supported by the U.S. Department of Energy's
`Exascale Computing Project <https://www.exascaleproject.org>`__.
|zfp| is a
`2023 R&D 100 Award Winner <https://www.rdworldonline.com/2023-rd-100-award-winners/>`__.


Availability
------------

|zfp| is freely available as open source on
`GitHub <https://github.com/LLNL/zfp>`__ and is distributed under the terms
of a permissive three-clause :ref:`BSD license <license>`.  |zfp| may be
:ref:`installed <installation>` using CMake or GNU Make.  Installation from
source code is recommended for users who wish to configure the internals of
|zfp| and select which components (e.g., programming models, language
bindings) to install.

|zfp| is also available through several package managers, including
Conda (both `C/C++ <https://anaconda.org/conda-forge/zfp>`__ and
`Python <https://anaconda.org/conda-forge/zfpy>`__ packages are available),
`PIP <https://pypi.org/project/zfpy>`__,
`Spack <https://packages.spack.io/package.html?name=zfp>`__, and
`MacPorts <https://ports.macports.org/port/zfp/details/>`__.
`Linux packages <https://repology.org/project/zfp/versions>`__ are available
for several distributions and may be installed, for example, using :code:`apt`
and :code:`yum`.


.. _app-support:

Application Support
-------------------

|zfp| has been incorporated into several independently developed applications,
plugins, and formats, such as

* `Compressed file I/O <https://adios2.readthedocs.io/en/latest/operators/CompressorZFP.html>`__
  in `ADIOS <https://www.olcf.ornl.gov/center-projects/adios/>`__.

* `Compression codec <https://www.blosc.org/posts/support-lossy-zfp/>`__
  in the `BLOSC <https://www.blosc.org>`__ meta compressor.

* `H5Z-ZFP <https://github.com/LLNL/H5Z-ZFP>`__ plugin for
  `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__\ |reg|.  |zfp| is also one of the
  select compressors shipped with
  `HDF5 binaries <https://www.hdfgroup.org/downloads/hdf5/>`__.

* `Compression functions <https://www.intel.com/content/www/us/en/developer/articles/technical/parallel-compression-and-decompression-in-intel-integrated-performance-primitives-zfp-.html>`__
  for Intel\ |reg| `Integrated Performance Primitives <https://software.intel.com/en-us/intel-ipp>`__.

* `Compressed MPI messages <https://doi.org/10.1109/IPDPS49936.2021.00053>`__
  in `MVAPICH2-GDR <https://mvapich.cse.ohio-state.edu/userguide/gdr/>`__.

* `Compressed file I/O <https://www.openinventor.com/en/features/oil-gas-geoscience/zfp-compression/>`__
  in `OpenInventor <https://www.openinventor.com>`__\ |tm|.

* `Compression codec <https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/seismic/open-zgy/-/raw/master/doc/compress.html>`__
  underlying the
  `OpenZGY <https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/seismic/open-zgy>`__
  format.

* `Compressed file I/O <https://topology-tool-kit.github.io/doc/html/TopologicalCompression_8cpp_source.html>`__
  in `TTK <https://topology-tool-kit.github.io>`__.

* `Third-party module <https://gitlab.kitware.com/vtk/vtk/tree/master/ThirdParty/zfp>`__
  in `VTK <https://vtk.org>`__.

* `Compression worklet <http://m.vtk.org/documentation/namespacevtkm_1_1worklet_1_1zfp.html>`__
  in `VTK-m <http://m.vtk.org>`__.

* `Compression codec <https://numcodecs.readthedocs.io/en/stable/zfpy.html>`__ in `Zarr <https://github.com/zarr-developers/zarr-python>`__ via `numcodecs <https://github.com/zarr-developers/numcodecs>`__.

See
`this list <https://computing.llnl.gov/projects/floating-point-compression/related-projects>`__
for other software products that support |zfp|.


Usage
-----

The typical user will interact with |zfp| via one or more of its components,
specifically

* Via the :ref:`C API <hl-api>` when doing I/O in an application or otherwise
  performing data (de)compression online.  High-speed, parallel compression is
  supported via OpenMP and CUDA.

* Via |zfp|'s in-memory :ref:`compressed-array classes <arrays>` when
  performing computations on very large arrays that demand random access to
  array elements, e.g., in visualization, data analysis, or even in numerical
  simulation.  These classes can often substitute C/C++ arrays and STL
  vectors in applications with minimal code changes.

* Via the |zfp| :ref:`command-line tool <zfpcmd>` when compressing
  binary files offline.

* Via :ref:`third-party <app-support>` I/O libraries or tools that support |zfp|.



Technology
----------

|zfp| compresses *d*-dimensional (1D, 2D, 3D, and 4D) arrays of integer or
floating-point values by partitioning the array into cubical blocks of |4powd|
values, i.e., 4, 16, 64, or 256 values for 1D, 2D, 3D, and 4D arrays,
respectively.  Each such block is independently compressed to a fixed-
or variable-length bit string, and these bit strings may be concatenated
into a single stream of bits.

|zfp| usually truncates each per-block bit string to a fixed number of bits
to meet a storage budget or to some variable length needed to meet a given
error tolerance, as dictated by the compressibility of the data.
The bit string representing any given block may be truncated at any point and
still yield a valid approximation.  The early bits are most important; later
bits progressively refine the approximation, similar to how the last few bits
in a floating-point number have less significance than the first several bits.
The trailing bits can usually be discarded (zeroed) with limited impact on
accuracy.

|zfp| was originally designed for floating-point arrays only but has been
extended to also support integer data, and could for instance be used to
compress images and quantized volumetric data.  To achieve high compression
ratios, |zfp| generally uses lossy but optionally error-bounded compression.
Bit-for-bit lossless compression is also possible through one of |zfp|'s
:ref:`compression modes <modes>`.

|zfp| works best for 2D-4D arrays that exhibit spatial correlation, such as
continuous fields from physics simulations, images, regularly sampled terrain
surfaces, etc.  Although |zfp| also provides support for 1D arrays, e.g.,
for audio signals or even unstructured floating-point streams, the
compression scheme has not been well optimized for this use case, and
compression ratio and quality may not be competitive with floating-point
compressors designed specifically for 1D streams.

In all use cases, it is important to know how to use |zfp|'s
:ref:`compression modes <modes>` as well as what the
:ref:`limitations <limitations>` of |zfp| are.  Although it is not critical
to understand the
:ref:`compression algorithm <algorithm>` itself, having some familiarity with
its major components may help understand what to expect and how |zfp|'s
parameters influence the result.


Resources
---------

|zfp| is based on the :ref:`algorithm <algorithm>` described in the following
paper:

.. _tvcg-paper:

  | Peter Lindstrom
  | "`Fixed-Rate Compressed Floating-Point Arrays <https://www.researchgate.net/publication/264417607_Fixed-Rate_Compressed_Floating-Point_Arrays>`__"
  | IEEE Transactions on Visualization and Computer Graphics
  | 20(12):2674-2683, December 2014
  | `doi:10.1109/TVCG.2014.2346458 <http://doi.org/10.1109/TVCG.2014.2346458>`__

|zfp| has evolved since the original publication; the algorithm implemented
in the current version is described in:

.. _siam-paper:

  | James Diffenderfer, Alyson Fox, Jeffrey Hittinger, Geoffrey Sanders, Peter Lindstrom
  | "`Error Analysis of ZFP Compression for Floating-Point Data <https://www.researchgate.net/publication/331162006_Error_Analysis_of_ZFP_Compression_for_Floating-Point_Data>`__"
  | SIAM Journal on Scientific Computing
  | 41(3):A1867-A1898, 2019
  | `doi:10.1137/18M1168832 <http://doi.org/10.1137/18M1168832>`__

For more information on |zfp|, please see the |zfp|
`website <http://zfp.llnl.gov>`__.
For bug reports, please consult the
`GitHub issue tracker <https://github.com/LLNL/zfp/issues>`__.
For questions, comments, and requests, please
`contact us <mailto:zfp@llnl.gov>`__.
