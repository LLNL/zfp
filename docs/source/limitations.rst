.. include:: defs.rst

.. _limitations:

Limitations
===========

|zfp| has evolved over the years from a research prototype to a production
quality library.  However, the API and even the compression codec are
still undergoing changes as new important features are added.

Below is a list of known limitations of the current version of |zfp|.
See the section on :ref:`directions` for a discussion of planned features
that will address some of these limitations.

- Special floating-point values like infinity and NaN are supported in
  reversible mode but not in |zfp|'s lossy compression modes.  Subnormal
  floating-point numbers are, however, correctly handled.  There is an
  implicit assumption that floating point conforms to IEEE-754, though
  extensions to other floating-point formats should be possible with
  minor effort.

- The optional |zfp| :ref:`header <zfp-header>` supports arrays with at
  most 2\ :sup:`48` elements.  The |zfp| header limits each dimension
  to 2\ :sup:`48/d` elements in a *d*-dimensional array, i.e.,
  2\ :sup:`48`, 2\ :sup:`24`, 2\ :sup:`16`, and 2\ :sup:`12` for 1D through
  4D arrays, respectively.  Note that this limitation applies only to
  the header; array dimensions are otherwise limited only by the size
  supported by :code:`size_t`.

- The :ref:`compressed-array classes <arrays>` have additional size
  restrictions.  The :ref:`cache <caching>` supports at most
  2\ :sup:`p-1` - 1 blocks, where *p* is the number of bits in a :code:`uint`
  (usually *p* = 32).  Consequently, the number of elements in a
  *d*-dimensional compressed array is at most
  |4powd| |times| (2\ :sup:`p-1` - 1), or about 8 billion elements for 1D
  arrays.

- Conventional pointers and references to individual array elements are
  not available.  That is, constructions like :code:`double* ptr = &a[i];`
  are not possible when :code:`a` is a |zfp| array.  However, as of
  |zfp| |proxyrelease|, :ref:`proxy pointers <pointers>` are available that
  act much like pointers to uncompressed data.  Similarly, operators :code:`[]`
  and :code:`()` do not return regular C++ references.  Instead, a
  :ref:`proxy reference <references>` class is used (similar to how
  `STL bit vectors <https://en.cppreference.com/w/cpp/container/vector_bool>`__
  are implemented).  These proxy references and pointers can, however, safely
  be passed to functions and used where regular references and pointers can.

- The :ref:`read-only array classes <carray_classes>` do not yet support
  (de)serialization.

- |zfp| can potentially provide higher precision than conventional float
  and double arrays, but the interface currently does not expose this.
  For example, such added precision could be useful in finite difference
  computations, where catastrophic cancellation can be an issue when
  insufficient precision is available.

- Only single and double precision floating types are supported.
  Generalizations to IEEE half and quad precision would be useful.  For
  instance, compressed 64-bit-per-value storage of 128-bit quad-precision
  numbers could greatly improve the accuracy of double-precision
  floating-point computations using the same amount of storage.
  The |zfp| compressed-array classes do not yet support integer scalar
  types.

- Complex-valued arrays are not directly supported.  Real and imaginary
  components must be stored as separate arrays, which may result in lost
  opportunities for compression, e.g., if the complex magnitude is constant
  and only the phase varies.

- Version |omprelease| adds support for OpenMP compression.  However,
  OpenMP decompression is not yet supported.

- Version |cudarelease| adds support for CUDA compression and decompression.
  However, only the fixed-rate compression mode is so far supported.
  The CUDA implementation is further subject to
  :ref:`additional limitations <cuda-limitations>`.

- The |cfp| :ref:`C wrappers <cfp>` for |zfp|'s compressed arrays support
  only a subset of the C++ API.  |zfp| |4darrrelease| adds support for proxy
  references, pointers, and iterators, but views and read-only arrays are
  not yet supported.  Furthermore, |cfp| works only with the |zfp| codec.

- The Python and Fortran bindings do not yet support |zfp|'s compressed-array
  classes.  Moreover, only a select subset of the
  :ref:`high-level API <hl-api>` is available via Python.
