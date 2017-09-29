.. include:: defs.rst

.. _directions:

Future Directions
=================

|zfp| is actively being developed and plans have been made to add a number of
important features, including:

- **Support for 4D arrays**, e.g., for compressing time-varying 3D fields.
  Although the |zfp| compression algorithm trivially generalizes to higher
  dimensions, *d*, the current implementation is hampered by the lack of
  integer types large enough to hold |4powd| bits for *d* > 3.  For now,
  higher-dimensional data should be compressed as collections of independent
  3D fields.

- **Tagging of missing values**.  |zfp| currently assumes that arrays are
  dense, i.e., each array element stores a valid numerical value.  In many
  science applications this is not the case.  For instance, in climate
  modeling, ocean temperature is not defined over land.  In other
  applications, the domain is not rectangular but irregular and embedded in a
  rectangular array.  Such examples of sparse arrays demand a mechanism to tag
  values as missing or indeterminate.  Current solutions often rely on tagging
  missing values as NaNs or special, often very large sentinel values outside
  the normal range, which can lead to poor compression and complete loss of
  accuracy in nearby valid values.  See :ref:`FAQ #7 <q-missing>`.

- **Support for NaNs and infinities**.  Similar to missing values, some
  applications store special IEEE floating-point values that are not yet
  supported by |zfp|.  In fact, the presence of such values will currently
  result in undefined behavior and loss of data for all values within a
  block that contains non-finite values.

- **Lossless compression**.  Although |zfp| can usually limit compression
  errors to within floating-point roundoff error, some applications demand
  bit-for-bit accurate reconstruction.  Strategies for lossless compression
  are currently being evaluated.

- **Progressive decompression**.  Streaming large data sets from remote
  storage for visualization can be time consuming, even when the data is
  compressed.  Progressive streaming allows the data to be reconstructed
  at reduced precision over the entire domain, with quality increasing
  progressively as more data arrives.  The low-level bit stream interface
  already supports progressive access by interleaving bits across blocks
  (see :ref:`FAQ #13 <q-progressive>`), but |zfp| lacks a high-level API
  for generating and accessing progressive streams.

- **Parallel compression**.  |zfp|'s data partitioning into blocks invites
  opportunities for data parallelism on multithreaded platforms by dividing
  the blocks among threads.  An OpenMP implementation of parallel
  compression is under development that produces compressed streams that
  are identical to serially compressed streams.  An experimental
  `CUDA implementation <https://github.com/mclarsen/cuzfp/>`_ for parallel
  compression and decompression on the GPU is also under development.

- **Thread-safe arrays**.  |zfp|'s compressed arrays are not thread-safe,
  even when performing read accesses only.  The primary reason is that
  the arrays employ caching, which requires special protection to avoid
  race conditions.  Work is planned to support both read-only and
  read-write accessible arrays that are thread-safe, most likely by
  using thread-local caches for read-only access and disjoint sub-arrays
  for read-write access, where each thread has exclusive ownership of a
  portion of the array.

- **Variable-rate arrays**.  |zfp| currently supports only fixed-rate
  compressed arrays, which wastes bits in smooth regions with little
  information content while too few bits may be allocated to accurately
  preserve sharp features such as shocks and material interfaces, which
  tend to drive the physics in numerical simulations.  Two candidate
  solutions have been identified for read-only and read-write access
  to variable-rate arrays with very modest storage overhead.  These
  arrays will support both fixed precision and accuracy.

- **Array operations**.  |zfp|'s compressed arrays currently support basic
  indexing and initialization, but lack essential features such as shallow
  and deep copies, slicing, views, etc.  Work is underway to address these
  deficiencies.

- **Language bindings**.  The main compression codec is written in C89 to
  facilitate calls from other languages, but would benefit from language
  wrappers to ease integration.  |zfp|'s compressed arrays exploit the
  operator overloading provided by C++, and therefore can currently not
  be used in other languages, including C.  Work is planned to add complete
  language bindings for C, C++, Fortran, and Python.

Please contact `Peter Lindstrom <mailto:pl@llnl.gov>`__ with requests for
features not listed above.
