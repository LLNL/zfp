.. include:: defs.rst

.. _directions:

Future Directions
=================

|zfp| is actively being developed and plans have been made to add a number of
important features, including:

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
  applications store special IEEE floating-point values that are supported
  by |zfp| only in :ref:`reversible mode <mode-reversible>`.
  In fact, for all lossy compression modes, the presence of such values will
  currently result in undefined behavior and loss of data for all values
  within a block that contains non-finite values.

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
  compression is available that produces compressed streams that
  are identical to serially compressed streams.  However, parallel
  decompression is not yet supported.  |zfp| also supports compression and
  decompression on the GPU via CUDA.  However, only fixed-rate mode is
  so far supported.

- **Variable-rate arrays**.  |zfp| currently supports only fixed-rate
  compressed arrays, which wastes bits in smooth regions with little
  information content while too few bits may be allocated to accurately
  preserve sharp features such as shocks and material interfaces, which
  tend to drive the physics in numerical simulations.  Two candidate
  solutions have been identified for read-only and read-write access
  to variable-rate arrays with very modest storage overhead.  These
  arrays will support both fixed precision and accuracy.

- **Array operations**.  |zfp|'s compressed arrays currently support basic
  indexing and initialization, but lack array-wise operations such as
  arithmetic, reductions, etc.  Some such operations can exploit the
  higher precision (than IEEE) supported by |zfp|, as well as accelerated
  blockwise computations that need not fully decompress and convert the
  |zfp| representation to IEEE.

- **Language bindings**.  The main compression codec is written in C89 to
  facilitate calls from other languages, but would benefit from language
  wrappers to ease integration.  As of |zfp| |cfprelease|, C wrappers are
  available for a subset of the C++ compressed array API.  Work is planned
  to add complete language bindings for C, C++, Fortran, and Python.

Please contact `Peter Lindstrom <mailto:pl@llnl.gov>`__ with requests for
features not listed above.
