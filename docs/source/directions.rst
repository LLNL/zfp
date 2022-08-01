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
  accuracy in nearby valid values.  See FAQ :ref:`#7 <q-missing>`.

- **Support for NaNs and infinities**.  Similar to missing values, some
  applications store special IEEE floating-point values that are supported
  by |zfp| only in :ref:`reversible mode <mode-reversible>`.
  In fact, for all lossy compression modes, the presence of such values will
  currently result in undefined behavior and loss of data for all values
  within a block that contains non-finite values.

- **Support for more general data types**.  |zfp| currently does not
  directly support half and quad precision floating point.  Nor is there
  support for 8- and 16-bit integers.  With the emergence of new number
  representations like *posits* and *bfloat16*, we envision the need for
  a more general interface and a single unified |zfp| representation that
  would allow for *conversion* between |zfp| and *any* number representation.
  We are working on developing an uncompressed interchange format that acts
  like an intermediary between |zfp| and other number formats.  This format
  decouples the |zfp| compression pipeline from the external number type and
  allows new number formats to be supported via user-defined conversion
  functions to and from the common interchange format.

- **Progressive decompression**.  Streaming large data sets from remote
  storage for visualization can be time consuming, even when the data is
  compressed.  Progressive streaming allows the data to be reconstructed
  at reduced precision over the entire domain, with quality increasing
  progressively as more data arrives.  The low-level bit stream interface
  already supports progressive access by interleaving bits across blocks
  (see FAQ :ref:`#13 <q-progressive>`), but |zfp| lacks a high-level API
  for generating and accessing progressive streams.

- **Parallel compression**.  |zfp|'s data partitioning into blocks invites
  opportunities for data parallelism on multithreaded platforms by dividing
  the blocks among threads.  An OpenMP implementation of parallel
  compression is available that produces compressed streams that
  are identical to serially compressed streams.  However, parallel
  decompression is not yet supported.  |zfp| also supports compression and
  decompression on the GPU via CUDA.  However, only fixed-rate mode is
  so far supported.

- **Variable-rate arrays**.  |zfp| currently offers only fixed-rate
  compressed arrays with random-access write support; |zfp| |carrrelease|
  further provides read-only variable-rate arrays.  Fixed-rate arrays waste
  bits in smooth regions with little information content while too few bits
  may be allocated to accurately preserve sharp features such as shocks and
  material interfaces, which tend to drive the physics in numerical
  simulations.  A candidate solution has been developed for variable-rate
  arrays that support read-write random access with modest storage overhead.
  We expect to release this capability in the near future.

- **Array operations**.  |zfp|'s compressed arrays currently support basic
  indexing and initialization, but lack array-wise operations such as
  arithmetic, reductions, etc.  Some such operations can exploit the
  higher precision (than IEEE-754) supported by |zfp|, as well as accelerated
  blockwise computations that need not fully decompress and convert the
  |zfp| representation to IEEE-754.

- **Language bindings**.  The main compression codec is written in C89 to
  facilitate calls from other languages.  |zfp|'s compressed arrays, on
  the other hand, are written in C++.  |zfp| |cfprelease| and |zforprelease|
  add C wrappers around compressed arrays and Fortran and Python bindings to
  the high-level C API.  Work is planned to provide additional language
  bindings for C, C++, Fortran, and Python to expose the majority of |zfp|'s
  capabilities through all of these programming languages.

Please `contact us <mailto:zfp@llnl.gov>`__ with requests for
features not listed above.
