.. include:: defs.rst

.. index::
   single: Parallel execution
.. _execution:

Parallel Execution
==================

As of |zfp| |omprelease|, parallel compression (but not decompression) is
supported on multicore processors via `OpenMP <http://www.openmp.org>`_
threads.  Since |zfp| partitions arrays into small independent blocks, a
large amount of data parallelism is inherent in the compression scheme that
can be exploited.  In principle, concurrency is limited only by the number
of blocks that make up an array, though in practice each thread is
responsible for compressing a *chunk* of several contiguous blocks.

NOTE: |zfp| parallel compression is confined to shared memory on a single
compute node.  No effort is made to coordinate compression across distributed
memory on networked compute nodes, although |zfp|'s fine-grained partitioning
of arrays should facilitate distributed parallel compression.

This section describes the |zfp| parallel compression algorithm and explains
how to configure |libzfp| and enable parallel compression at run time via
its :ref:`high-level C API <hl-api>`.

Execution Policies
------------------

|zfp| supports multiple *execution policies*, which dictate how (e.g.,
sequentially, in parallel) and where (e.g., on the CPU or GPU) arrays are
compressed.  Currently two *execution policies* are available: :code:`serial`
and :code:`omp`.  The default mode is :code:`serial`, which ensures sequential
compression on a single thread.  The :code:`omp` execution policy
allows for data-parallel compression on multiple OpenMP threads.
Future versions of |zfp| will also support a
`CUDA <https://developer.nvidia.com/about-cuda>`_ execution policy.

The execution policy is set by :c:func:`zfp_stream_set_execution` and
pertains to a particular :c:type:`zfp_stream`.  Hence, each stream
(and array) may use a policy suitable for that stream.  For instance,
very small arrays are likely best compressed in serial, while parallel
compression is best reserved for very large arrays that can take the
most advantage of concurrent execution.

Execution Parameters
--------------------

Each execution policy allows tailoring the execution via the setting of
its associated *execution parameters*.  Examples include number of threads,
chunk size, scheduling, etc.  The :code:`serial` policy has no parameters.
The subsections below discuss the :code:`omp` parameters.

Whenever the execution policy is changed via
:c:func:`zfp_stream_set_execution`, its parameters (if any) are initialized
to their defaults, overwriting any prior setting.

OpenMP Thread Count
^^^^^^^^^^^^^^^^^^^

By default, the number of threads to use is given by the current setting
of the OpenMP internal control variable *nthreads-var*.  Unless the
calling thread has explicitly requested a thread count via the OpenMP
API, this control variable usually defaults to the number of threads
supported by the hardware (e.g. the number of available cores).

To set the number of requested threads to be used by |zfp|, which may
differ from the thread count of encapsulating or surrounding OpenMP
parallel regions, call :c:func:`zfp_stream_set_omp_threads`.

The user is advised to call the |zfp| API functions to modify OpenMP
behavior rather than make direct OpenMP calls.  For instance, use
:c:func:`zfp_stream_set_omp_threads` rather than
:code:`omp_set_num_threads()`.  To indicate that the current OpenMP
settings should be used, for instance as determined by the global
OpenMP environment variable :envvar:`OMP_NUM_THREADS`, pass a thread
count of zero to :c:func:`zfp_stream_set_omp_threads`.

Note that |zfp| does not modify *nthreads-var* or other control variables
but uses a :code:`num_threads` clause on the OpenMP :code:`#pragma` line.
Hence, no OpenMP state is changed and any subsequent OpenMP code is not
impacted by |zfp|'s parallel compression.

.. index::
   single: Chunks
.. _chunks:

OpenMP Chunk Size
^^^^^^^^^^^^^^^^^

The *d*-dimensional array is partitioned into *chunks*, with each chunk
representing a contiguous sequence of :ref:`blocks <algorithm>` of |4powd|
array elements each.  Chunks represent the unit of parallel work assigned
to a thread, although in practice a thread often processes more than one
chunk.  The chunk size, in number of |zfp| blocks, varies depending on *d*,
and can be directly selected by the user only for 1D arrays.

For 2D arrays of dimensions *nx* |times| *ny*, each chunk corresponds
to 4 |times| *nx* values.  There are, thus, *ny* / 4 total chunks and
work for at most that many threads.  Hence, it is important that *ny* is
large enough to engage the requested number of threads.  If *ny* is small
but *nx* is large, then it may make sense to transpose the array before
compressing it.

For 3D arrays of dimensions *nx* |times| *ny* |times| *nz*, a similar
strategy is employed whereby the array is partitioned into *nz* / 4
layers of blocks.  As with 2D arrays, it is important that *nz* is
large enough to expose enough parallelism.

A generalization of the above work partitioning scheme to 1D would yield
one block per chunk (and thread), which not only introduces a large
thread creation overhead but also a significant storage overhead, as
in most instances memory is allocated independently for each chunk
(see below for exceptions).  The compressed chunks must then be
concatenated into a single stream, which adds additional overhead.

For these reasons, 1D compression makes explicit use of a user-selected
chunk size.  By default, this chunk size is |chunksize| blocks (1024
scalars), which can be overridden either at compile time by setting the
:c:macro:`ZFP_OMP_CHUNK_SIZE` macro or at run time by calling
:c:func:`zfp_stream_set_omp_chunk_size`.

OpenMP Scheduling
^^^^^^^^^^^^^^^^^

|zfp| uses static OpenMP scheduling.  By default, each thread is assigned
consecutive chunks and, therefore, a single contiguous portion of the array
being compressed.  This promotes good cache reuse and reduces NUMA traffic.
On the other hand, since compression throughput is determined almost solely
by compression ratio, such work scheduling may lead to load imbalance and
poor parallel efficiency if the compression ratio varies significantly over
the array, leaving threads idle once their consecutive chunks have been
processed.

When this is the case and the number of chunks is significantly larger
than the number of threads, it may be beneficial to interleave chunks
so that each thread processes chunks that are distributed over the whole
array.  To enable such interleaved scheduling, define the compile-time
macro :c:macro:`ZFP_OMP_INTERLEAVE`.

Fixed- vs. Variable-Rate Compression
------------------------------------

Following partitioning into chunks, |zfp| assigns each chunk to a thread.
If there are more chunks than threads supported, chunks are processed in
round robin fashion.

In :ref:`variable-rate mode <modes>`, there is no way to predict the exact
number of bits that each chunk compresses to.  Therefore, |zfp| allocates
a temporary memory buffer for each chunk.  Once all chunks have been
compressed, they are concatenated into a single bit stream in serial,
after which the temporary buffers are deallocated.

In :ref:`fixed-rate mode <mode-fixed-rate>`, the final location of each
chunk's bit stream is known ahead of time, and |zfp| may not have to
allocate temporary buffers.  However, if the chunks are not aligned on
:ref:`word boundaries <bs-api>`, then race conditions may occur.  In other
words, for chunk size *C*, rate *R*, and word size *W*, the rate and chunk
size must be such that *C* |times| |4powd| |times| *R* is a multiple of *W*
to avoid temporary buffers.  Since *W* is a small power of two no larger
than 64, this is usually an easy requirement to satisfy.

When chunks are whole multiples of the word size, no temporary buffers
are allocated and the threads write compressed data directly to the
target buffer.

Enabling OpenMP
---------------

In order to support parallel compression, |zfp| must be compiled with
OpenMP support.  If built with CMake, OpenMP support is automatically
enabled when available.  To manually disable OpenMP support, see the
:c:macro:`ZFP_WITH_OPENMP` macro.

To avoid compilation errors on systems with spotty OpenMP support
(e.g. macOS), OpenMP is by default disabled in GNU builds.  To enable
OpenMP, edit the :file:`Config` file and see instructions on how to
set the :c:macro:`ZFP_WITH_OPENMP` macro.

Setting the Execution Policy
----------------------------

Enabling OpenMP parallel compression at run time is often as simple as
calling :c:func:`zfp_stream_set_execution`
::

    if (zfp_stream_set_execution(stream, zfp_exec_omp)) {
      // use OpenMP parallel compression
      ...
      zfpsize = zfp_compress(stream, field);
    }

before calling :c:func:`zfp_compress`.  If OpenMP is disabled or not
supported, then the return value of functions setting the :code:`omp`
execution policy and parameters will indicate failure.  Execution
parameters are optional and may be set using the functions discussed
above.

The source code for the |zfpcmd| command-line tool includes further examples
on how to set the execution policy.  To use parallel compression in this
tool, see the :option:`-x` command-line option.


Parallel Compression
--------------------

Once the execution policy and parameters have been selected, compression
is executed by calling :c:func:`zfp_compress` from a single thread.  This
function in turn inspects the execution policy given by the
:c:type:`zfp_stream` argument and dispatches the appropriate function
for executing compression.


Parallel Decompression
----------------------

Parallel decompression is in principle possible using the same strategy
as used for compression.  However, in |zfp|'s
:ref:`variable-rate modes <modes>`, the compressed blocks do not occupy
fixed storage, and therefore the decompressor needs to be instructed
where each compressed block resides in the bit stream to enable
parallel decompression.  Because the |zfp| bit stream does not currently
store such information, parallel decompression is not yet supported.

Future versions of |zfp| will allow efficient encoding of block sizes and/or
offsets to allow each thread to quickly locate the blocks it is responsible
for decompressing.
