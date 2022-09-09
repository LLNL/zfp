.. include:: defs.rst

.. index::
   single: Parallel execution
.. _execution:

Parallel Execution
==================

As of |zfp| |omprelease|, parallel compression (but not decompression) is
supported on multicore processors via `OpenMP <http://www.openmp.org>`_
threads.
|zfp| |cudarelease| adds `CUDA <https://developer.nvidia.com/about-cuda>`_
support for fixed-rate compression and decompression on the GPU.

Since |zfp| partitions arrays into small independent blocks, a
large amount of data parallelism is inherent in the compression scheme that
can be exploited.  In principle, concurrency is limited only by the number
of blocks that make up an array, though in practice each thread is
responsible for compressing a *chunk* of several contiguous blocks.

.. note::
  |zfp| parallel compression is confined to shared memory on a single
  compute node or GPU.  No effort is made to coordinate compression across
  distributed memory on networked compute nodes, although |zfp|'s fine-grained
  partitioning of arrays should facilitate distributed parallel compression.

This section describes the |zfp| parallel compression algorithm and explains
how to configure |libzfp| and enable parallel compression at run time via
its :ref:`high-level C API <hl-api>`.

.. note::
  Parallel compression is not supported via the :ref:`low-level API <ll-api>`,
  which ignores all execution policy settings and always executes in serial.

.. _exec-policies:

Execution Policies
------------------

|zfp| supports multiple *execution policies*, which dictate how (e.g.,
sequentially, in parallel) and where (e.g., on the CPU or GPU) arrays are
compressed.  Currently three execution policies are available:
``serial``, ``omp``, and ``cuda``.  The default mode is
``serial``, which ensures sequential compression on a single thread.
The ``omp`` and ``cuda`` execution policies allow for data-parallel
compression on multiple threads.

The execution policy is set by :c:func:`zfp_stream_set_execution` and
pertains to a particular :c:type:`zfp_stream`.  Hence, each stream
(and array) may use a policy suitable for that stream.  For instance,
very small arrays are likely best compressed in serial, while parallel
compression is best reserved for very large arrays that can take the
most advantage of concurrent execution.

As outlined in FAQ :ref:`#23 <q-parallel>`, the final compressed stream
is independent of execution policy.


Execution Parameters
--------------------

Each execution policy allows tailoring the execution via its associated
*execution parameters*.  Examples include number of threads, chunk size,
scheduling, etc.  The ``serial`` and ``cuda`` policies have no
parameters.  The subsections below discuss the ``omp`` parameters.

Whenever the execution policy is changed via
:c:func:`zfp_stream_set_execution`, its parameters (if any) are initialized
to their defaults, overwriting any prior setting.


OpenMP Thread Count
^^^^^^^^^^^^^^^^^^^

By default, the number of threads to use is given by the current setting
of the OpenMP internal control variable *nthreads-var*.  Unless the
calling thread has explicitly requested a thread count via the OpenMP
API, this control variable usually defaults to the number of threads
supported by the hardware (e.g., the number of available cores).

To set the number of requested threads to be used by |zfp|, which may
differ from the thread count of encapsulating or surrounding OpenMP
parallel regions, call :c:func:`zfp_stream_set_omp_threads`.

The user is advised to call the |zfp| API functions to modify OpenMP
behavior rather than make direct OpenMP calls.  For instance, use
:c:func:`zfp_stream_set_omp_threads` rather than
:code:`omp_set_num_threads()`.  To indicate that the current OpenMP
settings should be used, for instance as determined by the global
OpenMP environment variable :envvar:`OMP_NUM_THREADS`, pass a thread
count of zero (the default setting) to :c:func:`zfp_stream_set_omp_threads`.

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
to a thread.  By default, the array is partitioned so that each thread
processes one chunk.  However, the user may override this behavior by
setting the chunk size (in number of |zfp| blocks) via
:c:func:`zfp_stream_set_omp_chunk_size`.  See FAQ :ref:`#25 <q-omp-perf>`
for a discussion of chunk sizes and parallel performance.


OpenMP Scheduling
^^^^^^^^^^^^^^^^^

|zfp| does not specify how to schedule chunk processing.  The schedule
used is given by the OpenMP *def-sched-var* internal control variable.
If load balance is poor, it may be improved by using smaller chunks,
which may or may not impact performance depending on the OpenMP schedule
in use.  Future versions of |zfp| may allow specifying how threads are
mapped to chunks, whether to use static or dynamic scheduling, etc.


.. _exec-mode:

Fixed- vs. Variable-Rate Compression
------------------------------------

Following partitioning into chunks, |zfp| assigns each chunk to a thread.
If there are more chunks than threads supported, chunks are processed in
unspecified order.

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
The CUDA implementation uses atomics to avoid race conditions, and therefore
does not need temporary buffers, regardless of chunk alignment.


Using OpenMP
------------

In order to use OpenMP compression, |zfp| must be built with OpenMP
support.  If built with CMake, OpenMP support is automatically enabled when
available.  To manually disable OpenMP support, see the
:c:macro:`ZFP_WITH_OPENMP` macro.

To avoid compilation errors on systems with spotty OpenMP support
(e.g., macOS), OpenMP is by default disabled in GNU builds.  To enable
OpenMP, see :ref:`gnu_builds` and the :c:macro:`ZFP_WITH_OPENMP` macro.


Using CUDA
----------

CUDA support is by default disabled.  Enabling it requires an installation
of CUDA and a compatible host compiler.  Furthermore, the
:c:macro:`ZFP_WITH_CUDA` macro must be set and |zfp| must be built with
CMake.  See :c:macro:`ZFP_WITH_CUDA` for further details.

Device Memory Management
^^^^^^^^^^^^^^^^^^^^^^^^

The CUDA version of |zfp| supports both host and device memory.  If device
memory is allocated for fields or compressed streams, this is automatically
detected and handled in a consistent manner.  For example, with compression,
if host memory pointers are provided for both the field and compressed
stream, then device memory will transparently be allocated and the uncompressed
data will be copied to the GPU.  Once compression completes, the compressed
stream is copied back to the host and device memory is deallocated.  If both
pointers are device pointers, then no copies are made.  Additionally, any
combination of mixing host and device pointers is supported.

.. _cuda-limitations:

CUDA Limitations
^^^^^^^^^^^^^^^^

The CUDA implementation has a number of limitations:

* Only the :ref:`fixed-rate mode <mode-fixed-rate>` mode is supported.
  Other modes will be supported in a future release.
* 4D arrays are not supported.
* :ref:`Headers <header>` are not supported.  Any header already present in
  the stream will be silently overwritten on compression.
* |zfp| must be built with a :c:macro:`ZFP_BIT_STREAM_WORD_SIZE` of 64 bits.
* Although :ref:`strides <field>` are supported, fields must be contiguous
  when stored in host memory, i.e., with no unused memory addresses between
  the minimum and maximum address spanned by the field (see
  :c:func:`zfp_field_is_contiguous`).  This requirement avoids having to copy
  and allocate more temporary memory than needed to hold the array if it were
  not strided.  Note that the strides can still be arbitrary as long as they
  serve only to permute the array elements.  Moreover, this restriction
  applies only to the CUDA execution policy and the case where the
  uncompressed field resides on the host.

We expect to address these limitations over time.


Setting the Execution Policy
----------------------------

Enabling parallel compression at run time is often as simple as
calling :c:func:`zfp_stream_set_execution`
::

    if (zfp_stream_set_execution(stream, zfp_exec_omp)) {
      // use OpenMP parallel compression
      ...
      zfpsize = zfp_compress(stream, field);
    }

before calling :c:func:`zfp_compress`.  Replacing :code:`zfp_exec_omp`
with :code:`zfp_exec_cuda` enables CUDA execution.  If OpenMP or CUDA is
disabled or not supported, then the return value of functions setting these
execution policies and parameters will indicate failure.  Execution
parameters are optional and may be set using the functions discussed above.

The source code for the |zfpcmd| command-line tool includes further examples
on how to set the execution policy.  To use parallel compression and
decompression in this tool, see the :option:`-x` command-line option.

.. note::
  As of |zfp| |cudarelease|, the execution policy refers to both
  compression and decompression.  The OpenMP implementation does not
  yet support decompression, and hence :c:func:`zfp_decompress` will
  fail if the execution policy is not reset to :code:`zfp_exec_serial`
  before calling the decompressor.  Similarly, the CUDA implementation
  supports only fixed-rate mode and will fail if other compression modes
  are specified.

The following table summarizes which execution policies are supported
with which :ref:`compression modes <modes>`:

  +---------------------------------+---------+---------+---------+
  | (de)compression mode            | serial  | OpenMP  | CUDA    |
  +===============+=================+=========+=========+=========+
  |               | fixed rate      | |check| | |check| | |check| |
  |               +-----------------+---------+---------+---------+
  |               | fixed precision | |check| | |check| |         |
  | compression   +-----------------+---------+---------+---------+
  |               | fixed accuracy  | |check| | |check| |         |
  |               +-----------------+---------+---------+---------+
  |               | reversible      | |check| | |check| |         |
  +---------------+-----------------+---------+---------+---------+
  |               | fixed rate      | |check| |         | |check| |
  |               +-----------------+---------+---------+---------+
  |               | fixed precision | |check| |         |         |
  | decompression +-----------------+---------+---------+---------+
  |               | fixed accuracy  | |check| |         |         |
  |               +-----------------+---------+---------+---------+
  |               | reversible      | |check| |         |         |
  +---------------+-----------------+---------+---------+---------+

:c:func:`zfp_compress` and :c:func:`zfp_decompress` both return zero if the
current execution policy is not supported for the requested compression
mode.


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
store such information, variable-rate parallel decompression is not yet
supported, though plans are to make such functionality available in the
near future.

The CUDA implementation supports fixed-rate decompression.  OpenMP
fixed-rate decompression has been implemented and will be released in the
near future.

Future versions of |zfp| will allow efficient encoding of block sizes and/or
offsets to allow each thread to quickly locate the blocks it is responsible
for decompressing, which will allow for variable-rate compression and
decompression.  Such capabilities are already present in the implementation
of the |zfp| :ref:`read-only arrays <carray_classes>`.
