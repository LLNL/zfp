.. _caching:

Caching
-------

As mentioned above, the array classes maintain a software write-back cache
of at least one uncompressed block.  When a block in this cache is evicted
(e.g., due to a conflict), it is compressed back to permanent storage only
if it was modified while stored in the cache.

The size cache to use is specified by the user and is an important
parameter that needs careful consideration in order to balance the extra
memory usage, performance, and quality (recall that data loss is incurred
only when a block is evicted from the cache and compressed).  Although the
best choice varies from one application to another, we suggest allocating
at least two "layers" of blocks, e.g., 2 |times| (*nx* / 4) |times| (*ny* / 4)
blocks for 3D arrays, for applications that stream through the array and
perform stencil computations such as gathering data from neighboring elements.
This allows limiting the cache misses to compulsory ones.  If the *cache_size*
parameter provided to the constructor is set to zero bytes, then a default
cache size of at least |sqrt|\ *n* blocks is used, where *n* is the total
number of blocks contained in the array.

The cache size can be set during construction, or can be set at a later time
via :cpp:func:`array::set_cache_size`.  Note that if *cache_size* = 0, then
the array dimensions must have already been specified for the default size
to be computed correctly.  When the cache is resized, it is first flushed
if not already empty.  The cache can also be flushed explicitly if desired
by calling :cpp:func:`array::flush_cache`.  To empty the cache without
compressing any cached data, call :cpp:func:`array::clear_cache`.  To query
the byte size of the cache, use :cpp:func:`array::cache_size`.

By default, a direct-mapped cache is used with a hash function that maps
block indices to cache lines.  A faster but more collision prone hash
can be enabled by defining the preprocessor macro
:c:macro:`ZFP_WITH_CACHE_FAST_HASH`.
A two-way skew-associative cache is enabled by defining the preprocessor
macro :c:macro:`ZFP_WITH_CACHE_TWOWAY`.
