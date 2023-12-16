.. include:: defs.rst
.. _arrays:

Compressed-Array C++ Classes
============================

.. cpp:namespace:: zfp

|zfp|'s compressed arrays are C++ classes, plus :ref:`C wrappers <cfp>` around
these classes, that implement random-accessible single- and multi-dimensional
floating-point arrays.  Since its first release, |zfp| provides *fixed-rate*
arrays, :code:`zfp::array`, that support both read and write access to
individual array elements.  As of |carrrelease|, |zfp| also supports
read-only arrays, :code:`zfp::const_array`, for data that is static or is
updated only infrequently.  The read-only arrays support all of
|zfp|'s :ref:`compression modes <modes>` including variable-rate
and lossless compression.

For fixed-rate arrays, the storage size, specified in number of bits per
array element, is set by the user.  Such arbitrary storage is achieved via
|zfp|'s lossy :ref:`fixed-rate compression <mode-fixed-rate>` mode, by
partitioning each *d*-dimensional array into blocks of |4powd| values
and compressing each block to a fixed number of bits.  The more smoothly
the array values vary along each dimension, the more accurately |zfp| can
represent them.  In other words, these arrays are not suitable for
representing data where adjacent elements are not correlated.  Rather,
the expectation is that the array represents a regularly sampled and
predominantly continuous function, such as a temperature field in a physics
simulation.

The *rate*, measured in number of bits per array element, can be specified
in fractions of a bit (but see FAQs :ref:`#12 <q-granularity>` and
:ref:`#18 <q-rate>` for limitations).  |zfp| supports 1D, 2D, 3D, and (as
of version |4darrrelease|) 4D arrays.  For higher-dimensional arrays,
consider using an array of |zfp| arrays.  Note that array dimensions need not
be multiples of four; |zfp| transparently handles partial blocks on array
boundaries.

Read-only arrays allow setting compression mode and parameters on
construction, and can optionally be initialized with uncompressed data.
These arrays do not allow updating individual array elements, though
the contents of the whole array may be updated by re-compressing and
overwriting the array.  This may be useful in applications that decompress
the whole array, perform a computation that updates its contents (e.g.,
a stencil operation that advances the solution of a PDE), and then compress
to memory the updated array.

The C++ templated array classes are implemented entirely as header files
that call the |zfp| C library to perform compression and decompression.
These arrays cache decompressed blocks to reduce the number of compression
and decompression calls.  Whenever an array value is read, the corresponding
block is first looked up in the cache, and if found the uncompressed value
is returned.  Otherwise the block is first decompressed and stored in the
cache.  Whenever an array element is written (whether actually modified or
not), a "dirty bit" is set with its cached block to indicate that the block
must be compressed back to persistent storage when evicted from the cache.

This section documents the public interface to the array classes, including
base classes and member accessor classes like proxy references/pointers,
iterators, and views.

The following sections are available:

* :ref:`array_classes`
* :ref:`carray_classes`
* :ref:`caching`
* :ref:`serialization`
* :ref:`references`
* :ref:`pointers`
* :ref:`iterators`
* :ref:`views`
* :ref:`codec`
* :ref:`index`


.. _array_classes:

Read-Write Fixed-Rate Arrays
----------------------------

There are eight array classes for 1D, 2D, 3D, and 4D read-write arrays,
each of which can represent single- or double-precision values.
Although these arrays store values in a form different from conventional
single- and double-precision floating point, the user interacts with the
arrays via floats and doubles.

The array classes can often serve as direct substitutes for C/C++
single- and multi-dimensional floating-point arrays and STL vectors, but
have the benefit of allowing fine control over storage size.  All classes
below belong to the :cpp:any:`zfp` namespace.

.. note::
  Much of the compressed-array API was modified in |zfp| |64bitrelease|
  to support 64-bit indexing of very large arrays.  In particular, array
  dimensions and indices now use the :code:`size_t` type instead of
  :code:`uint` and strides use the :code:`ptrdiff_t` type instead of
  :code:`int`.

.. _array_base_class:

Base Class
^^^^^^^^^^

.. cpp:class:: array

  Virtual base class for common array functionality.

----

.. cpp:function:: zfp_type array::scalar_type() const

  Return the underlying scalar type (:c:type:`zfp_type`) of the array.

----

.. cpp:function:: uint array::dimensionality() const

  Return the dimensionality (aka. rank) of the array: 1, 2, 3, or 4.

----

.. cpp:function:: array::header array::get_header() const

  Deprecated function as of |zfp| |crpirelease|.  See the :ref:`header`
  section on how to construct a header.

----

.. _array_factory:
.. cpp:function:: static array* array::construct(const header& h, const void* buffer = 0, size_t buffer_size_bytes = 0)

  Construct a compressed-array object whose scalar type, dimensions, and rate
  are given by the :ref:`header <header>` *h*.  Return a base class pointer
  upon success.  The optional *buffer* points to compressed data that, when
  passed, is copied into the array.  If *buffer* is absent, the array is
  default initialized with all zeroes.  The optional *buffer_size_bytes*
  parameter specifies the buffer length in bytes.  When passed, a comparison
  is made to ensure that the buffer size is at least as large as the size
  implied by the header.  If this function fails for any reason, an
  :cpp:class:`exception` is thrown.


Common Methods
^^^^^^^^^^^^^^

The following methods are common to 1D, 2D, 3D, and 4D arrays, but are
implemented in the array class specific to each dimensionality rather than
in the base class.

.. cpp:function:: size_t array::size() const

  Total number of elements in array, e.g., *nx* |times| *ny* |times| *nz* for
  3D arrays.

----

.. cpp:function:: double array::rate() const

  Return rate in bits per value.

----

.. cpp:function:: double array::set_rate(double rate)

  Set desired compression rate in bits per value.  Return the closest rate
  supported.  See FAQ :ref:`#12 <q-granularity>` and FAQ :ref:`#18 <q-rate>`
  for discussions of the rate granularity.  This method destroys the previous
  contents of the array.

----

.. cpp:function:: size_t array::size_bytes(uint mask = ZFP_DATA_ALL) const

  Return storage size of components of array data structure indicated by
  *mask*.  The mask is constructed via bitwise OR of
  :ref:`predefined constants <data-macros>`.
  Available as of |zfp| |carrrelease|.

----

.. cpp:function:: size_t array::compressed_size() const

  Return number of bytes of storage for the compressed data.  This amount
  does not include the small overhead of other class members or the size
  of the cache.  Rather, it reflects the size of the memory buffer
  returned by :cpp:func:`compressed_data`.

----

.. cpp:function:: void* array::compressed_data() const

  Return pointer to compressed data for read or write access.  The size
  of the buffer is given by :cpp:func:`compressed_size`.

.. note::
  As of |zfp| |crpirelease|, the return value is :code:`void*` rather than
  :code:`uchar*` to simplify pointer conversion and to dispel any misconception
  that the compressed data needs only :code:`uchar` alignment.  Compressed
  streams are always word aligned (see :c:var:`stream_word_bits` and
  :c:macro:`BIT_STREAM_WORD_TYPE`).

----

.. cpp:function:: size_t array::cache_size() const

  Return the cache size in number of bytes.

----

.. cpp:function:: void array::set_cache_size(size_t bytes)

  Set minimum cache size in bytes.  The actual size is always a power of two
  bytes and consists of at least one block.  If *bytes* is zero, then a
  default cache size is used, which requires the array dimensions to be known.

----

.. cpp:function:: void array::clear_cache() const

  Empty cache without compressing modified cached blocks, i.e., discard any
  cached updates to the array.

----

.. cpp:function:: virtual void array::flush_cache() const

  Flush cache by compressing all modified cached blocks back to persistent
  storage and emptying the cache.  This method should be called before
  writing the compressed representation of the array to disk, for instance.

----

.. cpp:function:: void array::get(Scalar* p) const

  Decompress entire array and store at *p*, for which sufficient storage must
  have been allocated.  The uncompressed array is assumed to be contiguous
  (with default strides) and stored in the usual "row-major" order, i.e., with
  *x* varying faster than *y*, *y* varying faster than *z*, etc.

----

.. cpp:function:: void array::set(const Scalar* p)

  Initialize array by copying and compressing data stored at *p*.  The
  uncompressed data is assumed to be stored as in the :cpp:func:`get`
  method.  If *p* = 0, then the array is zero-initialized.

----

.. cpp:function:: const_reference array::operator[](size_t index) const

  Return :ref:`const reference <references>` to scalar stored at given flat
  index (inspector).  For a 3D array, :code:`index = x + nx * (y + ny * z)`.

.. note::
  As of |zfp| |crpirelease|, the return value is no longer :code:`Scalar` but
  is a :ref:`const reference <references>` to the corresponding array element
  (conceptually equivalent to :code:`const Scalar&`).  This API change was
  necessary to allow obtaining a const pointer to the element when the array
  itself is const qualified, e.g., :code:`const_pointer p = &a[index];`.

----

.. _lvref_idx:
.. cpp:function:: reference array::operator[](size_t index)

  Return :ref:`proxy reference <references>` to scalar stored at given flat
  index (mutator).  For a 3D array, :code:`index = x + nx * (y + ny * z)`.

----

.. cpp:function:: iterator array::begin()

  Return random-access mutable iterator to beginning of array.

----

.. cpp:function:: iterator array::end()

  Return random-access mutable iterator to end of array.  As with STL iterators,
  the end points to a virtual element just past the last valid array element.

----

.. cpp:function:: const_iterator array::begin() const
.. cpp:function:: const_iterator array::cbegin() const

  Return random-access const iterator to beginning of array.

----

.. cpp:function:: const_iterator array::end() const
.. cpp:function:: const_iterator array::cend() const

  Return random-access const iterator to end of array.

.. note::
  Const :ref:`references <references>`, :ref:`pointers <pointers>`, and
  :ref:`iterators <iterators>` are available as of |zfp| |crpirelease|.  

1D, 2D, 3D, and 4D Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^

Below are classes and methods specific to each array dimensionality and
template scalar type (:code:`float` or :code:`double`).  Since the classes
and methods share obvious similarities regardless of dimensionality, only
one generic description for all dimensionalities is provided.

Note: In the class declarations below, the class template for the scalar
type is omitted for readability, e.g.,
:code:`class array1` is used as shorthand for
:code:`template <typename Scalar> class array1`.  Wherever the type
:code:`Scalar` appears, it refers to this template argument.

..
  .. cpp:class:: template<typename Scalar> array1 : public array
  .. cpp:class:: template<typename Scalar> array2 : public array
  .. cpp:class:: template<typename Scalar> array3 : public array
  .. cpp:class:: template<typename Scalar> array4 : public array

.. cpp:class:: array1 : public array
.. cpp:class:: array2 : public array
.. cpp:class:: array3 : public array
.. cpp:class:: array4 : public array

  This is a 1D, 2D, 3D, or 4D array that inherits basic functionality
  from the generic :cpp:class:`array` base class.  The template argument,
  :cpp:type:`Scalar`, specifies the floating type returned for array
  elements.  The suffixes :code:`f` and :code:`d` can also be appended
  to each class to indicate float or double type, e.g.,
  :cpp:class:`array1f` is a synonym for :cpp:class:`array1\<float>`.

----

.. cpp:class:: arrayANY : public array

  Fictitious class used to refer to any one of :cpp:class:`array1`,
  :cpp:class:`array2`, :cpp:class:`array3`, and :cpp:class:`array4`.
  This class is not part of the |zfp| API.

----

.. _array_ctor_default:
.. cpp:function:: array1::array1()
.. cpp:function:: array2::array2()
.. cpp:function:: array3::array3()
.. cpp:function:: array4::array4()

  Default constructor.  Creates an empty array whose size and rate are both
  zero.

.. note::
  The default constructor is useful when the array size or rate is not known at
  time of construction.  Before the array can become usable, however, it must
  be :ref:`resized <array_resize>` and its rate must be set via
  :cpp:func:`array::set_rate`.  These two tasks can be performed in either
  order.  Furthermore, the desired cache size should be set using
  :cpp:func:`array::set_cache_size`, as the default constructor creates a
  cache that holds only one |zfp| block, i.e., the minimum possible.

----

.. _array_ctor:
.. cpp:function:: array1::array1(size_t n, double rate, const Scalar* p = 0, size_t cache_size = 0)
.. cpp:function:: array2::array2(size_t nx, size_t ny, double rate, const Scalar* p = 0, size_t cache_size = 0)
.. cpp:function:: array3::array3(size_t nx, size_t ny, size_t nz, double rate, const Scalar* p = 0, size_t cache_size = 0)
.. cpp:function:: array4::array4(size_t nx, size_t ny, size_t nz, size_t nw, double rate, const Scalar* p = 0, size_t cache_size = 0)

  Constructor of array with dimensions *n* (1D), *nx* |times| *ny* (2D),
  *nx* |times| *ny* |times| *nz* (3D), or
  *nx* |times| *ny* |times| *nz* |times| *nw* (4D) using *rate* bits per
  value, at least *cache_size* bytes of cache, and optionally initialized
  from flat, uncompressed array *p*.  If *cache_size* is zero, a default
  cache size suitable for the array dimensions is chosen.

----

.. _array_ctor_header:
.. cpp:function:: array1::array1(const array::header& h, const void* buffer = 0, size_t buffer_size_bytes = 0)
.. cpp:function:: array2::array2(const array::header& h, const void* buffer = 0, size_t buffer_size_bytes = 0)
.. cpp:function:: array3::array3(const array::header& h, const void* buffer = 0, size_t buffer_size_bytes = 0)
.. cpp:function:: array4::array4(const array::header& h, const void* buffer = 0, size_t buffer_size_bytes = 0)

  Constructor from previously :ref:`serialized <serialization>` compressed
  array.  The :ref:`header <header>`, *h*, contains array metadata, while the
  optional *buffer* points to the compressed data that is to be copied to the
  array.  The optional *buffer_size_bytes* parameter specifies the *buffer*
  length.  If the constructor fails, an :ref:`exception <exception>` is thrown.
  See :cpp:func:`array::construct` for further details on the *buffer* and
  *buffer_size_bytes* parameters.

----

.. _array_copy_constructor:
.. cpp:function:: array1::array1(const array1& a)
.. cpp:function:: array2::array2(const array2& a)
.. cpp:function:: array3::array3(const array3& a)
.. cpp:function:: array4::array4(const array4& a)

  Copy constructor.  Performs a deep copy.

----

.. cpp:function:: virtual array1::~array1()
.. cpp:function:: virtual array2::~array2()
.. cpp:function:: virtual array3::~array3()
.. cpp:function:: virtual array4::~array4()

  Virtual destructor (allows for inheriting from |zfp| arrays).

----

.. _array_copy:
.. cpp:function:: array1& array1::operator=(const array1& a)
.. cpp:function:: array2& array2::operator=(const array2& a)
.. cpp:function:: array3& array3::operator=(const array3& a)
.. cpp:function:: array4& array4::operator=(const array4& a)

  Assignment operator.  Performs a deep copy.

----

.. _array_dims:
.. cpp:function:: size_t array2::size_x() const
.. cpp:function:: size_t array2::size_y() const
.. cpp:function:: size_t array3::size_x() const
.. cpp:function:: size_t array3::size_y() const
.. cpp:function:: size_t array3::size_z() const
.. cpp:function:: size_t array4::size_x() const
.. cpp:function:: size_t array4::size_y() const
.. cpp:function:: size_t array4::size_z() const
.. cpp:function:: size_t array4::size_w() const

  Return array dimensions.

----

.. _array_resize:
.. cpp:function:: void array1::resize(size_t n, bool clear = true)
.. cpp:function:: void array2::resize(size_t nx, size_t ny, bool clear = true)
.. cpp:function:: void array3::resize(size_t nx, size_t ny, size_t nz, bool clear = true)
.. cpp:function:: void array4::resize(size_t nx, size_t ny, size_t nz, size_t nw, bool clear = true)

  Resize the array (all previously stored data will be lost).  If *clear* is
  true, then the array elements are all initialized to zero.

.. note::
  It is often desirable (though not a requirement) to also set the cache size
  when resizing an array, e.g., in proportion to the array size;
  see :cpp:func:`array::set_cache_size`.  This is particularly important when
  the array is default constructed, which initializes the cache size to the
  minimum possible of only one |zfp| block.

----

.. _array_accessor:
.. cpp:function:: const_reference array1::operator()(size_t i) const
.. cpp:function:: const_reference array2::operator()(size_t i, size_t j) const
.. cpp:function:: const_reference array3::operator()(size_t i, size_t j, size_t k) const
.. cpp:function:: const_reference array4::operator()(size_t i, size_t j, size_t k, size_t l) const

  Return const reference to element stored at multi-dimensional index given by
  *i*, *j*, *k*, and *l* (inspector).

.. note::
  As of |zfp| |crpirelease|, the return value is no longer :code:`Scalar` but
  is a :ref:`const reference <references>` to the corresponding array element
  (essentially equivalent to :code:`const Scalar&`).  This API change was
  necessary to allow obtaining a const pointer to the element when the array
  itself is const qualified, e.g.,
  :code:`const_pointer p = &a(i, j, k);`.

----

.. _lvref:
.. cpp:function:: reference array1::operator()(size_t i)
.. cpp:function:: reference array2::operator()(size_t i, size_t j)
.. cpp:function:: reference array3::operator()(size_t i, size_t j, size_t k)
.. cpp:function:: reference array4::operator()(size_t i, size_t j, size_t k, size_t l)

  Return :ref:`proxy reference <references>` to scalar stored at
  multi-dimensional index given by *i*, *j*, *k*, and *l* (mutator).


.. _carray_classes:

Read-Only Variable-Rate Arrays
------------------------------

Read-only arrays are preferable in applications that store static data,
e.g., constant tables or simulation output, or data that is updated only
periodically as a whole, such as when advancing the solution
of a partial differential equation.  Because such updates have to be applied
to the whole array, one may choose to tile large arrays into smaller |zfp|
arrays to support finer granularity updates.  Read-only arrays have the
benefit of supporting all of |zfp|'s :ref:`compression modes <modes>`, most
of which provide higher accuracy per bit stored than fixed-rate mode.

The read-only arrays share an API with the read-write fixed-rate arrays,
with only a few differences:

- All methods other than those that specify array-wide settings, such as
  compression mode and parameters, array dimensions, and array contents,
  are :code:`const` qualified.  There are, thus, no methods for obtaining
  a writeable reference, pointer, or iterator.  Consequently, one may not
  initialize such arrays one element at a time.  Rather, the user initializes
  the whole array by passing a pointer to uncompressed data.

- Whereas the constructors for fixed-rate arrays accept a *rate* parameter,
  the read-only arrays allow specifying any compression mode and
  corresponding parameters (if any) via a :c:type:`zfp_config` object.

- Additional methods are available for setting and querying compression
  mode and parameters after construction.

- Read-only arrays are templated on a block index class that encodes the
  bit offset to each block of data.  Multiple index classes are available
  that trade compactness and speed of access.  The default :cpp:class:`hybrid4`
  index represents 64-bit offsets using only 24 bits of amortized storage per
  block.  An "implicit" index is available for fixed-rate read-only arrays,
  which computes rather than stores offsets to equal-sized blocks.

.. note::
  Whereas variable-rate compression almost always improves accuracy per bit
  of compressed data over fixed rate, one should also weigh the storage and
  compute overhead associated with the block index needed for variable-rate
  storage.  The actual storage overhead can be determined by passing
  :c:macro:`ZFP_DATA_INDEX` to :cpp:func:`const_array::size_bytes`.  This
  overhead tends to be small for 3D and 4D arrays.

Array initialization may be done at construction time, by passing a pointer
to uncompressed data, or via the method :cpp:func:`const_array::set`,
which overwrites the contents of the whole array.  This method may be
called more than once to update (i.e., re-initialize) the array.

Read-only arrays support a subset of references, pointers, iterators, and
views; in particular those with a :code:`const_` prefix.

Currently, not all capabilities of read-write arrays are available for
read-only arrays.  For example, (de)serialization and construction from a
view have not yet been implemented, and there are no C bindings.

Read-only arrays derive from the :ref:`array base class <array_base_class>`.
Additional methods are documented below.

..
  .. cpp:class:: template<typename Scalar> const_array1 : public array
  .. cpp:class:: template<typename Scalar> const_array2 : public array
  .. cpp:class:: template<typename Scalar> const_array3 : public array
  .. cpp:class:: template<typename Scalar> const_array4 : public array

.. cpp:class:: const_array1 : public array
.. cpp:class:: const_array2 : public array
.. cpp:class:: const_array3 : public array
.. cpp:class:: const_array4 : public array

  1D, 2D, 3D, or 4D read-only array that inherits basic functionality
  from the generic :cpp:class:`array` base class.  The template argument,
  :cpp:type:`Scalar`, specifies the floating type returned for array
  elements.  The suffixes :code:`f` and :code:`d` can also be appended
  to each class to indicate float or double type, e.g.,
  :cpp:class:`const_array1f` is a synonym for
  :cpp:class:`const_array1\<float>`.

----

.. cpp:class:: const_array : public array

  Fictitious class used to denote one of the 1D, 2D, 3D, and 4D read-only
  array classes.  This pseudo base class serves only to document the API
  shared among the four arrays.

----

.. _carray_ctor_default:
.. cpp:function:: const_array1::const_array1()
.. cpp:function:: const_array2::const_array2()
.. cpp:function:: const_array3::const_array3()
.. cpp:function:: const_array4::const_array4()

  Default constructor.  Creates an empty array whose size is zero and whose
  compression mode is unspecified.  The array's cache size is initialized to
  the minimum possible, which can have performance implications; see
  :ref:`this note <array_ctor_default>`.

----

.. _carray_ctor:
.. cpp:function:: const_array1::const_array1(size_t n, const zfp_config& config, const Scalar* p = 0, size_t cache_size = 0)
.. cpp:function:: const_array2::const_array2(size_t nx, size_t ny, const zfp_config& config, const Scalar* p = 0, size_t cache_size = 0)
.. cpp:function:: const_array3::const_array3(size_t nx, size_t ny, size_t nz, const zfp_config& config, const Scalar* p = 0, size_t cache_size = 0)
.. cpp:function:: const_array4::const_array4(size_t nx, size_t ny, size_t nz, size_t nw, const zfp_config& config, const Scalar* p = 0, size_t cache_size = 0)

  Constructor of array with dimensions *n* (1D), *nx* |times| *ny* (2D),
  *nx* |times| *ny* |times| *nz* (3D), or
  *nx* |times| *ny* |times| *nz* |times| *nw* (4D).  The compression mode and
  parameters are given by *config* (see :ref:`configuration <hl-func-config>`).
  The array uses at least *cache_size* bytes of cache, and is optionally
  initialized from flat, uncompressed array *p*.  If *cache_size* is zero,
  a default cache size suitable for the array dimensions is chosen.

----

.. cpp:function:: const_array1::const_array1(const const_array1& a)
.. cpp:function:: const_array2::const_array2(const const_array2& a)
.. cpp:function:: const_array3::const_array3(const const_array3& a)
.. cpp:function:: const_array4::const_array4(const const_array4& a)

  Copy constructor.  Performs a deep copy.

----

.. cpp:function:: virtual const_array1::~const_array1()
.. cpp:function:: virtual const_array2::~const_array2()
.. cpp:function:: virtual const_array3::~const_array3()
.. cpp:function:: virtual const_array4::~const_array4()

  Virtual destructor (allows for inheritance).

----

.. _carray_copy:
.. cpp:function:: const_array1& const_array1::operator=(const const_array1& a)
.. cpp:function:: const_array2& const_array2::operator=(const const_array2& a)
.. cpp:function:: const_array3& const_array3::operator=(const const_array3& a)
.. cpp:function:: const_array4& const_array4::operator=(const const_array4& a)

  Assignment operator.  Performs a deep copy.

----

.. cpp:function:: size_t const_array::size() const

  Total number of elements in array, e.g., *nx* |times| *ny* |times| *nz* for
  3D arrays.

----

.. _carray_dims:
.. cpp:function:: size_t const_array2::size_x() const
.. cpp:function:: size_t const_array2::size_y() const
.. cpp:function:: size_t const_array3::size_x() const
.. cpp:function:: size_t const_array3::size_y() const
.. cpp:function:: size_t const_array3::size_z() const
.. cpp:function:: size_t const_array4::size_x() const
.. cpp:function:: size_t const_array4::size_y() const
.. cpp:function:: size_t const_array4::size_z() const
.. cpp:function:: size_t const_array4::size_w() const

  Return array dimensions.

----

.. _carray_resize:
.. cpp:function:: void const_array1::resize(size_t n, bool clear = true)
.. cpp:function:: void const_array2::resize(size_t nx, size_t ny, bool clear = true)
.. cpp:function:: void const_array3::resize(size_t nx, size_t ny, size_t nz, bool clear = true)
.. cpp:function:: void const_array4::resize(size_t nx, size_t ny, size_t nz, size_t nw, bool clear = true)

  Resize the array (all previously stored data will be lost).  If *clear* is
  true, then the array elements are all initialized to zero.  See also
  :ref:`this note <array_resize>`.

----

.. cpp:function:: zfp_mode const_array::mode() const

  Currently selected :ref:`compression mode <mode_struct>`.  If not yet
  specified, :code:`zfp_mode_null` is returned.

----

.. cpp:function:: double const_array::rate() const

  Return rate in compressed bits per value when
  :ref:`fixed-rate mode <mode-fixed-rate>` is enabled, else zero.

----

.. cpp:function:: uint const_array::precision() const

  Return precision in uncompressed bits per value when
  :ref:`fixed-precision mode <mode-fixed-precision>` is enabled, else zero.

----

.. cpp:function:: double const_array::accuracy() const

  Return accuracy as absolute error tolerance when
  :ref:`fixed-accuracy mode <mode-fixed-accuracy>` is enabled, else zero.

----

.. cpp:function:: void const_array::params(uint* minbits, uint* maxbits, uint* maxprec, int* minexp) const

  :ref:`Expert mode <mode-expert>` compression parameters (available for
  all compression modes).  Pointers may be :code:`null` if the corresponding
  parameter is not requested.

----

.. cpp:function:: double const_array::set_reversible()

  Enable :ref:`reversible mode <mode-reversible>`.  This method destroys
  the previous contents of the array.

----

.. cpp:function:: double const_array::set_rate(double rate)

  Set desired rate in compressed bits per value (enables
  :ref:`fixed-rate mode <mode-fixed-rate>`).  This method destroys the
  previous contents of the array.  See also :cpp:func:`array::set_rate`.

.. note::
  Whereas the :ref:`read-write fixed-rate arrays <array_classes>`
  (:cpp:class:`zfp::array`) require that block storage is word aligned, the
  read-only arrays (:cpp:class:`zfp::const_array`) are not subject to such
  restrictions and therefore support finer rate granularity.  For a
  *d*-dimensional :cpp:class:`const_array`, the rate granularity is
  4\ :sup:`-d` bits/value, e.g., a quarter bit/value for 1D arrays.

----

.. cpp:function:: uint const_array::set_precision(uint precision)

  Set desired precision in uncompressed bits per value (enables
  :ref:`fixed-precision mode <mode-fixed-precision>`).  This method destroys
  the previous contents of the array.

----

.. cpp:function:: double const_array::set_accuracy(double tolerance)

  Set desired accuracy as absolute error tolerance (enables
  :ref:`fixed-accuracy mode <mode-fixed-accuracy>`).  This method destroys
  the previous contents of the array.

----

.. cpp:function:: bool const_array::set_params(uint minbits, uint maxbits, uint maxprec, int minexp)

  Set :ref:`expert mode <mode-expert>` parameters.  This method destroys the
  previous contents of the array.  Return whether the codec supports the
  combination of parameters.

----

.. cpp:function:: void const_array::set_config(const zfp_config& config)

  Set compression mode and parameters given by *config*
  (see :ref:`configuration <hl-func-config>`).  This is a more general
  method for setting compression parameters such as rate, precision, accuracy,
  and :ref:`expert mode <mode-expert>` parameters.

----

.. cpp:function:: size_t const_array::size_bytes(uint mask = ZFP_DATA_ALL) const

  Return storage size of components of array data structure indicated by
  *mask*.  The mask is constructed via bitwise OR of
  :ref:`predefined constants <data-macros>`.

----

.. cpp:function:: size_t const_array::compressed_size() const

  Return number of bytes of storage for the compressed data.  This amount
  does not include the small overhead of other class members or the size
  of the cache.  Rather, it reflects the size of the memory buffer
  returned by :cpp:func:`compressed_data`.

----

.. cpp:function:: void* const_array::compressed_data() const

  Return pointer to compressed data for read or write access.  The size
  of the buffer is given by :cpp:func:`compressed_size`.

----

.. cpp:function:: size_t const_array::cache_size() const

  Return the cache size in number of bytes.

----

.. cpp:function:: void const_array::set_cache_size(size_t bytes)

  Set minimum cache size in bytes.  The actual size is always a power of two
  bytes and consists of at least one block.  If *bytes* is zero, then a
  default cache size is used, which requires the array dimensions to be known.

----

.. cpp:function:: void const_array::clear_cache() const

  Empty cache.

----

.. cpp:function:: void const_array::get(Scalar* p) const

  Decompress entire array and store at *p*, for which sufficient storage must
  have been allocated.  The uncompressed array is assumed to be contiguous
  (with default strides) and stored in the usual "row-major" order, i.e., with
  *x* varying faster than *y*, *y* varying faster than *z*, etc.

----

.. cpp:function:: void const_array::set(const Scalar* p, bool compact = true)

  Initialize array by copying and compressing floating-point data stored at
  *p*.  If *p* = 0, then the array is zero-initialized.  The uncompressed data
  is assumed to be stored as in the :cpp:func:`get` method.  Since the size of
  compressed data may not be known a priori, this method conservatively
  allocates enough space to hold it.  If *compact* is true, any unused storage
  for compressed data is freed after initialization.

----

.. _const_array_accessor:
.. cpp:function:: const_reference const_array1::operator()(size_t i) const
.. cpp:function:: const_reference const_array2::operator()(size_t i, size_t j) const
.. cpp:function:: const_reference const_array3::operator()(size_t i, size_t j, size_t k) const
.. cpp:function:: const_reference const_array4::operator()(size_t i, size_t j, size_t k, size_t l) const

  Return const reference to element stored at multi-dimensional index given by
  *i*, *j*, *k*, and *l* (inspector).

----

.. cpp:function:: const_reference const_array::operator[](size_t index) const

  Return :ref:`const reference <references>` to scalar stored at given flat
  index (inspector).  For a 3D array, :code:`index = x + nx * (y + ny * z)`.

----

.. cpp:function:: const_iterator const_array::begin() const
.. cpp:function::  const_iterator const_array::cbegin() const

  Return random-access const iterator to beginning of array.

----

.. cpp:function:: const_iterator end() const
.. cpp:function:: const_iterator cend() const

  Return random-access const iterator to end of array.

.. include:: caching.inc
.. include:: serialization.inc
.. include:: references.inc
.. include:: pointers.inc
.. include:: iterators.inc
.. include:: views.inc
.. include:: codec.inc
.. include:: index.inc
