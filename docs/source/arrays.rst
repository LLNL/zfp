.. include:: defs.rst
.. _arrays:

Compressed-Array C++ Classes
============================

.. cpp:namespace:: zfp

|zfp|'s compressed arrays are C++ classes, plus :ref:`C wrappers <cfp>` around
these classes, that implement random-accessible single- and multi-dimensional
floating-point arrays whose storage size, specified in number of bits per
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
* :ref:`caching`
* :ref:`serialization`
* :ref:`references`
* :ref:`pointers`
* :ref:`iterators`
* :ref:`views`


.. _array_classes:

Arrays
------

There are eight array classes for 1D, 2D, 3D, and 4D arrays, each of which
can represent single- or double-precision values.
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

Base Class
^^^^^^^^^^

.. cpp:class:: array

  Virtual base class for common array functionality.

----

.. cpp:function:: double array::rate() const

  Return rate in bits per value.

----

.. cpp:function:: double array::set_rate(double rate)

  Set desired compression rate in bits per value.  Return the closest rate
  supported.  See :ref:`FAQ #12 <q-granularity>` and :ref:`FAQ #18 <q-rate>`
  for discussions of the rate granularity.  This method destroys the previous
  contents of the array.

----

.. cpp:function:: virtual void array::clear_cache() const

  Empty cache without compressing modified cached blocks, i.e., discard any
  cached updates to the array.

----

.. cpp:function:: virtual void array::flush_cache() const

  Flush cache by compressing all modified cached blocks back to persistent
  storage and emptying the cache.  This method should be called before
  writing the compressed representation of the array to disk, for instance.

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

.. cpp:function:: uint array::dimensionality() const

  Return the dimensionality (aka. rank) of the array: 1, 2, 3, or 4.

----

.. cpp:function:: zfp_type array::scalar_type() const

  Return the underlying scalar type (:c:type:`zfp_type`) of the array.

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

----

Common Methods
^^^^^^^^^^^^^^

The following methods are common to 1D, 2D, 3D, and 4D arrays, but are
implemented in the array class specific to each dimensionality rather than
in the base class.

.. cpp:function:: size_t array::size() const

  Total number of elements in array, e.g., *nx* |times| *ny* |times| *nz* for
  3D arrays.

----

.. cpp:function:: size_t array::cache_size() const

  Return the cache size in number of bytes.

----

.. cpp:function:: void array::set_cache_size(size_t bytes)

  Set minimum cache size in bytes.  The actual size is always a power of two
  bytes and consists of at least one block.  If *bytes* is zero, then a
  default cache size is used, which requires the array dimensions to be known.

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
  method.

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

  Return mutable iterator to beginning of array.

----

.. cpp:function:: iterator array::end()

  Return mutable iterator to end of array.  As with STL iterators, the end
  points to a virtual element just past the last valid array element.

----

.. cpp:function:: const_iterator array::begin() const
.. cpp:function:: const_iterator array::cbegin() const

  Return const iterator to beginning of array.

----

.. cpp:function:: const_iterator array::end() const
.. cpp:function:: const_iterator array::cend() const

  Return const iterator to end of array.

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
  :cpp:func:`array::set_rate`.  These two tasks can be performed in either order.
  Furthermore, the desired cache size should be set using
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
  cache size is chosen.

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

.. include:: caching.inc
.. include:: serialization.inc
.. include:: references.inc
.. include:: pointers.inc
.. include:: iterators.inc
.. include:: views.inc
