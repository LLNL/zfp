.. index::
   single: Serialization
.. _serialization:

Serialization
-------------

.. cpp:namespace:: zfp

|zfp|'s read-write compressed arrays can be serialized to sequential,
contiguous storage and later recovered back into an object, e.g., to support
I/O of compressed-array objects.  Two pieces of information are needed
to describe a |zfp| array: the raw compressed data, obtained via
:cpp:func:`array::compressed_data` and :cpp:func:`array::compressed_size`,
and a :ref:`header <header>` that describes the array scalar type,
dimensions, and rate.
The user may concatenate the header and compressed data to form a
fixed-rate byte stream that can be read by the |zfp|
:ref:`command-line tool <zfpcmd>`.  When serializing the array,
the user should first call :cpp:func:`array::flush_cache` before
accessing the raw compressed data.

There are two primary ways to construct a compressed-array object from
compressed data: via array-specific :ref:`constructors <array_ctor_header>`
and via a generic :ref:`factory function <array_factory>`:

- When the array scalar type (i.e., :code:`float` or :code:`double`) and
  dimensionality (i.e., 1D, 2D, 3D, or 4D) are already known, the corresponding
  array :ref:`constructor <array_ctor_header>` may be used.  If the
  scalar type and dimensionality stored in the header do not match
  the array class, then an :ref:`exception <exception>` is thrown.

- |zfp| provides a :ref:`factory function <array_factory>` that can be used
  when the serialized array type is unknown but described in the header.
  This function returns a pointer to the abstract base class,
  :cpp:class:`array`, which the caller should dynamically cast to the
  corresponding derived array, e.g., by examining
  :cpp:func:`array::scalar_type` and :cpp:func:`array::dimensionality`.

  The (static) factory function is made available by including
  :file:`zfp/factory.hpp`.  This header must be included *after* first
  including the header files associated with the compressed arrays, i.e.,
  :file:`zfp/array1.hpp`, :file:`zfp/array2.hpp`, :file:`zfp/array3.hpp`, and
  :file:`zfp/array4.hpp`.  Only those arrays whose header files are included
  can be constructed by the factory function.  This design decouples the
  array classes so that they may be included independently, for example,
  to reduce compilation time.

Both types of deserialization functions accept an :cpp:class:`array::header`,
an optional buffer holding compressed data, and an optional buffer size.
If this buffer is provided, then a separate copy of the compressed data it
holds is made, which is used to initialize the array.  If the optional buffer
size is also provided, then these functions throw an
:ref:`exception <exception>` if the size is not at least as large as is
expected from the metadata stored in the header.  This safeguard is
implemented to avoid accessing memory beyond the end of the buffer.  If no
buffer is provided, then all array elements are default initialized to zero.
The array may later be initialized by directly reading/copying data into the
space pointed to by :cpp:func:`array::compressed_data` and calling
:cpp:func:`array::clear_cache` (in either order).

Below is a simple example of serialization of a 3D compressed array of doubles
(error checking has been omitted for clarity)::

  zfp::array3d a(nx, ny, nz, rate);
  ...
  a.flush_cache();
  zfp::array::header h(a);
  fwrite(h.data(), h.size_bytes(), 1, file);
  fwrite(a.compressed_data(), a.compressed_size(), 1, file);

We may then deserialize this array using the factory function.  The following
example reads the compressed data directly into the array without making a
copy::

  zfp::array::header h;
  fread(h.data(), h.size_bytes(), 1, file);
  zfp::array* p = zfp::array::construct(h);
  fread(p->compressed_data(), p->compressed_size(), 1, file);
  assert(p->dimensionality() == 3 && p->scalar_type() == zfp_type_double);
  zfp::array3d& a = *dynamic_cast<zfp::array3d*>(p);

When the array is no longer in use, call :code:`delete p;` to deallocate it.

.. note::
  The array serialization API changed significantly in |zfp| |crpirelease|.
  The :cpp:func:`array::get_header` function is now deprecated and has been
  replaced with a :ref:`header constructor <header_ctor>` that takes an
  array as parameter.  Exceptions are now part of the main :code:`zfp`
  namespace rather than nested within the array header.  The header is no
  longer a simple POD data structure but should be queried for its data
  pointer and size.

.. index::
   single: Header

.. _header:

Header
^^^^^^

.. cpp:namespace:: zfp

Short 12-byte headers are used to describe array metadata and compression
parameters when serializing a compressed array.  This header is the same as
supported by the :c:func:`zfp_read_header` and :c:func:`zfp_write_header`
functions, using :c:macro:`ZFP_HEADER_FULL` to indicate that complete metadata
is to be stored in the header.  The header is also compatible with the |zfp|
:ref:`command-line tool <zfpcmd>`.  Processing of the header may result in an
:ref:`exception <exception>` being thrown.

.. note::
  Compressed-array headers use |zfp|'s most concise representation of only
  96 bits.  Such short headers support compressed blocks up to 2048 bits long.
  This implies that the highest rate for 3D arrays is 2048/4\ :sup:`3` = 32
  bits/value; the highest rate for 4D arrays is only 2048/4\ :sup:`4` = 8
  bits/value.  3D and 4D arrays whose rate exceeds these limits cannot be
  serialized and result in an exception being thrown.  1D and 2D arrays
  support rates up to 512 and 128 bits/value, respectively, which both
  are large enough to represent all usable rates.

.. cpp:class:: array::header

  The header stores information such as scalar type, array dimensions, and
  compression parameters such as rate.  Compressed-array headers are always
  96 bits long.

.. cpp:namespace:: zfp::array

----

.. cpp:function:: header::header()

  Default constructor for header.

----

.. _header_ctor:
.. cpp:function:: header::header(const array& a)

  Construct header for compressed-array *a*.  Throws an
  :ref:`exception <exception>` upon failure.

----

.. _header_ctor_buffer:
.. cpp:function:: header::header(const void* buffer, size_t bytes = 0)

  Deserialize header from memory buffer given by *buffer* of optional
  size *bytes*.  This memory buffer is obtained from an existing
  header during serialization via :cpp:func:`header::data` and
  :cpp:func:`header::size_bytes`.  The constructor throws an
  :ref:`exception <exception>` upon failure.

----

.. cpp:function:: zfp_type header::scalar_type() const

  Scalar type associated with array (see :cpp:func:`array::scalar_type`).

----

.. cpp:function:: uint header::dimensionality() const

  Dimensionality associated with array (see :cpp:func:`array::dimensionality`).

----

.. _header_dims:
.. cpp:function:: size_t header::size_x() const
.. cpp:function:: size_t header::size_y() const
.. cpp:function:: size_t header::size_z() const
.. cpp:function:: size_t header::size_w() const

  :ref:`Array dimensions <array_dims>`.  Unused dimensions have a size of zero.

----

.. cpp:function:: double header::rate() const

  Rate in bits per value (see :cpp:func:`array::rate`);

----

.. cpp:function:: virtual const void* header::data() const = 0

  Return pointer to header data.

----

.. cpp:function:: virtual size_t header::size_bytes(uint mask = ZFP_DATA_HEADER) const = 0

  When *mask* = :c:macro:`ZFP_DATA_HEADER`, return header payload size in
  bytes pointed to by :cpp:func:`header::data`.  Only those bytes are needed
  to (de)serialize a header.  The header object stores additional (redundant)
  metadata whose size can be queried via :c:macro:`ZFP_DATA_META`.

.. index::
   single: Exceptions

.. _exception:

Exceptions
^^^^^^^^^^

.. cpp:class:: exception : public std::runtime_error

  Compressed arrays may throw this exception upon serialization, when
  constructing a header via its :ref:`constructor <header_ctor>`, or
  deserialization, when constructing a compressed array via its
  :ref:`constructor <array_ctor_header>` or
  :ref:`factory function <array_factory>`.
  The :cpp:func:`exception::what` method returns a :code:`std::string`
  error message that indicates the cause of the exception.
  Most error messages changed in |zfp| |4darrrelease|.
