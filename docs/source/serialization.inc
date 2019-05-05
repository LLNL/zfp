.. index::
   single: Serialization
.. _serialization:

Serialization
-------------

.. cpp:namespace:: zfp

|zfp|'s compressed arrays can be serialized to sequential, contiguous
storage and later recovered back into an object, for example, to support
I/O of compressed-array objects.  Two pieces of information are needed
to describe a |zfp| array: the raw compressed data, obtained via
:cpp:func:`array::compressed_data` and :cpp:func:`array::compressed_size`,
and a :ref:`header <header>` that describes the array scalar type,
dimensions, and rate, obtained via :cpp:func:`array::get_header`.
The user may concatenate the header and compressed data to form a
fixed-rate byte stream that can be read by the |zfp|
:ref:`command-line tool <zfpcmd>`.  When serializing the array,
the user should first call :cpp:func:`array::flush_cache()` before
accessing the raw compressed data.

There are two primary ways to construct a compressed-array object from
compressed data: via array-specific :ref:`constructors <array_ctor_header>`
and via a generic :ref:`factory function <array_factory>`:

- When the array scalar type (i.e., :code:`float` or :code:`double`) and
  dimensionality (i.e., 1D, 2D, or 3D) are already known, the corresponding
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
  :file:`zfpfactory.h`.  This header must be included *after* first
  including the header files associated with the compressed arrays, i.e.,
  :file:`zfparray1.h`, :file:`zfparray2.h`, and :file:`zfparray3.h`.
  Only those arrays whose header files are included can be constructed by
  the factory function.  This design decouples the array classes so that
  they may be included independently, for example, to reduce compilation
  time.

Both types of deserialization functions accept an :cpp:class:`array::header`,
an optional buffer holding compressed data, and an optional buffer size.
If this buffer is provided, then a separate copy of the compressed data it
holds is made, which is used to initialize the array.  If the optional buffer
size is also provided, then these functions throw an exception if the size
is not at least as large as is expected from the metadata stored in the
header.  This safeguard is implemented to avoid accessing memory beyond the
end of the buffer.  If no buffer is provided, then all array elements are
default initialized to zero.  The array may later be initialized by directly
reading/copying data into the space pointed to by
:cpp:func:`array::compressed_data`.

Below is a simple example of serialization of a 3D compressed array of doubles
(error checking has been omitted for clarity)::

  zfp::array3d a(nx, ny, nz, rate);
  ...
  a.flush_cache();
  zfp::array::header h = a.get_header();
  fwrite(&h, sizeof(h), 1, file);
  fwrite(a.compressed_data(), a.compressed_size(), 1, file);

We may then deserialize this array using the factory function.  The following
example reads the compressed data directly into the array without making a
copy::

  zfp::array::header h;
  fread(&h, sizeof(h), 1, file);
  zfp::array* p = zfp::array::construct(h);
  fread(p->compressed_data(), p->compressed_size(), 1, file);
  assert(p->dimensionality() == 3 && p->scalar_type() == zfp_type_double);
  zfp::array3d& a = *dynamic_cast<zfp::array3d*>(p);

.. index::
   single: Header
   single: Exceptions

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

.. cpp:class:: array::header

  The header stores information such as scalar type, array dimensions, and
  rate.  Compressed-array headers are always 96 bits long.  These bits are
  stored in the :code:`header::buffer` field, the only field represented
  in the header.
  ::

    struct header {
      uchar buffer[BITS_TO_BYTES(ZFP_HEADER_SIZE_BITS)];
    };

.. _exception:
.. cpp:class:: array::header::exception : public std::runtime_error

  Compressed arrays can throw this exception upon serialization, when forming
  a header via :cpp:func:`get_header`, or deserialization, when constructing
  a compressed array via its :ref:`constructor <array_ctor_header>` or
  :ref:`factory function <array_factory>`.

.. note::
  Compressed-array headers use |zfp|'s most concise representation of only
  96 bits.  Such short headers support compressed blocks up to 2048 bits long.
  This implies that the highest rate for 3D arrays is 2048/4\ :sup:`3` = 32
  bits/value.  3D arrays whose rate exceeds 32 cannot be serialized using
  :cpp:func:`array::get_header`, which for such arrays throws an exception.
  1D and 2D arrays do not suffer from this limitation.
