.. index::
   single: array-factory
.. _array_factory:

Serializing and Deserializing Compressed Arrays
-----------------------------------------------

|zfp|'s compressed arrays can be written to disk and recovered back into an
object. A header is generated through :cpp:func:`get_header`. Also record the
compressed-data (blocks) using :cpp:func:`compressed_data`, having length
:cpp:func:`compressed_size` (in bytes).

There are two ways to construct a compressed array object from memory. The
first is to use the correct array type's constructor, which accepts a
:cpp:type:`zfp::array::header`, optional *buffer*, and optional *buffer_size*.
If the wrong array type is used, then :cpp:class:`zfp::array::header::exception`
is thrown.

The second method is useful when the serialized array type is unknown, but
described in the header. Include zfpfactory.h after including the 1, 2, and/or 3
dimensional compressed array headers, and use (static)
:cpp:func:`zfp::array::construct`, listed below. This function also accepts a
*header*, optional *buffer*, and optional *buffer_size*.

.. cpp:function:: static array* zfp::array::construct(const zfp::header& h, const uchar* buffer = 0, size_t buffer_size_bytes = 0)

  Construct the correct compressed array object from what the header describes,
  returning a pointer to the base class upon success, otherwise throwing a
  :cpp:class:`zfp::array::header::exception`. Just as with the constructor,
  :cpp:type:`zfp::array::header` holds the header and optional *buffer* points to the
  compressed data. Optional *buffer_size_bytes* argument specifies *buffer*
  length, in bytes.
