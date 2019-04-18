.. index::
   single: Header
.. _header:

Header
------

.. cpp:namespace:: zfp::array

..
  .. cpp:class:: header

.. cpp:class:: header

  This struct stores information needed to describe a compressed-array, should
  a user want to serialize it. Compressed-arrays always have 96 bit headers,
  stored in the buffer array. This header is fully compatible with the C API.
  ::

    class header {
      uchar buffer[BITS_TO_BYTES(ZFP_HEADER_SIZE_BYTES)];
    };
