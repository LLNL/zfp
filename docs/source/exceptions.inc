.. index::
   single: Exceptions
.. _exceptions:

Exceptions
----------

.. cpp:namespace:: zfp::array::header

..
  .. cpp:class:: exception : public std::runtime_error

.. cpp:class:: exception : public std::runtime_error

  Compressed arrays can throw this exception when calling the constructor taking
  a :cpp:type:`zfp::array::header` and buffer. This exception will be thrown if
  the header is invalid or describes compressed data that is incompatible with
  this compressed-array. Also note this exception lies in the :cpp:any:`zfp`
  namespace.
