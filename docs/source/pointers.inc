.. index::
   single: Pointers
.. _pointers:

Pointers
--------

.. cpp:namespace:: zfp

..
  .. cpp:class:: template<typename Scalar> array1::pointer
  .. cpp:class:: template<typename Scalar> array2::pointer
  .. cpp:class:: template<typename Scalar> array3::pointer

.. cpp:class:: array1::pointer
.. cpp:class:: array2::pointer
.. cpp:class:: array3::pointer

Similar to :ref:`references <references>`, |zfp| supports proxies for
pointers to individual array elements.  From the user's perspective, such
pointers behave much like regular pointers to uncompressed data, e.g.,
instead of
::

    float a[ny][nx];     // uncompressed 2D array of floats
    float* p = &a[0][0]; // point to first array element
    p[nx] = 1;           // set a[1][0] = 1
    *++p = 2;            // set a[0][1] = 2

one would write
::

    zfp::array2<float> a(nx, ny, rate);       // compressed 2D array of floats
    zfp::array2<float>::pointer p = &a(0, 0); // point to first array element
    p[nx] = 1;                                // set a(0, 1) = 1
    *++p = 2;                                 // set a(1, 0) = 2

However, even though |zfp|'s proxy pointers point to individual scalars,
they are associated with the array that those scalars are stored in, including
the array's dimensionality.  Pointers into arrays of different dimensionality
have incompatible type.  Moreover, pointers to elements in different arrays
are incompatible.  For example, one cannot take the difference between
pointers into two different arrays.

Unlike |zfp|'s proxy references, its proxy pointers support traversing
arrays using conventional pointer arithmetic.  In particular, unlike the
:ref:`iterators <iterators>` below, |zfp|'s pointers are oblivious to the
fact that the compressed arrays are partitioned into blocks, and the pointers
traverse arrays element by element as though the arrays were flattened in
standard C row-major order.  That is, if :code:`p` points to the first
element of a 3D array :code:`a(nx, ny, nz)`, then
:code:`a(i, j, k) == p[i + nx * (j + ny * k)]`.  In other words, pointer
indexing follows the same order as flat array indexing
(see :cpp:func:`array::operator[]`).

A pointer remains valid during the lifetime of the scalar that it points to.
Like conventional pointers, proxy pointers can be passed to other functions
and manipulated there, for instance by passing the pointer by reference via
:code:`pointer&`.

The following operators are defined for proxy pointers.  Below *p* refers
to the pointer being acted upon.

.. cpp:namespace:: zfp::arrayANY

.. cpp:function:: pointer pointer::operator=(const pointer& q)

  Assignment operator.  Assigns *q* to *p*.

.. cpp:function:: reference pointer::operator*() const

  Dereference operator.  Return proxy reference to the value pointed to by
  *p*.

.. cpp:function:: reference pointer::operator[](ptrdiff_t d) const

  Index operator.  Return reference to the value stored at :code:`p[d]`.

.. cpp:function:: pointer& pointer::operator++()
.. cpp:function:: pointer& pointer::operator--()

  Pre increment (decrement) pointer, e.g., :code:`++p`.  Return reference to
  the incremented (decremented) pointer.

.. cpp:function:: pointer pointer::operator++(int)
.. cpp:function:: pointer pointer::operator--(int)

  Post increment (decrement) pointer, e.g., :code:`p++`.  Return a copy of
  the pointer before it was incremented (decremented).

.. cpp:function:: pointer pointer::operator+=(ptrdiff_t d)
.. cpp:function:: pointer pointer::operator-=(ptrdiff_t d)

  Increment (decrement) pointer by *d*.  Return a copy of the incremented
  (decremented) pointer.

.. cpp:function:: pointer pointer::operator+(ptrdiff_t d) const
.. cpp:function:: pointer pointer::operator-(ptrdiff_t d) const

  Return a copy of the pointer incremented (decremented) by *d*.

.. cpp:function:: ptrdiff_t pointer::operator-(const pointer& q) const

  Return difference *p - q*.  Defined only for pointers within the same
  array.

.. cpp:function:: bool pointer::operator==(const pointer& q) const
.. cpp:function:: bool pointer::operator!=(const pointer& q) const

  Pointer comparison.  Return true (false) if *p* and *q* do (do not) point
  to the same array element.
