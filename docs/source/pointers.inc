.. index::
   single: Pointers
.. _pointers:

Pointers
--------

.. cpp:namespace:: zfp

.. cpp:class:: array1::const_pointer
.. cpp:class:: array2::const_pointer
.. cpp:class:: array3::const_pointer
.. cpp:class:: array4::const_pointer
.. cpp:class:: array1::pointer : public array1::const_pointer
.. cpp:class:: array2::pointer : public array2::const_pointer
.. cpp:class:: array3::pointer : public array3::const_pointer
.. cpp:class:: array4::pointer : public array4::const_pointer

Similar to :ref:`references <references>`, |zfp| supports proxy pointers
(also known as fancy pointers) to individual array elements.  From the
user's perspective, such pointers behave much like regular pointers to
uncompressed data, e.g., instead of
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
traverse arrays element by element as though the arrays were flattened to
one-dimensional arrays.  That is, if :code:`p` points to the first element
of a 3D array :code:`a(nx, ny, nz)`, then
:code:`a(i, j, k) == p[i + nx * (j + ny * k)]`.  In other words, pointer
indexing follows the same order as flat array indexing
(see :cpp:func:`array::operator[]`).

A pointer remains valid during the lifetime of the array into which it points.
Like conventional pointers, proxy pointers can be passed to other functions
and manipulated there, for instance, by passing the pointer by reference via
:code:`pointer&`.

As of |zfp| |crpirelease|, const qualified pointers :code:`const_pointer`
are available, and conceptually are equivalent to :code:`const Scalar*`.
Pointers are available for :ref:`read-only arrays <carray_classes>` also.

The following operators are defined for proxy pointers.  Below *p* refers
to the pointer being acted upon.

.. cpp:namespace:: zfp::arrayANY

.. cpp:function:: pointer pointer::operator=(const pointer& q)
.. cpp:function:: const_pointer const_pointer::operator=(const const_pointer& q)

  Assignment operator.  Assigns *q* to *p*.

----

.. cpp:function:: reference pointer::operator*() const
.. cpp:function:: const_reference const_pointer::operator*() const

  Dereference operator.  Return proxy (const) reference to the value pointed
  to by *p*.

----

.. cpp:function:: reference pointer::operator[](ptrdiff_t d) const
.. cpp:function:: const_reference const_pointer::operator[](ptrdiff_t d) const

  Offset dereference operator.  Return proxy (const) reference to the value
  stored at :code:`p[d]`.

----

.. cpp:function:: pointer pointer::operator+(ptrdiff_t d) const
.. cpp:function:: const_pointer const_pointer::operator+(ptrdiff_t d) const

  Return a copy of the pointer incremented by *d*.

----

.. cpp:function:: pointer pointer::operator-(ptrdiff_t d) const
.. cpp:function:: const_pointer const_pointer::operator-(ptrdiff_t d) const

  Return a copy of the pointer decremented by *d*.

----

.. cpp:function:: ptrdiff_t pointer::operator-(const pointer& q) const
.. cpp:function:: ptrdiff_t const_pointer::operator-(const const_pointer& q) const

  Return difference *p - q*.  Defined only for pointers within the same
  array.

----

.. cpp:function:: bool pointer::operator==(const pointer& q) const
.. cpp:function:: bool const_pointer::operator==(const const_pointer& q) const

  Return true if *p* and *q* point to the same array element.

----

.. cpp:function:: bool pointer::operator!=(const pointer& q) const
.. cpp:function:: bool const_pointer::operator!=(const const_pointer& q) const

  Return true if *p* and *q* do not point to the same array element.
  This operator returns false if *p* and *q* do not point into the same array.

----

.. _ptr_inequalities:
.. cpp:function:: bool pointer::operator<=(const pointer& q) const
.. cpp:function:: bool pointer::operator>=(const pointer& q) const
.. cpp:function:: bool pointer::operator<(const pointer& q) const
.. cpp:function:: bool pointer::operator>(const pointer& q) const
.. cpp:function:: bool const_pointer::operator<=(const const_pointer& q) const
.. cpp:function:: bool const_pointer::operator>=(const const_pointer& q) const
.. cpp:function:: bool const_pointer::operator<(const const_pointer& q) const
.. cpp:function:: bool const_pointer::operator>(const const_pointer& q) const

  Return true if the two pointers satisfy the given relationship.  These operators
  return false if *p* and *q* do not point into the same array.

----

.. cpp:function:: pointer& pointer::operator++()
.. cpp:function:: const_pointer& const_pointer::operator++()

  Prefix increment pointer, i.e., :code:`++p`.  Return reference to
  the incremented pointer.

----

.. cpp:function:: pointer& pointer::operator--()
.. cpp:function:: const_pointer& const_pointer::operator--()

  Prefix decrement pointer, i.e., :code:`--p`.  Return reference to
  the decremented pointer.

----

.. cpp:function:: pointer pointer::operator++(int)
.. cpp:function:: const_pointer const_pointer::operator++(int)

  Postfix increment pointer, i.e., :code:`p++`.  Return a copy of
  the pointer before it was incremented.

----

.. cpp:function:: pointer pointer::operator--(int)
.. cpp:function:: const_pointer const_pointer::operator--(int)

  Postfix decrement pointer, i.e., :code:`p--`.  Return a copy of
  the pointer before it was decremented.

----

.. cpp:function:: pointer pointer::operator+=(ptrdiff_t d)
.. cpp:function:: const_pointer const_pointer::operator+=(ptrdiff_t d)

  Increment pointer by *d*.  Return a copy of the incremented pointer.

----

.. cpp:function:: pointer pointer::operator-=(ptrdiff_t d)
.. cpp:function:: const_pointer const_pointer::operator-=(ptrdiff_t d)

  Decrement pointer by *d*.  Return a copy of the decremented pointer.
