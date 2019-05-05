.. index::
   single: Iterators
.. _iterators:

Iterators
---------

.. cpp:namespace:: zfp

..
  .. cpp:class:: template<typename Scalar> array1::iterator
  .. cpp:class:: template<typename Scalar> array2::iterator
  .. cpp:class:: template<typename Scalar> array3::iterator

.. cpp:class:: array1::iterator
.. cpp:class:: array2::iterator
.. cpp:class:: array3::iterator

Iterators provide a mechanism for sequentially traversing a possibly
multi-dimensional array without having to track array indices or bounds.
They are also the preferred mechanism, compared to nested index loops, for
initializing arrays, because they are guaranteed to visit the array one block
at a time.  This allows all elements of a block to be initialized together
and ensures that the block is not compressed to memory before it has been
fully initialized, which might otherwise result in poor compression and,
consequently, larger errors than when the entire block is initialized as
a whole.  Note that the iterator traversal order differs in this respect
from traversal by :ref:`pointers <pointers>`.

The order of blocks visited is row-major (as in C), and the elements
within a block are also visited in row-major order, i.e., first by *x*,
then by *y*, and finally by *z*.  All |4powd| values in a block are visited
before moving on to the next block.

The iterators provided by |zfp| are sequential forward iterators, except
for 1D array iterators, which are random access iterators.  The reason
why higher dimensional iterators do not support random access is that
this would require very complicated index computations, especially for
arrays with partial blocks.  |zfp| iterators are
`STL <https://www.sgi.com/tech/stl/>`_ compliant and can
be used in STL algorithms that support forward and random access iterators.

All Iterators
^^^^^^^^^^^^^

Per STL mandate, the iterators define several types:

.. cpp:namespace:: zfp::arrayANY

.. cpp:type:: iterator::value_type

  The scalar type associated with the array that the iterator points into.

.. cpp:type:: iterator:: difference_type

  Difference between two iterators in number of array elements.

.. cpp:type:: iterator::reference
.. cpp:type:: iterator::pointer

  The reference and pointer type associated with the iterator's parent array
  class.

.. cpp:type:: iterator::iterator_category

  Type of iterator: :cpp:type:`std::random_access_iterator_tag` for 1D arrays;
  :cpp:type:`std::forward_iterator_tag` for all other arrays.

The following operations are defined on iterators:

.. cpp:function:: iterator iterator::operator=(const iterator& it)

  Assignment (copy) operator.  Make the iterator point to the same element
  as *it*.

.. cpp:function:: reference iterator::operator*() const

  Dereference operator.  Return reference to the value pointed to by the
  iterator.

.. cpp:function:: iterator& iterator::operator++()

  Pre increment.  Return a reference to the incremented iterator.

.. cpp:function:: iterator iterator::operator++(int)

  Post increment.  Return the value of the iterator before being incremented.

.. cpp:function:: bool iterator::operator==(const iterator& it) const
.. cpp:function:: bool iterator::operator!=(const iterator& it) const

  Return true (false) if the two iterators do (do not) point to the same
  element.

.. cpp:function:: uint iterator::i() const
.. cpp:function:: uint iterator::j() const
.. cpp:function:: uint iterator::k() const

  Return array index of element pointed to by the iterator.
  :cpp:func:`iterator::i` is defined for all arrays.
  :cpp:func:`iterator::j` is defined only for 2D and 3D arrays.
  :cpp:func:`iterator::k` is defined only for 3D arrays.

1D Array Iterators
^^^^^^^^^^^^^^^^^^

The following operators are defined **only for 1D arrays**:

.. cpp:namespace:: zfp

.. cpp:function:: array1::reference array1::iterator::operator[](difference_type d) const

  Random access index operator.

.. cpp:function:: array1::iterator& array1::iterator::operator--()

  Pre decrement.  Return a reference to the decremented iterator.

.. cpp:function:: array1::iterator array1::iterator::operator--(int)

  Post decrement.  Return the value of the iterator before being decremented.

.. cpp:function:: array1::iterator array1::iterator::operator+=(difference_type d)
.. cpp:function:: array1::iterator array1::iterator::operator-=(difference_type d)

  Increment (decrement) iterator *d* times.  Return value of incremented
  (decremented) iterator.

.. cpp:function:: array1::iterator array1::iterator::operator+(difference_type d) const
.. cpp:function:: array1::iterator array1::iterator::operator-(difference_type d) const

  Return a new iterator that has been incremented (decremented) by *d*.

.. cpp:function:: difference_type array1::iterator::operator-(const array1::iterator& it) const

  Return difference between this iterator and *it* in number of elements.
  The iterators must refer to elements in the same array.

.. cpp:function:: bool array1::iterator::operator<=(const array1::iterator& it) const
.. cpp:function:: bool array1::iterator::operator>=(const array1::iterator& it) const
.. cpp:function:: bool array1::iterator::operator<(const array1::iterator& it) const
.. cpp:function:: bool array1::iterator::operator>(const array1::iterator& it) const

  Return true if the two iterators satisfy the given relationship.
  For two iterators, *p* and *q*, within the same array, *p* < *q*
  if and only if *q* can be reached by incrementing *p* one or more times.
