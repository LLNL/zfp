.. index::
   single: Iterators
.. _iterators:

Iterators
---------

.. cpp:namespace:: zfp

.. cpp:class:: array1::const_iterator
.. cpp:class:: array2::const_iterator
.. cpp:class:: array3::const_iterator
.. cpp:class:: array1::iterator : public array1::const_iterator
.. cpp:class:: array2::iterator : public array2::const_iterator
.. cpp:class:: array3::iterator : public array3::const_iterator

Iterators provide a mechanism for sequentially traversing a possibly
multi-dimensional array without having to track array indices or bounds.
They are also the preferred mechanism, compared to nested index loops, for
initializing arrays, because they are guaranteed to visit the array one block
at a time.  This allows all elements of a block to be initialized together
and ensures that the block is not compressed to memory before it has been
fully initialized, which might otherwise result in poor compression and,
consequently, larger compression errors than when the entire block is
initialized as a whole.  Note that the iterator traversal order differs in
this respect from traversal by :ref:`pointers <pointers>`.

Blocks are visited is in raster order similarly to how indidivual array
elements are indexed, that is, first by *x*, then by *y*, and finally
by *z*.  Within each block, elements are visited in the same raster order.
All |4powd| values in a block are visited before moving on to the next block.

The iterators provided by |zfp| are sequential forward iterators, except
for 1D array iterators, which are random access iterators.  The reason
why higher dimensional iterators do not support random access is that
this would require very complicated index computations, especially for
arrays with partial blocks.  |zfp| iterators are
`STL <http://www.cplusplus.com/reference/stl/>`_ compliant and can
be used in STL algorithms that support forward and random access iterators.

|zfp| |crpirelease| adds :code:`const` qualified versions of iterators,
given by the :code:`const_iterator` class.

All Iterators
^^^^^^^^^^^^^

Per STL mandate, the iterators define several types:

.. cpp:namespace:: zfp::arrayANY

.. cpp:type:: iterator::value_type

  The scalar type associated with the array that the iterator points into.

.. cpp:type:: iterator::difference_type

  Difference between two iterators in number of array elements.

.. cpp:type:: iterator::reference
.. cpp:type:: iterator::pointer

  The reference and pointer type associated with the iterator's parent array
  class.

.. cpp:type:: iterator::iterator_category

  Type of iterator: :cpp:type:`std::random_access_iterator_tag` for 1D arrays;
  :cpp:type:`std::forward_iterator_tag` for all other arrays.

The following operations are defined on const iterators:

.. cpp:function:: iterator iterator::operator=(const iterator& it)
.. cpp:function:: const_iterator const_iterator::operator=(const const_iterator& it)

  Assignment (copy) operator.  Make the iterator point to the same element
  as *it*.

.. cpp:function:: reference iterator::operator*() const
.. cpp:function:: const_reference const_iterator::operator*() const

  Dereference operator.  Return (const) reference to the value pointed to by
  the iterator.

.. cpp:function:: iterator& iterator::operator++()
.. cpp:function:: const_iterator& const_iterator::operator++()

  Pre increment.  Return a (const) reference to the incremented iterator.

.. cpp:function:: iterator iterator::operator++(int)
.. cpp:function:: const_iterator const_iterator::operator++(int)

  Post increment.  Return the value of the iterator before being incremented.

.. cpp:function:: bool iterator::operator==(const iterator& it) const
.. cpp:function:: bool const_iterator::operator==(const const_iterator& it) const

  Return true if the two iterators point to the same element.

.. cpp:function:: bool iterator::operator!=(const iterator& it) const
.. cpp:function:: bool const_iterator::operator!=(const const_iterator& it) const

  Return true if the two iterators do not point to the same element.

.. cpp:function:: size_t iterator::i() const
.. cpp:function:: size_t iterator::j() const
.. cpp:function:: size_t iterator::k() const
.. cpp:function:: size_t const_iterator::i() const
.. cpp:function:: size_t const_iterator::j() const
.. cpp:function:: size_t const_iterator::k() const

  Return array index of element pointed to by the iterator.
  :cpp:func:`iterator::i` is defined for all arrays.
  :cpp:func:`iterator::j` is defined only for 2D and 3D arrays.
  :cpp:func:`iterator::k` is defined only for 3D arrays.

1D Array Iterators
^^^^^^^^^^^^^^^^^^

The following operators are defined **only for 1D array iterators**:

.. cpp:namespace:: zfp::array1

.. cpp:function:: reference iterator::operator[](difference_type d) const
.. cpp:function:: const_reference const_iterator::operator[](difference_type d) const

  Random access index operator.

.. cpp:function:: iterator& iterator::operator--()
.. cpp:function:: const_iterator& const_iterator::operator--()

  Prefix decrement.  Return a reference to the decremented iterator.

.. cpp:function:: iterator iterator::operator--(int)
.. cpp:function:: const_iterator const_iterator::operator--(int)

  Postfix decrement.  Return the value of the iterator before being decremented.

.. cpp:function:: iterator iterator::operator+=(difference_type d)
.. cpp:function:: const_iterator const_iterator::operator+=(difference_type d)

  Increment iterator *d* times.  Return value of incremented iterator.

.. cpp:function:: iterator iterator::operator-=(difference_type d)
.. cpp:function:: const_iterator const_iterator::operator-=(difference_type d)

  Decrement iterator *d* times.  Return value of decremented iterator.

.. cpp:function:: iterator iterator::operator+(difference_type d) const
.. cpp:function:: const_iterator const_iterator::operator+(difference_type d) const

  Return a new iterator that has been incremented by *d*.

.. cpp:function:: iterator iterator::operator-(difference_type d) const
.. cpp:function:: const_iterator const_iterator::operator-(difference_type d) const

  Return a new iterator that has been decremented by *d*.

.. cpp:function:: difference_type iterator::operator-(const iterator& it) const
.. cpp:function:: difference_type const_iterator::operator-(const const_iterator& it) const

  Return difference between this iterator and *it* in number of elements.
  The iterators must refer to elements in the same array.

.. cpp:function:: bool iterator::operator<=(const iterator& it) const
.. cpp:function:: bool iterator::operator>=(const iterator& it) const
.. cpp:function:: bool iterator::operator<(const iterator& it) const
.. cpp:function:: bool iterator::operator>(const iterator& it) const
.. cpp:function:: bool const_iterator::operator<=(const const_iterator& it) const
.. cpp:function:: bool const_iterator::operator>=(const const_iterator& it) const
.. cpp:function:: bool const_iterator::operator<(const const_iterator& it) const
.. cpp:function:: bool const_iterator::operator>(const const_iterator& it) const

  Return true if the two iterators satisfy the given relationship.
  For two iterators, *p* and *q*, within the same array, *p* < *q*
  if and only if *q* can be reached by incrementing *p* one or more times.
