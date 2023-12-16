.. index::
   single: Iterators
.. _iterators:

Iterators
---------

.. cpp:namespace:: zfp

.. cpp:class:: array1::const_iterator
.. cpp:class:: array2::const_iterator
.. cpp:class:: array3::const_iterator
.. cpp:class:: array4::const_iterator
.. cpp:class:: array1::iterator : public array1::const_iterator
.. cpp:class:: array2::iterator : public array2::const_iterator
.. cpp:class:: array3::iterator : public array3::const_iterator
.. cpp:class:: array4::iterator : public array4::const_iterator

Iterators provide a mechanism for traversing a possibly
multi-dimensional array---or a :ref:`view <views>` of a subset of an
array---without having to track array indices or bounds.
They are also the preferred mechanism, compared to nested index loops, for
initializing arrays, because they sequentially visit the array one block
at a time.  This allows all elements of a block to be initialized together
and ensures that the block is not compressed to memory before it has been
fully initialized, which might otherwise result in poor compression and,
consequently, larger compression errors than when the entire block is
initialized as a whole.  Note that the iterator traversal order differs in
this respect from traversal by :ref:`pointers <pointers>`.

Blocks are visited in raster order similarly to how individual array
elements are indexed, that is, first by *x*, then by *y*, then by *z*,
etc.  Within each block, elements are visited in the same raster
order.  All |4powd| values in a block are visited before moving on to the
next block (see :numref:`view-indexing`).

As of |zfp| |raiterrelease|, all iterators provided by |zfp| are random
access iterators (previously, multi-dimensional array iterators were only
forward iterators).  |zfp| iterators are
`STL <http://www.cplusplus.com/reference/stl/>`_ compliant and can
be used in STL algorithms that support random access iterators.

|zfp| |crpirelease| adds :code:`const` qualified versions of iterators,
given by the :code:`const_iterator` class.  Such iterators are available
also for :ref:`read-only arrays <carray_classes>`.

Per STL mandate, the iterators define several types:

.. cpp:namespace:: zfp::arrayANY

.. cpp:type:: iterator::value_type

  The scalar type associated with the array that the iterator points into.

----

.. cpp:type:: iterator::difference_type

  Difference between two iterators in number of array elements.

----

.. cpp:type:: iterator::reference

  The :ref:`reference <references>` type associated with the iterator's parent
  array class.

----

.. cpp:type:: iterator::pointer

  The :ref:`pointer <pointers>` type associated with the iterator's parent
  array class.

----

.. cpp:type:: iterator::iterator_category

  Type of iterator: :cpp:type:`std::random_access_iterator_tag`.

For const iterators, the following additional types are defined:

.. cpp:type:: const_iterator::const_reference

  The immutable reference type associated with the iterator's container class.

.. cpp:type:: const_iterator::const_pointer

  The immutable pointer type associated with the iterator's container class.

The following operations are defined on iterators:

.. cpp:function:: iterator iterator::operator=(const iterator& it)
.. cpp:function:: const_iterator const_iterator::operator=(const const_iterator& it)

  Assignment (copy) operator.  Make the iterator point to the same element
  as *it*.

----

.. cpp:function:: reference iterator::operator*() const
.. cpp:function:: const_reference const_iterator::operator*() const

  Dereference operator.  Return (const) reference to the value pointed to by
  the iterator.

----

.. cpp:function:: reference iterator::operator[](difference_type d) const
.. cpp:function:: const_reference const_iterator::operator[](difference_type d) const

  Offset dereference operator.  Return (const) reference to the value *d*
  elements relative to the current element in the iteration sequence (*d* may
  be negative).  This operator executes in constant time regardless of array
  dimensionality but is more costly than sequential iteration via
  :cpp:func:`iterator::operator++`.

----

.. cpp:function:: iterator iterator::operator+(difference_type d) const
.. cpp:function:: const_iterator const_iterator::operator+(difference_type d) const

  Return a new iterator that has been incremented by *d*.

----

.. cpp:function:: iterator iterator::operator-(difference_type d) const
.. cpp:function:: const_iterator const_iterator::operator-(difference_type d) const

  Return a new iterator that has been decremented by *d*.

----

.. cpp:function:: difference_type iterator::operator-(const iterator& it) const
.. cpp:function:: difference_type const_iterator::operator-(const const_iterator& it) const

  Return difference between this iterator and *it* in number of elements.
  The difference *p* |minus| *q* between two iterators, *p* and *q*, is
  negative if *p* < *q*.  The iterators must refer to elements in the same
  array.

----

.. cpp:function:: bool iterator::operator==(const iterator& it) const
.. cpp:function:: bool const_iterator::operator==(const const_iterator& it) const

  Return true if the two iterators point to the same element.

----

.. cpp:function:: bool iterator::operator!=(const iterator& it) const
.. cpp:function:: bool const_iterator::operator!=(const const_iterator& it) const

  Return true if the two iterators do not point to the same element.

----

.. _iter_inequalities:
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

----

.. cpp:function:: iterator& iterator::operator++()
.. cpp:function:: const_iterator& const_iterator::operator++()

  Prefix increment (:code:`++it`).  Return a reference to the
  incremented iterator.

----

.. cpp:function:: iterator iterator::operator++(int)
.. cpp:function:: const_iterator const_iterator::operator++(int)

  Postfix increment (:code:`it++`).  Return the value of the iterator
  before being incremented.

----

.. cpp:function:: iterator& iterator::operator--()
.. cpp:function:: const_iterator& const_iterator::operator--()

  Prefix decrement (:code:`--it`).  Return a reference to the
  decremented iterator.

----

.. cpp:function:: iterator iterator::operator--(int)
.. cpp:function:: const_iterator const_iterator::operator--(int)

  Postfix decrement (:code:`it--`).  Return the value of the
  iterator before being decremented.

----

.. cpp:function:: iterator iterator::operator+=(difference_type d)
.. cpp:function:: const_iterator const_iterator::operator+=(difference_type d)

  Increment iterator *d* times.  Return value of incremented iterator.
  Although :cpp:expr:`++it` and :cpp:expr:`it += 1` are semantically
  equivalent, the former is more efficient for multidimensional arrays.

----

.. cpp:function:: iterator iterator::operator-=(difference_type d)
.. cpp:function:: const_iterator const_iterator::operator-=(difference_type d)

  Decrement iterator *d* times.  Return value of decremented iterator.
  Although :cpp:expr:`--it` and :cpp:expr:`it -= 1` are semantically
  equivalent, the former is more efficient for multidimensional arrays.

----

.. cpp:function:: size_t iterator::i() const
.. cpp:function:: size_t iterator::j() const
.. cpp:function:: size_t iterator::k() const
.. cpp:function:: size_t iterator::l() const
.. cpp:function:: size_t const_iterator::i() const
.. cpp:function:: size_t const_iterator::j() const
.. cpp:function:: size_t const_iterator::k() const
.. cpp:function:: size_t const_iterator::l() const

  Return array index or local view index of element pointed to by the
  iterator.
  :cpp:func:`iterator::i` is defined for all arrays.
  :cpp:func:`iterator::j` is defined only for 2D, 3D, and 4D arrays.
  :cpp:func:`iterator::k` is defined only for 3D and 4D arrays.
  :cpp:func:`iterator::l` is defined only for 4D arrays.
