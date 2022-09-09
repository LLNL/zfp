.. index::
   single: References
.. _references:

References
----------

.. cpp:namespace:: zfp

.. cpp:class:: array1::const_reference
.. cpp:class:: array2::const_reference
.. cpp:class:: array3::const_reference
.. cpp:class:: array4::const_reference
.. cpp:class:: array1::reference : public array1::const_reference
.. cpp:class:: array2::reference : public array2::const_reference
.. cpp:class:: array3::reference : public array3::const_reference
.. cpp:class:: array4::reference : public array4::const_reference

Array :ref:`indexing operators <lvref_idx>` must return lvalue references that
alias array elements and serve as vehicles for assigning values to those
elements.  Unfortunately, |zfp| cannot simply return a standard C++ reference
(e.g., :code:`float&`) to an uncompressed array element since the element in
question may exist only in compressed form or as a transient cached entry that
may be invalidated (evicted) at any point.

To address this, |zfp| provides *proxies* for references and pointers that
act much like regular references and pointers, but which refer to elements
by array and index rather than by memory address.  When assigning to an
array element through such a proxy reference or pointer, the corresponding
element is decompressed to cache (if not already cached) and immediately
updated.

|zfp| references may be freely passed to other functions and they remain
valid during the lifetime of the corresponding array element.  One may also
take the address of a reference, which yields a
:ref:`proxy pointer <pointers>`.  When a reference appears as an rvalue in
an expression, it is implicitly converted to a value.

|zfp| |crpirelease| adds ``const`` qualified versions of references,
pointers, and iterators to support const correctness and potential performance
improvements when only read access is needed.  As with STL containers, the
corresponding types are prefixed by ``const_``, e.g.,
``const_reference``.  The mutable versions of these classes inherit
the read-only API from the corresponding const versions.

Only references into :ref:`read-write arrays <array_classes>` are discussed
here; the :ref:`read-only arrays <carray_classes>` support the same
``const_reference`` API.

.. note::
  Do not confuse :code:`const_reference` and :code:`const reference`.  The
  former is a reference to an immutable array element, while the latter means
  that the proxy reference object itself is immutable.

References define a single type:

.. cpp:namespace:: zfp::arrayANY

.. cpp:type:: reference::value_type
.. cpp:type:: const_reference::value_type

  Scalar type associated with referenced array elements.

----

The following operators are defined for |zfp| references.  They act on the
referenced array element in the same manner as operators defined for
conventional C++ references.  References are obtained via
:ref:`array inspectors <array_accessor>`
and :ref:`mutators <lvref>`.

----

.. cpp:function:: value_type reference::operator value_type() const
.. cpp:function:: value_type const_reference::operator value_type() const

  Conversion operator for dereferencing the reference.  Return the value
  of the referenced array element.

----

.. cpp:function:: pointer reference::operator&() const
.. cpp:function:: const_pointer const_reference::operator&() const

  Return (const) pointer to the referenced array element.
 
----

.. _ref_copy:
.. cpp:function:: reference reference::operator=(const reference& ref)

  Assignment (copy) operator.  The referenced element, *elem*, is assigned the
  value stored at the element referenced by *ref*.  Return :code:`*this`.

----

.. _ref_mutators:
.. cpp:function:: reference reference::operator=(Scalar val)
.. cpp:function:: reference reference::operator+=(Scalar val)
.. cpp:function:: reference reference::operator-=(Scalar val)
.. cpp:function:: reference reference::operator*=(Scalar val)
.. cpp:function:: reference reference::operator/=(Scalar val)

  Assignment and compound assignment operators.  For a given operator
  :code:`op`, update the referenced element, *elem*, via
  *elem* :code:`op` *val*.  Return :code:`*this`.
