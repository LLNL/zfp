.. index::
   single: References
.. _references:

References
----------

.. cpp:namespace:: zfp

..
  .. cpp:class:: template<typename Scalar> array1::reference
  .. cpp:class:: template<typename Scalar> array2::reference
  .. cpp:class:: template<typename Scalar> array3::reference

.. cpp:class:: array1::reference
.. cpp:class:: array2::reference
.. cpp:class:: array3::reference

Array :ref:`indexing operators <lvref>` must return lvalue references that
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

The following operators are defined for |zfp| references.  They act on the
referenced array element in the same manner as operators defined for
conventional C++ references.

.. cpp:namespace:: zfp::arrayANY

.. cpp:function:: reference reference::operator=(const reference& ref)

  Assignment (copy) operator.  The referenced element, *elem*, is assigned the
  value stored at the element referenced by *ref*.  Return :code:`*this`.

.. cpp:function:: reference reference::operator=(Scalar val)
.. cpp:function:: reference reference::operator+=(Scalar val)
.. cpp:function:: reference reference::operator-=(Scalar val)
.. cpp:function:: reference reference::operator*=(Scalar val)
.. cpp:function:: reference reference::operator/=(Scalar val)

  Assignment and compound assignment operators.  For a given operator
  :code:`op`, update the referenced element, *elem*, via
  *elem* :code:`op` *val*.  Return :code:`*this`.

.. cpp:function:: pointer reference::operator&()

  Return pointer to the referenced array element.

Finally, |zfp| proxy references serve as a building block for implementing
proxy :ref:`pointers <pointers>` and :ref:`iterators <iterators>`.
