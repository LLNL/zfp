.. index::
   single: Views
.. _views:

Views
-----

.. cpp:namespace:: zfp

|zfp| |viewsrelease| adds array views.
Much like how :ref:`references <references>` allow indirect access to
single array elements, *views* provide indirect access to whole arrays,
or more generally to rectangular subsets of arrays.  A view of an array
does not allocate any storage for the array elements.  Rather, the
view accesses shared storage managed by the underlying array.  This
allows for multiple entries into an array without the need for expensive
deep copies.  In a sense, views can be thought of as *shallow copies*
of arrays.

When a view exposes a whole array :code:`array<type>`, it provides
similar functionality to a C++ reference :code:`array<type>&` or
pointer :code:`array<type>*` to the array.  However, views are more
general in that they also allow restricting access to a user-specified
subset of the array, and unlike pointers also provide for the same
syntax when accessing the array, e.g., :code:`array_view(i, j)` instead
of :code:`(*array_ptr)(i, j)`.

|zfp|'s *nested views* further provide for multidimensional
array access analogous to the C/C++ nested array syntax :code:`array[i][j]`.
Finally, |zfp|'s *private views* can be used to ensure thread-safe access
to its compressed arrays.

Access to array elements through a view is via inspectors and mutators
that return a :code:`const_reference` or :code:`reference`, respectively
(see :ref:`references`).  As of |zfp| |crpirelease|, it is also possible
to obtain pointers to array elements through views and to iterate over them.
View pointers and iterators allow referencing only the elements visible
through the view, e.g., a rectangular subset of an array
(see :numref:`view-indexing`).  Those elements are indexed as if the
view were a contiguous array, and pointer arithmetic assumes that the
possibly smaller view and not the underlying array is flattened.
:ref:`Private views <private_immutable_view>` maintain their own cache
and therefore implement their own proxy references, pointers, and
iterators.

.. _view-indexing:
.. figure:: view-indexing.pdf
  :figwidth: 90 %
  :align: center
  :alt: "2D view indexing"

  An 11 |times| 9 element view of a 2D array of dimensions 16 |times| 12.
  The numbered elements indicate the order in which the view is
  traversed using pointers and iterators.  We have
  :code:`view(10, 7) == (&view(0, 0))[87] == view.begin()[97] == view.end()[-2]`.

With the |zfp| |carrrelease| release of
:ref:`read-only arrays <carray_classes>`, such arrays also support the two
kinds of immutable views (:code:`const_view` and :code:`private_const_view`).
The documentation below applies to views into read-only arrays as well.

.. note::
  Like iterators and proxy references and pointers, a view is valid only
  during the lifetime of the array that it references.  **No reference
  counting** is done to keep the array alive.  It is up to the user to
  ensure that the referenced array object is valid when accessed through
  a view.

There are several types of views distinguished by these attributes:

* Read-only vs. read-write access.
* Shared vs. private access.
* Flat vs. nested indexing.

Each of these attributes is discussed in detail below in these
sections:

* :ref:`immutable_view`
* :ref:`mutable_view`
* :ref:`flat_view`
* :ref:`nested_view`
* :ref:`slicing`
* :ref:`private_immutable_view`
* :ref:`private_mutable_view`


.. _immutable_view:

Immutable view
^^^^^^^^^^^^^^

The most basic view is the immutable :code:`const_view`, which
supports read-only access to the array elements it references.
This view serves primarily as a base class for more specialized
views.  Its constructors allow establishing access to a whole
array or to a rectangular subset of an array.  Note that like
references, pointers, and iterators, views are types nested within
the arrays that they reference.

..
  .. cpp:class:: template<typename Scalar> array1::const_view
  .. cpp:class:: template<typename Scalar> array2::const_view
  .. cpp:class:: template<typename Scalar> array3::const_view
  .. cpp:class:: template<typename Scalar> array4::const_view

.. cpp:class:: array1::const_view
.. cpp:class:: array2::const_view
.. cpp:class:: array3::const_view
.. cpp:class:: array4::const_view

  Immutable view into 1D, 2D, 3D, and 4D array.

----

.. _view_ctor:
.. cpp:function:: array1::const_view::const_view(array1* array)
.. cpp:function:: array2::const_view::const_view(array2* array)
.. cpp:function:: array3::const_view::const_view(array3* array)
.. cpp:function:: array4::const_view::const_view(array4* array)

  Constructor for read-only access to a whole array.  As already
  mentioned, these views are valid only during the lifetime of the
  underlying array object.

----

.. cpp:function:: array1::const_view::const_view(array1* array, size_t x, size_t nx)
.. cpp:function:: array2::const_view::const_view(array2* array, size_t x, size_t y, size_t nx, size_t ny)
.. cpp:function:: array3::const_view::const_view(array3* array, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz)
.. cpp:function:: array4::const_view::const_view(array4* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw)

  Constructors for read-only access to a rectangular subset of an
  array.  The subset is specified by an offset, e.g.,
  (*x*, *y*, *z*) for a 3D array, and dimensions, e.g.,
  (*nx*, *ny*, *nz*) for a 3D array.  The rectangle must fit within
  the surrounding array.

----

.. cpp:function:: size_t array1::const_view::global_x(size_t i) const
.. cpp:function:: size_t array2::const_view::global_x(size_t i) const
.. cpp:function:: size_t array2::const_view::global_y(size_t j) const
.. cpp:function:: size_t array3::const_view::global_x(size_t i) const
.. cpp:function:: size_t array3::const_view::global_y(size_t j) const
.. cpp:function:: size_t array3::const_view::global_z(size_t k) const
.. cpp:function:: size_t array4::const_view::global_x(size_t i) const
.. cpp:function:: size_t array4::const_view::global_y(size_t j) const
.. cpp:function:: size_t array4::const_view::global_z(size_t k) const
.. cpp:function:: size_t array4::const_view::global_w(size_t l) const

  Return global array index associated with local view index.  For
  instance, if a 1D view has been constructed with offset *x*, then
  :code:`global_x(i)` returns *x* + *i*.

----

.. cpp:function:: size_t array1::const_view::size_x() const
.. cpp:function:: size_t array2::const_view::size_x() const
.. cpp:function:: size_t array2::const_view::size_y() const
.. cpp:function:: size_t array3::const_view::size_x() const
.. cpp:function:: size_t array3::const_view::size_y() const
.. cpp:function:: size_t array3::const_view::size_z() const
.. cpp:function:: size_t array4::const_view::size_x() const
.. cpp:function:: size_t array4::const_view::size_y() const
.. cpp:function:: size_t array4::const_view::size_z() const
.. cpp:function:: size_t array4::const_view::size_w() const

  Return dimensions of view.

----

.. _view_accessor:
.. cpp:function:: const_reference array1::const_view::operator()(size_t i) const
.. cpp:function:: const_reference array2::const_view::operator()(size_t i, size_t j) const
.. cpp:function:: const_reference array3::const_view::operator()(size_t i, size_t j, size_t k) const
.. cpp:function:: const_reference array4::const_view::operator()(size_t i, size_t j, size_t k, size_t l) const

  Return reference to scalar stored at multi-dimensional index given by
  *x* + *i*, *y* + *j*, *z* + *k*, and *w* + *l*, where *x*, *y*, *z*, and *w*
  specify the offset into the array.

----

.. cpp:function:: const_reference array1::const_view::operator[](size_t index) const

  Alternative inspector for 1D arrays identical to
  :cpp:func:`array1::const_view::operator()`.

----

.. cpp:function:: array1::const_view::const_iterator array1::const_view::begin() const
.. cpp:function:: array2::const_view::const_iterator array2::const_view::begin() const
.. cpp:function:: array3::const_view::const_iterator array3::const_view::begin() const
.. cpp:function:: array4::const_view::const_iterator array4::const_view::begin() const
.. cpp:function:: array1::const_view::const_iterator array1::const_view::cbegin() const
.. cpp:function:: array2::const_view::const_iterator array2::const_view::cbegin() const
.. cpp:function:: array3::const_view::const_iterator array3::const_view::cbegin() const
.. cpp:function:: array4::const_view::const_iterator array4::const_view::cbegin() const

  Random-access const iterator to first element of view.

----

.. cpp:function:: array1::const_view::const_iterator array1::const_view::end() const
.. cpp:function:: array2::const_view::const_iterator array2::const_view::end() const
.. cpp:function:: array3::const_view::const_iterator array3::const_view::end() const
.. cpp:function:: array4::const_view::const_iterator array4::const_view::end() const
.. cpp:function:: array1::const_view::const_iterator array1::const_view::cend() const
.. cpp:function:: array2::const_view::const_iterator array2::const_view::cend() const
.. cpp:function:: array3::const_view::const_iterator array3::const_view::cend() const
.. cpp:function:: array4::const_view::const_iterator array4::const_view::cend() const

  Random-access const iterator to end of view.

There are a number of common methods inherited from a base class,
:code:`preview`, further up the class hierarchy.

.. cpp:function:: double arrayANY::const_view::rate() const

  Return rate in bits per value.  Same as :cpp:func:`array::rate`.
  
----

.. cpp:function:: size_t arrayANY::const_view::size() const

  Total number of elements in view, e.g., *nx* |times| *ny* |times| *nz* for
  3D views.

With the above definitions, the following example shows how a 2D view is
constructed and accessed::

  zfp::array2d a(200, 100, rate); // define 200x100 array of doubles
  zfp::array2d::const_view v(&a, 10, 5, 20, 20); // v is a 20x20 view into array a
  assert(v(2, 1) == a(12, 6)); // v(2, 1) == a(10 + 2, 5 + 1) == a(12, 6)
  assert(v.size() == 400); // 20x20 == 400


.. _mutable_view:

Mutable view
^^^^^^^^^^^^

The basic mutable :code:`view` derives from the :code:`const_view` but
adds operators for write-access.  Its constructors are similar to those
for the :code:`const_view`.

..
  .. cpp:class:: template<typename Scalar> array1::view
  .. cpp:class:: template<typename Scalar> array2::view
  .. cpp:class:: template<typename Scalar> array3::view
  .. cpp:class:: template<typename Scalar> array4::view

.. cpp:class:: array1::view : public array1::const_view
.. cpp:class:: array2::view : public array2::const_view
.. cpp:class:: array3::view : public array3::const_view
.. cpp:class:: array4::view : public array4::const_view

  Mutable view into 1D, 2D, 3D, and 4D array.

----

.. cpp:function:: array1::view::view(array1* array)
.. cpp:function:: array2::view::view(array2* array)
.. cpp:function:: array3::view::view(array3* array)
.. cpp:function:: array4::view::view(array4* array)
.. cpp:function:: array1::view::view(array1* array, size_t x, size_t nx)
.. cpp:function:: array2::view::view(array2* array, size_t x, size_t y, size_t nx, size_t ny)
.. cpp:function:: array3::view::view(array3* array, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz)
.. cpp:function:: array4::view::view(array4* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw)

  Whole-array and sub-array mutable view constructors.  See
  :ref:`const_view constructors <view_ctor>` for details.

----

.. cpp:function:: reference array1::view::operator()(size_t i)
.. cpp:function:: reference array2::view::operator()(size_t i, size_t j)
.. cpp:function:: reference array3::view::operator()(size_t i, size_t j, size_t k)
.. cpp:function:: reference array4::view::operator()(size_t i, size_t j, size_t k, size_t l)

  These operators, whose arguments have the same meaning as in the
  :ref:`array accessors <array_accessor>`, return
  :ref:`proxy references <references>` to individual array elements for
  write access.


.. _flat_view:

Flat view
^^^^^^^^^

The views discussed so far require multidimensional indexing, e.g.,
(*i*, *j*, *k*) for 3D views.  Some applications prefer one-dimensional
linear indexing, which is provided by the specialized flat view.  For
example, in a 3D view with dimensions (*nx*, *ny*, *nz*), a multidimensional
index (*i*, *j*, *k*) corresponds to the flat view index
::

  index = i + nx * (j + ny * k)

This is true regardless of the view offset (*x*, *y*, *z*).

The flat view derives from the mutable view and adds :code:`operator[]`
for flat indexing.  This operator is essentially equivalent to
:cpp:func:`array::operator[]` defined for 2D, 3D, and 4D arrays.  Flat
views also provide functions for converting between multidimensional and
flat indices.  

Flat views are available only for 2D, 3D, and 4D arrays.  The basic mutable
view, :cpp:class:`array1::view`, for 1D arrays can be thought of as
either a flat or a nested view.

..
  .. cpp:class:: template<typename Scalar> array2::flat_view
  .. cpp:class:: template<typename Scalar> array3::flat_view
  .. cpp:class:: template<typename Scalar> array4::flat_view

.. cpp:class:: array2::flat_view : public array2::view
.. cpp:class:: array3::flat_view : public array3::view
.. cpp:class:: array4::flat_view : public array4::view

  Flat, mutable views for 2D, 3D, and 4D arrays.

----

.. cpp:function:: array2::flat_view::flat_view(array2* array)
.. cpp:function:: array3::flat_view::flat_view(array3* array)
.. cpp:function:: array4::flat_view::flat_view(array4* array)
.. cpp:function:: array2::flat_view::flat_view(array2* array, size_t x, size_t y, size_t nx, size_t ny)
.. cpp:function:: array3::flat_view::flat_view(array3* array, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz)
.. cpp:function:: array4::flat_view::flat_view(array4* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw)

  Whole-array and sub-array flat view constructors.  See
  :ref:`const_view constructors <view_ctor>` for details.

----

.. cpp:function:: size_t array2::flat_view::index(size_t i, size_t j) const
.. cpp:function:: size_t array3::flat_view::index(size_t i, size_t j, size_t k) const
.. cpp:function:: size_t array4::flat_view::index(size_t i, size_t j, size_t k, size_t l) const

  Return flat index associated with multidimensional index.

----

.. cpp:function:: void array2::flat_view::ij(size_t& i, size_t& j, size_t index) const
.. cpp:function:: void array3::flat_view::ijk(size_t& i, size_t& j, size_t& k, size_t index) const
.. cpp:function:: void array4::flat_view::ijkl(size_t& i, size_t& j, size_t& k, size_t& l, size_t index) const

  Convert flat index to multidimensional index.

----

.. cpp:function:: const_reference array2::flat_view::operator[](size_t index) const
.. cpp:function:: const_reference array3::flat_view::operator[](size_t index) const
.. cpp:function:: const_reference array4::flat_view::operator[](size_t index) const

  Return array element associated with given flat index.

----

.. cpp:function:: reference array2::flat_view::operator[](size_t index)
.. cpp:function:: reference array3::flat_view::operator[](size_t index)
.. cpp:function:: reference array4::flat_view::operator[](size_t index)

  Return reference to array element associated with given flat index.


.. _nested_view:

Nested view
^^^^^^^^^^^

C and C++ support nested arrays (arrays of arrays), e.g.,
:code:`double a[10][20][30]`, which are usually accessed via nested indexing
:code:`a[i][j][k]`.  Here :code:`a` is a 3D array, :code:`a[i]` is a 2D array,
and :code:`a[i][j]` is a 1D array.  This 3D array can also be accessed
via flat indexing, e.g.,
::

  a[i][j][k] == (&a[0][0][0])[600 * i + 30 * j + k]

Nested views provide a mechanism to access array elements through
nested indexing and to extract lower-dimensional "slices" of
multidimensional arrays.  Nested views are mutable.

Nested views are associated with a dimensionality.  For instance,
if :code:`v` is a 3D nested view of a 3D array, then :code:`v[i]`
is a 2D nested view (of a 3D array), :code:`v[i][j]` is a 1D nested
view (of a 3D array), and :code:`v[i][j][k]` is a (reference to a) scalar
array element.  Note that the order of indices is reversed when using
nested indexing compared to multidimensional indexing, e.g.,
:code:`v(i, j, k) == v[k][j][i]`.

Whereas :code:`operator[]` on an array object accesses an element
through flat indexing, the same array can be accessed through a
nested view to in effect provide nested array indexing::

  zfp::array3d a(30, 20, 10, rate); // define 30x20x10 3D array
  assert(a[32] == a(2, 1, 0)); // OK: flat and multidimensional indexing
  assert(a[32] == a[0][1][2]); // ERROR: a does not support nested indexing
  zfp::array3d::nested_view v(&a); // define a nested view of a
  assert(a[32] == v[0][1][2]); // OK: v supports nested indexing
  zfp::array2d b(v[5]); // define and deep copy 30x20 2D slice of a
  assert(a(2, 1, 5) == b(2, 1)); // OK: multidimensional indexing

..
  .. cpp:class:: template<typename Scalar> array2::nested_view1

.. cpp:class:: array2::nested_view1

  View of a 1D slice of a 2D array.

----

..
  .. cpp:class:: template<typename Scalar> array2::nested_view2

.. cpp:class:: array2::nested_view2

  2D view of a 2D (sub)array.

----

..
  .. cpp:class:: template<typename Scalar> array3::nested_view1

.. cpp:class:: array3::nested_view1

  View of a 1D slice of a 3D array.

----

..
  .. cpp:class:: template<typename Scalar> array3::nested_view2

.. cpp:class:: array3::nested_view2

  View of a 2D slice of a 3D array.

----

..
  .. cpp:class:: template<typename Scalar> array3::nested_view3

.. cpp:class:: array3::nested_view3

  3D view of a 3D (sub)array.

----

..
  .. cpp:class:: template<typename Scalar> array4::nested_view1

.. cpp:class:: array4::nested_view1

  View of a 1D slice of a 4D array.

----

..
  .. cpp:class:: template<typename Scalar> array4::nested_view2

.. cpp:class:: array4::nested_view2

  View of a 2D slice of a 4D array.

----

..
  .. cpp:class:: template<typename Scalar> array4::nested_view3

.. cpp:class:: array4::nested_view3

  View of a 3D slice of a 4D array.

----

..
  .. cpp:class:: template<typename Scalar> array4::nested_view4

.. cpp:class:: array4::nested_view4

  4D view of a 4D (sub)array.

----

.. cpp:function:: array2::nested_view2::nested_view2(array2* array)
.. cpp:function:: array3::nested_view3::nested_view3(array3* array)
.. cpp:function:: array4::nested_view4::nested_view4(array4* array)
.. cpp:function:: array2::nested_view2::nested_view2(array2* array, size_t x, size_t y, size_t nx, size_t ny)
.. cpp:function:: array3::nested_view3::nested_view3(array3* array, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz)
.. cpp:function:: array4::nested_view4::nested_view4(array4* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw)

  Whole-array and sub-array nested view constructors.  See
  :ref:`const_view <immutable_view>` constructors for details.
  Lower-dimensional view constructors are not accessible to the
  user but are invoked when accessing views via nested indexing.

----

.. cpp:function:: size_t array2::nested_view1::size_x() const
.. cpp:function:: size_t array2::nested_view2::size_x() const
.. cpp:function:: size_t array2::nested_view2::size_y() const
.. cpp:function:: size_t array3::nested_view1::size_x() const
.. cpp:function:: size_t array3::nested_view2::size_x() const
.. cpp:function:: size_t array3::nested_view2::size_y() const
.. cpp:function:: size_t array3::nested_view3::size_x() const
.. cpp:function:: size_t array3::nested_view3::size_y() const
.. cpp:function:: size_t array3::nested_view3::size_z() const
.. cpp:function:: size_t array4::nested_view1::size_x() const
.. cpp:function:: size_t array4::nested_view2::size_x() const
.. cpp:function:: size_t array4::nested_view2::size_y() const
.. cpp:function:: size_t array4::nested_view3::size_x() const
.. cpp:function:: size_t array4::nested_view3::size_y() const
.. cpp:function:: size_t array4::nested_view3::size_z() const
.. cpp:function:: size_t array4::nested_view4::size_x() const
.. cpp:function:: size_t array4::nested_view4::size_y() const
.. cpp:function:: size_t array4::nested_view4::size_z() const
.. cpp:function:: size_t array4::nested_view4::size_w() const

  View dimensions.

----

.. cpp:function:: array4::nested_view3 array4::nested_view4::operator[](size_t index) const

  Return view to a 3D slice of 4D array.

----

.. cpp:function:: array3::nested_view2 array3::nested_view3::operator[](size_t index) const
.. cpp:function:: array4::nested_view2 array4::nested_view3::operator[](size_t index) const

  Return view to a 2D slice of a 3D or 4D array.

----

.. cpp:function:: array2::nested_view1 array2::nested_view2::operator[](size_t index) const
.. cpp:function:: array3::nested_view1 array3::nested_view2::operator[](size_t index) const
.. cpp:function:: array4::nested_view1 array4::nested_view2::operator[](size_t index) const

  Return view to a 1D slice of a 2D, 3D, or 4D array.

----

.. cpp:function:: const_reference array2::nested_view1::operator[](size_t index) const
.. cpp:function:: const_reference array3::nested_view1::operator[](size_t index) const
.. cpp:function:: const_reference array4::nested_view1::operator[](size_t index) const

  Return scalar element of a 2D, 3D, or 4D array.

----

.. cpp:function:: reference array2::nested_view1::operator[](size_t index)
.. cpp:function:: reference array3::nested_view1::operator[](size_t index)
.. cpp:function:: reference array4::nested_view1::operator[](size_t index)

  Return reference to a scalar element of a 2D, 3D, or 4D array.

----

.. cpp:function:: const_reference array2::nested_view1::operator()(size_t i) const
.. cpp:function:: const_reference array2::nested_view2::operator()(size_t i, size_t j) const
.. cpp:function:: const_reference array3::nested_view1::operator()(size_t i) const
.. cpp:function:: const_reference array3::nested_view2::operator()(size_t i, size_t j) const
.. cpp:function:: const_reference array3::nested_view3::operator()(size_t i, size_t j, size_t k) const
.. cpp:function:: const_reference array4::nested_view1::operator()(size_t i) const
.. cpp:function:: const_reference array4::nested_view2::operator()(size_t i, size_t j) const
.. cpp:function:: const_reference array4::nested_view3::operator()(size_t i, size_t j, size_t k) const
.. cpp:function:: const_reference array4::nested_view4::operator()(size_t i, size_t j, size_t k, size_t l) const

  Return const reference to a scalar element of a 2D, 3D, or 4D array.

----

.. cpp:function:: reference array2::nested_view1::operator()(size_t i)
.. cpp:function:: reference array2::nested_view2::operator()(size_t i, size_t j)
.. cpp:function:: reference array3::nested_view1::operator()(size_t i)
.. cpp:function:: reference array3::nested_view2::operator()(size_t i, size_t j)
.. cpp:function:: reference array3::nested_view3::operator()(size_t i, size_t j, size_t k)
.. cpp:function:: reference array4::nested_view1::operator()(size_t i)
.. cpp:function:: reference array4::nested_view2::operator()(size_t i, size_t j)
.. cpp:function:: reference array4::nested_view3::operator()(size_t i, size_t j, size_t k)
.. cpp:function:: reference array4::nested_view4::operator()(size_t i, size_t j, size_t k, size_t l)

  Return reference to a scalar element of a 2D, 3D, or 4D array.


.. _slicing:

Slicing
^^^^^^^

Arrays can be constructed as deep copies of slices of higher-dimensional
arrays, as the code example above shows (i.e.,
:code:`zfp::array2d b(v[5]);`).  Unlike views, which have reference
semantics, such array *slicing* has value semantics.  In this example,
2D array *b* is initialized as a (deep) copy of a slice of 3D array *a*
via nested view *v*.  Subsequent modifications of *b* have no effect on
*a*.

Slicing is implemented as array constructors templated on views.
Upon initialization, elements are copied one at a time from the view
via multidimensional indexing, e.g., :code:`v(i, j, k)`.  Note that
view and array dimensionalities must match, but aside from this an
array may be constructed from any view.

Slicing needs not change the dimensionality, but can be used to copy
an equidimensional subset of one array to another array, as in this
example::

  zfp::array3d a(30, 20, 10, rate);
  zfp::array3d::const_view v(&a, 1, 2, 3, 4, 5, 6);
  zfp::array3d b(v);
  assert(b(0, 0, 0) == a(1, 2, 3));
  assert(b.size_x() == 4);
  assert(b.size_y() == 5);
  assert(b.size_z() == 6);

Slicing adds the following templated array constructors.

.. cpp:function:: template<class View> array1::array1(const View& v)
.. cpp:function:: template<class View> array2::array2(const View& v)
.. cpp:function:: template<class View> array3::array3(const View& v)
.. cpp:function:: template<class View> array4::array4(const View& v)

  Construct array from a view via a deep copy.  The view, *v*, must support
  :ref:`multidimensional indexing <view_accessor>`.
  The rate for the constructed array is initialized to the rate of the array
  associated with the view.  Note that the actual rate may differ if the
  constructed array is a lower-dimensional slice of a higher-dimensional
  array due to lower rate granularity (see FAQ :ref:`#12 <q-granularity>`).
  The cache size of the constructed array is set to the default size.


.. _private_immutable_view:

Private immutable view
^^^^^^^^^^^^^^^^^^^^^^

|zfp|'s compressed arrays are in general not thread-safe.  The main
reason for this is that each array maintains its own cache of
uncompressed blocks.  Race conditions on the cache would occur unless
it were locked upon each and every array access, which would have a
prohibitive performance cost.

To ensure thread-safe access, |zfp| provides private mutable and
immutable views of arrays that maintain their own private caches.
The :code:`private_const_view` immutable view
provides read-only access to the underlying array.  It is similar
to a :ref:`const_view <immutable_view>` in this sense, but differs in
that it maintains its own private cache rather than sharing the
cache owned by the array.  Multiple threads may thus access the
same array in parallel through their own private views.

.. note::
  Thread safety is ensured only for OpenMP threads, and the |zfp|
  views must be compiled by an OpenMP compliant compiler.  As the
  |zfp| compressed-array class implementation is defined in headers,
  the application code using |zfp| must also be compiled with OpenMP
  enabled if multithreaded access to |zfp| arrays is desired.

.. note::
  Private views **do not guarantee cache coherence**.  If, for example,
  the array is modified, then already cached data in a private view is
  not automatically updated.  It is up to the user to ensure cache
  coherence by flushing (compressing modified blocks) or clearing
  (emptying) caches when appropriate.

The cache associated with a private view can be manipulated in the
same way an array's cache can.  For instance, the user may set the
cache size on a per-view basis.

Unlike with :ref:`private mutable views <private_mutable_view>`,
private immutable views may freely access any element in the
array visible through the view, i.e., multiple threads may
read the same array element simultaneously.  For an example of how
to use private views for both read and write multithreaded access,
see the :ref:`diffusion <ex-diffusion>` code example.

Private views support only multidimensional indexing, i.e., they
are neither flat nor nested.

..
  .. cpp:class:: template<typename Scalar> array1::private_const_view
  .. cpp:class:: template<typename Scalar> array2::private_const_view
  .. cpp:class:: template<typename Scalar> array3::private_const_view
  .. cpp:class:: template<typename Scalar> array4::private_const_view

.. _private_const_view:
.. cpp:class:: array1::private_const_view
.. cpp:class:: array2::private_const_view
.. cpp:class:: array3::private_const_view
.. cpp:class:: array4::private_const_view

  Immutable views of 1D, 2D, 3D, and 4D arrays with private caches.

----

.. cpp:function:: array1::private_const_view::private_const_view(array1* array)
.. cpp:function:: array2::private_const_view::private_const_view(array2* array)
.. cpp:function:: array3::private_const_view::private_const_view(array3* array)
.. cpp:function:: array4::private_const_view::private_const_view(array4* array)
.. cpp:function:: array1::private_const_view::private_const_view(array1* array, size_t x, size_t nx)
.. cpp:function:: array2::private_const_view::private_const_view(array2* array, size_t x, size_t y, size_t nx, size_t ny)
.. cpp:function:: array3::private_const_view::private_const_view(array3* array, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz)
.. cpp:function:: array4::private_const_view::private_const_view(array4* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw)

  Whole-array and sub-array private immutable view constructors.  See
  :ref:`const_view constructors <view_ctor>` for details.

----

.. cpp:function:: size_t array1::private_const_view::size_x() const
.. cpp:function:: size_t array2::private_const_view::size_x() const
.. cpp:function:: size_t array2::private_const_view::size_y() const
.. cpp:function:: size_t array3::private_const_view::size_x() const
.. cpp:function:: size_t array3::private_const_view::size_y() const
.. cpp:function:: size_t array3::private_const_view::size_z() const
.. cpp:function:: size_t array4::private_const_view::size_x() const
.. cpp:function:: size_t array4::private_const_view::size_y() const
.. cpp:function:: size_t array4::private_const_view::size_z() const
.. cpp:function:: size_t array4::private_const_view::size_w() const

  View dimensions.

----

.. cpp:function:: const_reference array1::private_const_view::operator()(size_t i) const
.. cpp:function:: const_reference array2::private_const_view::operator()(size_t i, size_t j) const
.. cpp:function:: const_reference array3::private_const_view::operator()(size_t i, size_t j, size_t k) const
.. cpp:function:: const_reference array4::private_const_view::operator()(size_t i, size_t j, size_t k, size_t l) const

  Return const reference to scalar element of a 1D, 2D, 3D, or 4D array.

The following functions are common among all dimensionalities:

.. cpp:function:: size_t arrayANY::private_const_view::cache_size() const
.. cpp:function:: void arrayANY::private_const_view::set_cache_size(size_t csize)
.. cpp:function:: void arrayANY::private_const_view::clear_cache() const

  Cache manipulation.  See :ref:`caching` for details.


.. _private_mutable_view:

Private mutable view
^^^^^^^^^^^^^^^^^^^^

The mutable :code:`private_view` supports both read and write access
and is backed by a private cache.  Because block compression, as needed
to support write access, is not an atomic operation, mutable views
and multithreading imply potential race conditions on the compressed
blocks stored by an array.  Although locking the array or individual
blocks upon compression would be a potential solution, this would either
serialize compression, thus hurting performance, or add a possibly large
memory overhead by maintaining a lock with each block.

.. note::
  To avoid multiple threads simultaneously compressing the same block,
  **private mutable views of an array must reference disjoint,
  block-aligned subarrays** for thread-safe access.  Each block of |4powd|
  array elements must be associated with at most one private mutable view,
  and therefore these views must reference non-overlapping rectangular
  subsets that are aligned on block boundaries, except possibly for partial
  blocks on the array boundary.  (Expert users may alternatively ensure
  serialization of block compression calls and cache coherence in other
  ways, in which case overlapping private views may be permitted.)

Aside from this requirement, the user may partition the array into
disjoint views in whatever manner is suitable for the application.
The :code:`private_view` API supplies a very basic partitioner to
facilitate this task, but may not result in optimal partitions or
good load balance.

When multithreaded write access is desired, any direct accesses to the
array itself (i.e., not through a view) could invoke compression.  Even
a read access may trigger compression if a modified block is evicted
from the cache.  Hence, such direct array accesses should be confined
to serial code sections when private views are used.

As with private immutable views, **cache coherence is not enforced**.
Although this is less of an issue for private mutable views due to
the requirement that views may not overlap, each private mutable view
overlaps an index space with the underlying array whose cache is not
automatically synchronized with the view's private cache.  See
the :ref:`diffusion <ex-diffusion>` for an example of how to enforce
cache coherence with mutable and immutable private views.

The :code:`private_view` class inherits all public functions from
:code:`private_const_view`.

..
  .. cpp:class:: template<typename Scalar> array1::private_view
  .. cpp:class:: template<typename Scalar> array2::private_view
  .. cpp:class:: template<typename Scalar> array3::private_view
  .. cpp:class:: template<typename Scalar> array4::private_view

.. cpp:class:: array1::private_view : public array1::private_const_view
.. cpp:class:: array2::private_view : public array2::private_const_view
.. cpp:class:: array3::private_view : public array3::private_const_view
.. cpp:class:: array4::private_view : public array4::private_const_view

  Mutable views of 1D, 2D, 3D, and 4D arrays with private caches.

----

..
  .. cpp:class:: template<typename Scalar> array1::private_view::view_reference
  .. cpp:class:: template<typename Scalar> array2::private_view::view_reference
  .. cpp:class:: template<typename Scalar> array3::private_view::view_reference
  .. cpp:class:: template<typename Scalar> array4::private_view::view_reference

.. cpp:class:: array1::private_view::view_reference
.. cpp:class:: array2::private_view::view_reference
.. cpp:class:: array3::private_view::view_reference
.. cpp:class:: array4::private_view::view_reference

  Proxy references to array elements specialized for mutable
  private views.

----

.. cpp:function:: array1::private_view::private_view(array1* array)
.. cpp:function:: array2::private_view::private_view(array2* array)
.. cpp:function:: array3::private_view::private_view(array3* array)
.. cpp:function:: array4::private_view::private_view(array4* array)
.. cpp:function:: array1::private_view::private_view(array1* array, size_t x, size_t nx)
.. cpp:function:: array2::private_view::private_view(array2* array, size_t x, size_t y, size_t nx, size_t ny)
.. cpp:function:: array3::private_view::private_view(array3* array, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz)
.. cpp:function:: array4::private_view::private_view(array4* array, size_t x, size_t y, size_t z, size_t w, size_t nx, size_t ny, size_t nz, size_t nw)

  Whole-array and sub-array private mutable view constructors.  See
  :ref:`const_view constructors <view_ctor>` for details.

----

.. cpp:function:: array1::private_view::view_reference array1::private_view::operator()(size_t i) const
.. cpp:function:: array2::private_view::view_reference array2::private_view::operator()(size_t i, size_t j) const
.. cpp:function:: array3::private_view::view_reference array3::private_view::operator()(size_t i, size_t j, size_t k) const
.. cpp:function:: array4::private_view::view_reference array4::private_view::operator()(size_t i, size_t j, size_t k, size_t l) const

  Return reference to a scalar element of a 1D, 2D, 3D, or 4D array.

The following functions are common among all dimensionalities:

.. cpp:function:: void arrayANY::private_view::partition(size_t index, size_t count)

  Partition the current view into *count* roughly equal-size pieces along the
  view's longest dimension and modify the view's extents to match the piece
  indexed by *index*, with 0 |leq| *index* < *count*.
  These functions may be called multiple times, e.g., to recursively
  partition along different dimensions.  The partitioner does not generate
  new views; it merely modifies the current values of the view's offsets
  and dimensions.  Note that this may result in empty views whose dimensions
  are zero, e.g., if there are more pieces than blocks along a dimension.

----

.. cpp:function:: void arrayANY::private_view::flush_cache() const

  Flush cache by compressing any modified blocks and emptying the cache.
