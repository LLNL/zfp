.. index::
   single: cfp
.. _cfp:

C Bindings
----------

.. cpp:namespace:: zfp

|zfp| |cfprelease| adds |cfp|: C language bindings for compressed arrays
via wrappers around the C++ classes.  |zfp| |cfprpirelease| modifies its
API (see below).

The C API has been designed to facilitate working with compressed arrays
without the benefits of C++ operator overloading and self-aware objects,
which greatly simplify the syntax.  Whereas one possible design considered
is to map each C++ method to a C function with a prefix, such as
:code:`zfp_array3d_get(a, i, j, k)` in place of :code:`a(i, j, k)` for
accessing an element of a 3D array of doubles, such code would quickly
become unwieldy when part of longer expressions.

Instead, |cfp| uses the notion of nested C *namespaces* that are structs
of function pointers, such as :code:`cfp.array3d`.  Although this may
seem no more concise than a design based on prefixes, the user may alias
these namespaces (somewhat similar to C++ :code:`using namespace`
declarations) using far shorter names via C macros or local variables.
For instance::

  const cfp_array3d_api _ = cfp.array3d; // _ is a namespace alias
  cfp_array3d a = _.ctor(nx, ny, nz, rate, 0, 0);
  double value = _.get(a, i, j, k);
  _.set(a, i, j, k, value + 1);

which is a substitute for the C++ code
::

  zfp::array3d a(nx, ny, nz, rate, 0, 0);
  double value = a(i, j, k);
  a(i, j, k) = value + 1;

Because the underlying C++ array objects have no corresponding C
representation, and because C objects are not self aware (they have no
implicit :code:`this` pointer), the C interface interacts with compressed
arrays through array object *pointers*, wrapped in structs, that |cfp|
converts to pointers to the corresponding C++ objects.  As a consequence,
|cfp| compressed arrays must be allocated on the heap and must be explicitly
freed via designated destructor functions to avoid memory leaks (this is
not necessary for references, pointers, and iterators, which have their
own C representation).  The C++ constructors are mapped to C by allocating
objects via C++ :code:`new`.  Moreover, the C API requires passing an array
*self pointer* (wrapped within a cfp array struct) in order to manipulate
the array.

As with the :ref:`C++ classes <array_classes>`, array elements can be
accessed via multidimensional array indexing, e.g., :code:`get(array, i, j)`,
and via flat, linear indexing, e.g., :code:`get_flat(array, i + nx * j)`.

.. note::

  The |cfp| API changed in |zfp| |cfprpirelease| by wrapping array
  *self pointers* in structs to align the interface more closely with the
  C++ API and to avoid confusion when discussing arrays (now
  :code:`cfp.array` rather than :code:`cfp.array*`) and pointers to
  arrays (now :code:`cfp.array*` rather than :code:`cfp.array**`).
  Furthermore, |zfp| |cfprpirelease| adds support for proxy references,
  proxy pointers, and iterators that also wrap C++ classes.  Manipulating
  those indirectly via pointers (like the old |cfp| arrays) would require
  additional user effort to destroy dynamically allocated lightweight objects
  and would also reduce code readability, e.g., :code:`cfp_ptr1d*` (whose
  corresponding C++ type is :code:`zfp::array1d::pointer*`) reads more
  naturally as a raw pointer to a proxy pointer than an indirectly referenced
  proxy pointer object that the user must remember to implicitly dereference.


.. _cfp_arrays:

Cfp Arrays
^^^^^^^^^^

|cfp| implements eight array types for 1D, 2D, 3D, and 4D arrays of floats and
doubles.  These array types share many functions that have the same signature.
To reduce redundancy in the documentation, we define fictitious types
:c:type:`cfp_arrayf` and :c:type:`cfp_arrayd` for *N*-dimensional
(1 |leq| *N* |leq| 4) arrays of floats or doubles, :c:type:`cfp_array1`,
:c:type:`cfp_array2`, and :c:type:`cfp_array3` for 1D, 2D, 3D, and 4D arrays
of either floats or doubles, and :c:type:`cfp_array` for arrays of any
dimensionality and type.  We also make use of corresponding namespaces,
e.g., :c:struct:`cfp.array1` refers to the API common to one-dimensional
arrays of floats or doubles.  These types and namespaces are not actually
part of the |cfp| API.

.. Note::

  Support for 4D arrays was added to cfp in version |cfprpirelease|.

-----

.. c:type:: cfp_array1f
.. c:type:: cfp_array2f
.. c:type:: cfp_array3f
.. c:type:: cfp_array4f
.. c:type:: cfp_array1d
.. c:type:: cfp_array2d
.. c:type:: cfp_array3d
.. c:type:: cfp_array4d

  Opaque types for 1D, 2D, 3D, and 4D compressed arrays of floats and doubles.

-----

.. c:type:: cfp_array1
.. c:type:: cfp_array2
.. c:type:: cfp_array3
.. c:type:: cfp_array4

  Fictitious types denoting 1D, 2D, 3D, and 4D arrays of any scalar type.

-----

.. c:type:: cfp_arrayf
.. c:type:: cfp_arrayd

  Fictitious types denoting any-dimensional arrays of floats and doubles.

-----

.. c:type:: cfp_array

  Fictitious type denoting array of any dimensionality and scalar type.

-----

.. c:struct:: cfp

  .. c:struct:: array1f
  .. c:struct:: array1d
  .. c:struct:: array2f
  .. c:struct:: array2d
  .. c:struct:: array3f
  .. c:struct:: array3d
  .. c:struct:: array4f
  .. c:struct:: array4d

  Nested C "namespaces" for encapsulating the |cfp| API.  The outer
  :c:struct:`cfp` namespace may be redefined at compile-time via the macro
  :c:macro:`CFP_NAMESPACE`, e.g., to avoid symbol clashes.  The inner
  namespaces hold function pointers to the |cfp| wrappers documented below.

-----

.. c:function:: cfp_array1f cfp.array1f.ctor(uint nx, double rate, const float* p, size_t csize)
.. c:function:: cfp_array1d cfp.array1d.ctor(uint nx, double rate, const double* p, size_t csize)
.. c:function:: cfp_array2f cfp.array2f.ctor(uint nx, uint ny, double rate, const float* p, size_t csize)
.. c:function:: cfp_array2d cfp.array2d.ctor(uint nx, uint ny, double rate, const double* p, size_t csize)
.. c:function:: cfp_array3f cfp.array3f.ctor(uint nx, uint ny, uint nz, double rate, const float* p, size_t csize)
.. c:function:: cfp_array3d cfp.array3d.ctor(uint nx, uint ny, uint nz, double rate, const double* p, size_t csize)
.. c:function:: cfp_array4f cfp.array4f.ctor(uint nx, uint ny, uint nz, uint nw, double rate, const float* p, size_t csize)
.. c:function:: cfp_array4d cfp.array4d.ctor(uint nx, uint ny, uint nz, uint nw, double rate, const double* p, size_t csize)

  :ref:`Array constructors <array_ctor>`.
  If *p* is not :code:`NULL`, then the array is initialized from uncompressed
  storage; otherwise the array is zero initialized.  *csize* is the minimum
  size cache (in bytes) to use.  If *csize* is zero, a default size is chosen.

-----

.. c:function:: cfp_array cfp.array.ctor_default()
  
  Default constructor.  Allocate an empty array that later can be
  :ref:`resized <cfp_resize>` and whose rate and cache size can be
  set by :c:func:`cfp.array.set_rate` and
  :c:func:`cfp.array.set_cache_size`.

-----

.. c:function:: cfp_array cfp.array.ctor_copy(const cfp_array src)

  :ref:`Copy constructor <array_ctor_default>`.

-----

.. c:function:: void cfp.array.dtor(cfp_array self)

  Destructor.  The destructor not only deallocates any compressed data
  owned by the array, but also frees memory for itself, invalidating
  the *self* object upon return.  Note that the user must explicitly
  call the destructor to avoid memory leaks.

-----

.. c:function:: void cfp.array.deep_copy(cfp_array self, const cfp_array src)

  Perform a deep copy of *src* analogous to the
  :ref:`C++ assignment operator <array_copy>`.

-----

.. c:function:: float cfp.array1f.get(const cfp_array1f a, uint i)
.. c:function:: float cfp.array2f.get(const cfp_array2f a, uint i, uint j)
.. c:function:: float cfp.array3f.get(const cfp_array3f a, uint i, uint j, uint k)
.. c:function:: float cfp.array4f.get(const cfp_array4f a, uint i, uint j, uint k, uint l)
.. c:function:: double cfp.array1d.get(const cfp_array1d a, uint i)
.. c:function:: double cfp.array2d.get(const cfp_array2d a, uint i, uint j)
.. c:function:: double cfp.array3d.get(const cfp_array3d a, uint i, uint j, uint k)
.. c:function:: double cfp.array4d.get(const cfp_array4d a, uint i, uint j, uint k, uint l)

  :ref:`Array accessors <array_accessor>` via multidimensional indexing.

-----

.. c:function:: void cfp.array1f.set(const cfp_array1f a, uint i, float val)
.. c:function:: void cfp.array2f.set(const cfp_array2f a, uint i, uint j, float val)
.. c:function:: void cfp.array3f.set(const cfp_array3f a, uint i, uint j, uint k, float val)
.. c:function:: void cfp.array4f.set(const cfp_array4f a, uint i, uint j, uint k, uint l, float val)
.. c:function:: void cfp.array1d.set(const cfp_array1d a, uint i, double val)
.. c:function:: void cfp.array2d.set(const cfp_array2d a, uint i, uint j, double val)
.. c:function:: void cfp.array3d.set(const cfp_array3d a, uint i, uint j, uint k, double val)
.. c:function:: void cfp.array4d.set(const cfp_array4d a, uint i, uint j, uint k, uint l, double val)

  Array mutators for assigning values to array elements via multidimensional
  indexing.

-----

.. c:function:: float cfp.arrayf.get_flat(const cfp_arrayf self, uint index)
.. c:function:: double cfp.arrayd.get_flat(const cfp_arrayd self, uint index)

  Flat index array accessors; see :cpp:func:`array::operator[]`.

-----

.. c:function:: void cfp.arrayf.set_flat(cfp_arrayf self, uint index, float val)
.. c:function:: void cfp.arrayd.set_flat(cfp_arrayd self, uint index, double val)

  Flat index array mutators; set array element with flat *index* to *val*.

-----

.. c:function:: void cfp.arrayf.get_array(const cfp_arrayf self, float* p)
.. c:function:: void cfp.arrayd.get_array(const cfp_arrayd self, double* p)

  Decompress entire array; see :cpp:func:`array::get`.

-----

.. c:function:: void cfp.arrayf.set_array(cfp_arrayf self, const float* p)
.. c:function:: void cfp.arrayd.set_array(cfp_arrayd self, const double* p)

  Initialize entire array; see :cpp:func:`array::set`.

-----

.. c:function:: uint cfp.array2.size_x(const cfp_array2 self)
.. c:function:: uint cfp.array2.size_y(const cfp_array2 self)
.. c:function:: uint cfp.array3.size_x(const cfp_array3 self)
.. c:function:: uint cfp.array3.size_y(const cfp_array3 self)
.. c:function:: uint cfp.array3.size_z(const cfp_array3 self)
.. c:function:: uint cfp.array4.size_x(const cfp_array4 self)
.. c:function:: uint cfp.array4.size_y(const cfp_array4 self)
.. c:function:: uint cfp.array4.size_z(const cfp_array4 self)
.. c:function:: uint cfp.array4.size_w(const cfp_array4 self)

  :ref:`Array dimensions <array_dims>`.

-----

.. _cfp_resize:
.. c:function:: void cfp.array1.resize(cfp_array1 self, uint n, int clear)
.. c:function:: void cfp.array2.resize(cfp_array2 self, uint nx, uint ny, int clear)
.. c:function:: void cfp.array3.resize(cfp_array3 self, uint nx, uint ny, uint nz, int clear)
.. c:function:: void cfp.array4.resize(cfp_array4 self, uint nx, uint ny, uint nz, uint nw, int clear)

  :ref:`Resize array <array_resize>`.

-----

.. c:function:: size_t cfp.array.size(const cfp_array self)

  See :cpp:func:`array::size`.

-----

.. c:function:: double cfp.array.rate(const cfp_array self)

  See :cpp:func:`array::rate`.

-----

.. c:function:: double cfp.array.set_rate(cfp_array self, double rate)

  See :cpp:func:`array::set_rate`.

-----

.. c:function:: size_t cfp.array.cache_size(const cfp_array self)

  See :cpp:func:`array::cache_size`.

-----

.. c:function:: void cfp.array.set_cache_size(cfp_array self, size_t csize)

  See :cpp:func:`array::set_cache_size`.

-----

.. c:function:: void cfp.array.clear_cache(const cfp_array self)

  See :cpp:func:`array::clear_cache`.

-----

.. c:function:: void cfp.array.flush_cache(const cfp_array self)

  See :cpp:func:`array::flush_cache`.

-----

.. c:function:: size_t cfp.array.compressed_size(const cfp_array self)

  See :cpp:func:`array::compressed_size`.

-----

.. c:function:: uchar* cfp.array.compressed_data(const cfp_array self)

  See :cpp:func:`array::compressed_data`.


Cfp Array Accessors
###################

.. note ::

  .. cpp:namespace:: zfp::arrayANY
  
  |zfp| |cfprpirelease| adds |cfp| support for proxy
  :ref:`references <references>` and :ref:`pointers <pointers>` to individual
  array elements, as well as :ref:`iterators <iterators>` for traversing arrays.
  These are analogues to the corresponding C++ classes. As with
  :ref:`arrays <cfp_arrays>`, fictitious types and namespaces are used to
  shorten the documentation.
  
  .. _cfp_rpi_value_semantics:
  
  Unlike the case of arrays, for which the surrounding struct stores a pointer
  to the underlying array object to allow modifications of the array, the
  |cfp| proxy reference, proxy pointer, and iterator objects are all passed
  by value, and hence none of the functions below modify the *self* argument.
  To increment a pointer, for instance, one should call
  :c:expr:`p = cfp.array.pointer.inc(p)`. Note that while the references,
  pointers, and iterators are not themselves modified, the array elements
  that they reference can be modified.


.. _cfp_references:

Cfp References
^^^^^^^^^^^^^^

|cfp| proxy references wrap the C++ :ref:`reference <references>` classes.
References are constructed via :c:func:`cfp.array.get_ref`, 
:c:func:`cfp.array.pointer.get_ref`, and :c:func:`cfp.array.iterator.get_ref` 
(as well as associated ref_flat calls).

.. note::

  |cfp| references exist primarily to provide parity with |zfp| references. As 
  references do not exist in C, it is suggested in most cases to use cfp proxy 
  :ref:`array <cfp_arrays>`, :ref:`pointer <cfp_pointers>`, or 
  :ref:`iterator <cfp_iterators>` calls to access array storage.
  
  Cfp references do provide the same guarantees as C++ references, functioning 
  as aliases to initialized members of the cfp wrapped zfp array. This is with 
  the caveat that they are only accessed via cfp API calls (use of the = C 
  assignment operator to shallow copy a :c:type:`cfp_ref` is also allowed in 
  this case).

-----

.. c:type:: cfp_ref1f
.. c:type:: cfp_ref2f
.. c:type:: cfp_ref3f
.. c:type:: cfp_ref4f
.. c:type:: cfp_ref1d
.. c:type:: cfp_ref2d
.. c:type:: cfp_ref3d
.. c:type:: cfp_ref4d

  Opaque types for proxy references to 1D, 2D, 3D, and 4D compressed float or
  double array elements.

-----

.. c:type:: cfp_ref1
.. c:type:: cfp_ref2
.. c:type:: cfp_ref3
.. c:type:: cfp_ref4

  Fictitious types denoting references into 1D, 2D, 3D, and 4D arrays of any
  scalar type.

-----

.. c:type:: cfp_reff
.. c:type:: cfp_refd

  Fictitious types denoting references into float or double arrays of any
  dimensionality.

-----

.. c:type:: cfp_ref

  Fictitious type denoting reference into array of any dimensionality and
  scalar type.

-----

.. c:function:: cfp_ref1 cfp.array1.ref(cfp_array1 self, uint i)
.. c:function:: cfp_ref2 cfp.array2.ref(cfp_array2 self, uint i, uint j)
.. c:function:: cfp_ref3 cfp.array3.ref(cfp_array3 self, uint i, uint j, uint k)
.. c:function:: cfp_ref4 cfp.array4.ref(cfp_array4 self, uint i, uint j, uint k, uint l)

  Reference :ref:`constructor <lvref_bidx>` via multidimensional indexing.

-----

.. c:function:: cfp_ref cfp.array.ref_flat(cfp_array self, uint i)

  Reference :ref:`constructor <lvref_idx>` via flat indexing.

-----

.. c:function:: void cfp.array.reference.copy(cfp_ref self, cfp_ref src)

  Copy value referenced by *src* to value referenced by *self*;
  see :cpp:func:`reference::operator=()`.  This performs a
  deep copy.  This is in contrast to :code:`self = src;`, which performs
  only a shallow copy.

-----

.. c:function:: float  cfp.arrayf.reference.get(cfp_reff self)
.. c:function:: double cfp.arrayd.reference.get(cfp_refd self)

  Retrieve value referenced by *self*.

-----

.. c:function:: void cfp.arrayf.reference.set(cfp_reff self, float  val)
.. c:function:: void cfp.arrayd.reference.set(cfp_refd self, double val)

  Update value referenced by *self*;
  see :cpp:func:`reference::operator=()`.

-----

.. c:function:: cfp_ptr cfp.array.reference.ptr(cfp_ref self)

  Obtain proxy pointer to value referenced by *self*;
  see :cpp:func:`reference::operator&()`.


.. _cfp_pointers:

Cfp Pointers
^^^^^^^^^^^^

|cfp| proxy pointers wrap the C++ :ref:`pointer <pointers>` classes.
Pointers are constructed via :c:func:`cfp.array.ptr` and
:c:func:`cfp.array.reference.ptr` (and associated ptr_flat calls). 
All pointers are :ref:`passed by value <cfp_rpi_value_semantics>` 
and are themselves not modified by these functions.


.. note::

  As with :cpp:class:`array::pointer`, :c:type:`cfp_ptr` indexing is 
  based on row-major ordering and is unaware of zfp blocks. This 
  may result in a suboptimal access pattern if sequentially 
  accessing array members. To take advantage of zfp block 
  traversal optimization see :ref:`iterators <cfp_iterators>`.

-----

.. c:type:: cfp_ptr1f
.. c:type:: cfp_ptr2f
.. c:type:: cfp_ptr3f
.. c:type:: cfp_ptr4f
.. c:type:: cfp_ptr1d
.. c:type:: cfp_ptr2d
.. c:type:: cfp_ptr3d
.. c:type:: cfp_ptr4d

  Opaque types for proxy pointers to 1D, 2D, 3D, and 4D compressed float or
  double array elements.

-----

.. c:type:: cfp_ptr1
.. c:type:: cfp_ptr2
.. c:type:: cfp_ptr3
.. c:type:: cfp_ptr4

  Fictitious types denoting pointers into 1D, 2D, 3D, and 4D arrays of any
  scalar type.

-----

.. c:type:: cfp_ptrf
.. c:type:: cfp_ptrd

  Fictitious types denoting pointers into float or double arrays of any
  dimensionality.

-----

.. c:type:: cfp_ptr

  Fictitious type denoting pointer into array of any dimensionality and
  scalar type.

-----

.. c:function:: cfp_ptr1 cfp.array1.ptr(cfp_array1 self, uint i)
.. c:function:: cfp_ptr2 cfp.array2.ptr(cfp_array2 self, uint i, uint j)
.. c:function:: cfp_ptr3 cfp.array3.ptr(cfp_array3 self, uint i, uint j, uint k)
.. c:function:: cfp_ptr4 cfp.array4.ptr(cfp_array4 self, uint i, uint j, uint k, uint l)

  Obtain pointer to array element via multidimensional indexing.

-----

.. c:function:: cfp_ptr cfp.array.ptr_flat(cfp_array self, uint i)

  Obtain pointer to array element via flat indexing.

-----

.. c:function:: float cfp.arrayf.pointer.get(cfp_ptrf self)
.. c:function:: double cfp.arrayd.pointer.get(cfp_ptrd self)

  Dereference operator; :c:expr:`*self`.
  See :cpp:func:`pointer::operator*()`.

-----

.. c:function:: float cfp.arrayf.pointer.get_at(cfp_ptrf self, ptrdiff_t d)
.. c:function:: double cfp.arrayd.pointer.get_at(cfp_ptrd self, ptrdiff_t d)

  Offset dereference operator; :c:expr:`self[d]`.
  See :cpp:func:`pointer::operator[]()`.

-----

.. c:function:: void cfp.arrayf.pointer.set(cfp_ptrf self, float val)
.. c:function:: void cfp.arrayd.pointer.set(cfp_ptrd self, double val)

  Dereference operator with assignment; :c:expr:`*self = val`.
  See :cpp:func:`pointer::operator*()`.

-----

.. c:function:: void cfp.arrayf.pointer.set_at(cfp_ptrf self, ptrdiff_t d, float val)
.. c:function:: void cfp.arrayd.pointer.set_at(cfp_ptrd self, ptrdiff_t d, double val)

  Offset dereference operator with assignment; :c:expr:`self[d] = val`.
  See :cpp:func:`pointer::operator[]()`.

-----

.. c:function:: cfp_ref cfp.array.pointer.ref(cfp_ptr self)

  Get proxy reference to element stored at :code:`*self`.
  See :cpp:func:`pointer::operator*()`.

-----

.. c:function:: cfp_ref cfp.array.pointer.ref_at(cfp_ptr self, ptrdiff_t d)

  Get proxy reference to element stored at :code:`self[d]`.
  See :cpp:func:`pointer::operator[]()`.

-----

.. c:function:: int cfp.array.pointer.lt(cfp_ptr self, cfp_ptr src)
.. c:function:: int cfp.array.pointer.gt(cfp_ptr self, cfp_ptr src)
.. c:function:: int cfp.array.pointer.leq(cfp_ptr self, cfp_ptr src)
.. c:function:: int cfp.array.pointer.geq(cfp_ptr self, cfp_ptr src)
  
  Return true if the two pointers satisfy the given relationship;
  :c:expr:`self < it`, :c:expr:`self > it`, :c:expr:`self <= it`, 
  :c:expr:`self >= it`.

-----

.. c:function:: int cfp.array.pointer.eq(cfp_ptr self, cfp_ptr src)

  Compare two proxy pointers for equality; :c:expr:`self == src`.
  The pointers must be to elements with the same index within the same
  array to satisfy equality.  See :cpp:func:`pointer::operator==()`.

-----

.. c:function:: int cfp.array.pointer.neq(cfp_ptr self, cfp_ptr src)

  Compare two proxy pointers for inequality; :c:expr:`self != src`.
  The pointers are not equal if they point to different arrays or to
  elements with different index within the same array.  See
  :cpp:func:`pointer::operator!=()`.

-----

.. c:function:: ptrdiff_t cfp.array.pointer.difference(cfp_ptr self, cfp_ptr src)

  Difference between two proxy pointers in number of linear array elements;
  :c:expr:`self - src`.  See :cpp:func:`pointer::operator-()`.

-----

.. c:function:: cfp_ptr cfp.array.pointer.next(cfp_ptr self, ptrdiff_t d)

  Advance proxy pointer by a given distance;
  :c:expr:`self + d`.  See :cpp:func:`pointer::operator+()`.

-----

.. c:function:: cfp_ptr cfp.array.pointer.prev(cfp_ptr self, ptrdiff_t d)

  Move proxy pointer back by a given distance;
  :c:expr:`self - d`.  See :cpp:func:`pointer::operator-()`.

-----

.. c:function:: cfp_ptr cfp.array.pointer.inc(cfp_ptr self)

  Increment proxy pointer and return the result; :c:expr:`self + 1`.
  See :cpp:func:`pointer::operator++()`.

-----

.. c:function:: cfp_ptr cfp.array.pointer.dec(cfp_ptr self)

  Decrement proxy pointer and return the result; :c:expr:`self - 1`.
  See :cpp:func:`pointer::operator--()`.


.. _cfp_iterators:

Cfp Iterators
^^^^^^^^^^^^^

|cfp| iterators wrap the C++ :ref:`iterator <iterators>` classes.
All iterators are :ref:`passed by value <cfp_rpi_value_semantics>` and
are themselves not modified by these functions. Iterators are constructed 
similar to C++ iterators via :c:func:`cfp.array.begin` and 
:c:func:`cfp.array.end`. Iterator usage maps closely to equivalent C++ 
iterator syntax. For example::

  // _ and _iter are namespace aliases
  const cfp_array3d_api      _ = cfp.array3d; 
  const cfp_iterer3d_api _iter = _.iterator;

  cfp_array3d a = _.ctor(nx, ny, nz, rate, 0, 0);
  cfp_iterer3d it;

  for (it = _.begin(a); _iter.neq(it, _.end(a)); it = _iter.inc(it))
    _iter.set(it, (double)(rand() % 1000) / 1000.0);

-----

.. c:type:: cfp_iter1f
.. c:type:: cfp_iter2f
.. c:type:: cfp_iter3f
.. c:type:: cfp_iter4f
.. c:type:: cfp_iter1d
.. c:type:: cfp_iter2d
.. c:type:: cfp_iter3d
.. c:type:: cfp_iter4d

  Opaque types for block iterators over 1D, 2D, 3D, and 4D compressed float
  or double array elements.

-----

.. c:type:: cfp_iter1
.. c:type:: cfp_iter2
.. c:type:: cfp_iter3
.. c:type:: cfp_iter4

  Fictitious types denoting iterators over 1D, 2D, 3D, and 4D arrays of any
  scalar type.

-----

.. c:type:: cfp_iterf
.. c:type:: cfp_iterd

  Fictitious types denoting iterators over float or double arrays of any
  dimensionality.

-----

.. c:type:: cfp_iter

  Fictitious type denoting iterator over array of any dimensionality and
  scalar type.

-----

.. c:function:: cfp_iter cfp.array.begin(cfp_array self)

  Return iterator to beginning of array;
  see :cpp:func:`array::begin()`.

-----

.. c:function:: cfp_iter cfp.array.end(cfp_array self)

  Return iterator to end of array;
  see :cpp:func:`array::end()`.

-----

.. c:function:: float cfp.arrayf.iterator.get(cfp_iterf self)
.. c:function:: double cfp.arrayd.iterator.get(cfp_iterd self)

  Return element referenced by iterator; :c:expr:`*self`.
  See :cpp:func:`iterator::operator*()`.

-----

.. c:function:: float cfp.array1f.iterator.get_at(cfp_iter1f self, ptrdiff_t d)
.. c:function:: double cfp.array1d.iterator.get_at(cfp_iter1d self, ptrdiff_t d)

  Return element *d* elements (may be negative) from iterator (only
  supported for one-dimensional array iterators); :c:expr:`self[d]`.
  See :cpp:func:`array1::iterator::operator[]()`.

-----

.. c:function:: void cfp.arrayf.iterator.set(cfp_iterf self, float val)
.. c:function:: void cfp.arrayd.iterator.set(cfp_iterd self, double val)

  Update element referenced by iterator; :c:expr:`*self = val`.
  See :cpp:func:`iterator::operator*()`.

-----

.. c:function:: void cfp.array1f.iterator.set_at(cfp_iter1 self, ptrdiff_t d, float val)
.. c:function:: void cfp.array1d.iterator.set_at(cfp_iter1 self, ptrdiff_t d, double val)

  Update element *d* elements (may be negative) from iterator (only supported
  for one-dimensional array iterators); :c:expr:`self[d] = val`.
  See :cpp:func:`iterator::operator[]()`.

-----

.. c:function:: cfp_ref cfp.array.iterator.ref(cfp_iter self)

  Return reference to element referenced by iterator; :c:expr:`*self`.
  See :cpp:func:`iterator::operator*()`.

-----

.. c:function:: cfp_ref1 cfp.array1f.iterator.ref_at(cfp_iter1f self, ptrdiff_t d)
.. c:function:: cfp_ref1 cfp.array1d.iterator.ref_at(cfp_iter1d self, ptrdiff_t d)

  Return reference to an element offset *d* elements (may be negative) from
  iterator (only supported for one-dimensional array iterators);
  :c:expr:`self[d]`.  See :cpp:func:`array1::iterator::operator[]()`.

-----

.. c:function:: cfp_ptr cfp.array.iterator.ptr(cfp_iter self)

  Return pointer to element referenced by iterator;
  :c:expr:`ptr = &*iter`.

-----

.. c:function:: cfp_ptr1f cfp.array1f.iterator.ptr_at(cfp_iter1f self, ptrdiff_t d)
.. c:function:: cfp_ptr1d cfp.array1d.iterator.ptr_at(cfp_iter1d self, ptrdiff_t d)

  Return pointer to element offset *d* elements (may be negative) from 
  iterator (only supported for one-dimensional array iterators);
  :c:expr:`ptr = &*iter[d]`.

-----

.. c:function:: int cfp.array1.iterator.lt(cfp_iter1 self, cfp_iter1 it)
.. c:function:: int cfp.array1.iterator.gt(cfp_iter1 self, cfp_iter1 it)
.. c:function:: int cfp.array1.iterator.leq(cfp_iter1 self, cfp_iter1 it)
.. c:function:: int cfp.array1.iterator.geq(cfp_iter1 self, cfp_iter1 it)

  Return true if the two iterators satisfy the given relationship (only
  supported for one-dimensional array iterators); :c:expr:`self < it`,
  :c:expr:`self > it`, :c:expr:`self <= it`, :c:expr:`self >= it`.

-----

.. c:function:: int cfp.array.iterator.eq(cfp_iter self, cfp_iter it)

  Return whether two iterators are equal; :c:expr:`self == it`.
  See :cpp:func:`iterator::operator==()`.

-----

.. c:function:: int cfp.array.iterator.neq(cfp_iter self, cfp_iter it)

  Return whether two iterators are unequal; :c:expr:`self != it`.
  See :cpp:func:`iterator::operator!=()`.

-----

.. c:function:: ptrdiff_t cfp.array1.iterator.distance(cfp_iter1 self, cfp_iter1 it)

  Return the difference between two iterators (only supported for
  one-dimensional array iterators); :c:expr:`it - self`.
  See :cpp:func:`array1::iterator::operator-()`.

-----

.. c:function:: cfp_iter1 cfp.array1.iterator.next(cfp_iter1 self, ptrdiff_t d)

  Advance iterator by *d* elements (only supported for one-dimensional array
  iterators) and return the result; :c:expr:`self + d`.
  See :cpp:func:`array1::iterator::operator+()`.

-----

.. c:function:: cfp_iter1 cfp.array1.iterator.prev(cfp_iter1 self, ptrdiff_t d)

  Move iterator backwards by *d* elements (only supported for one-dimensional 
  array iterators) and return the result; :c:expr:`self - d`.
  See :cpp:func:`array1::iterator::operator-()`.

-----

.. c:function:: cfp_iter cfp.array.iterator.inc(cfp_iter self)

  Increment iterator by one and return the resulting iterator;
  :c:expr:`self + 1`.  See :cpp:func:`iterator::operator++()`.

-----

.. c:function:: cfp_iter cfp.array1.iterator.dec(cfp_iter1 self)

  Decrement iterator by one and return the resulting iterator (only supported
  for one-dimensional array iterators); :c:expr:`self - 1`.
  See :cpp:func:`iterator iterator::operator--()`.

-----

.. c:function:: uint cfp.array1.iterator.i(cfp_iter1 self)
.. c:function:: uint cfp.array2.iterator.i(cfp_iter2 self)
.. c:function:: uint cfp.array3.iterator.i(cfp_iter3 self)
.. c:function:: uint cfp.array4.iterator.i(cfp_iter4 self)

  Get *i* component of array element referenced by iterator; see
  :cpp:func:`iterator::i()`.

-----

.. c:function:: uint cfp.array2.iterator.j(cfp_iter2 self)
.. c:function:: uint cfp.array3.iterator.j(cfp_iter3 self)
.. c:function:: uint cfp.array4.iterator.j(cfp_iter4 self)

  Get *j* component of array element referenced by iterator; see
  :cpp:func:`iterator::j()`.

-----

.. c:function:: uint cfp.array3.iterator.k(cfp_iter3 self)
.. c:function:: uint cfp.array4.iterator.k(cfp_iter4 self)

  Get *k* component of array element referenced by iterator; see
  :cpp:func:`iterator::k()`.

-----

.. c:function:: uint cfp.array4.iterator.l(cfp_iter4 self)

  Get *l* component of array element referenced by iterator; see
  :cpp:func:`iterator::l()`.
