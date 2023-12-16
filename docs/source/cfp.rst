.. include:: defs.rst
.. index::
   single: cfp
.. _cfp:

Compressed-Array C Bindings
===========================

.. cpp:namespace:: zfp

|zfp| |cfprelease| adds |cfp|: C language bindings for compressed arrays
via wrappers around the :ref:`C++ classes <arrays>`.  |zfp| |crpirelease|
modifies its API (see below).

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

  The |cfp| API changed in |zfp| |crpirelease| by wrapping array
  *self pointers* in structs to align the interface more closely with the
  C++ API and to avoid confusion when discussing arrays (now
  :code:`cfp.array` rather than :code:`cfp.array*`) and pointers to
  arrays (now :code:`cfp.array*` rather than :code:`cfp.array**`).
  Furthermore, |zfp| |crpirelease| adds support for proxy references,
  proxy pointers, and iterators that also wrap C++ classes.  Manipulating
  those indirectly via pointers (like the old |cfp| arrays) would require
  additional user effort to destroy dynamically allocated lightweight objects
  and would also reduce code readability, e.g., :code:`cfp_ptr1d*` (whose
  corresponding C++ type is :code:`zfp::array1d::pointer*`) reads more
  naturally as a raw pointer to a proxy pointer than an indirectly referenced
  proxy pointer object that the user must remember to implicitly dereference.

The following sections are available:

* :ref:`cfp_arrays`
* :ref:`cfp_serialization`
* :ref:`cfp_references`
* :ref:`cfp_pointers`
* :ref:`cfp_iterators`


.. _cfp_arrays:

Arrays
------

|cfp| implements eight array types for 1D, 2D, 3D, and 4D arrays of floats and
doubles.  These array types share many functions that have the same signature.
To reduce redundancy in the documentation, we define fictitious types
:c:type:`cfp_arrayf` and :c:type:`cfp_arrayd` for *N*-dimensional
(1 |leq| *N* |leq| 4) arrays of floats or doubles, :c:type:`cfp_array1`,
:c:type:`cfp_array2`, :c:type:`cfp_array3`, and :c:type:`cfp_array4` for
1D, 2D, 3D, and 4D arrays of either floats or doubles, and :c:type:`cfp_array`
for arrays of any dimensionality and type.  We also make use of corresponding
namespaces, e.g., :c:struct:`cfp.array1` refers to the API common to
one-dimensional arrays of floats or doubles.  These types and namespaces are
not actually part of the |cfp| API.

.. note::

  The |cfp| array API makes use of :code:`const` qualifiers for :code:`struct`
  parameters (passed by value) merely to indicate when the corresponding object
  is not modified, e.g., :code:`const cfp_array1f self`.  This construction
  serves to document functions that are analogous to :code:`const` qualified
  C++ member functions.

.. note::

  Support for 4D arrays was added to cfp in version |crpirelease|.

.. c:type:: cfp_array1f
.. c:type:: cfp_array1d
.. c:type:: cfp_array2f
.. c:type:: cfp_array2d
.. c:type:: cfp_array3f
.. c:type:: cfp_array3d
.. c:type:: cfp_array4f
.. c:type:: cfp_array4d

  Opaque types for 1D, 2D, 3D, and 4D compressed arrays of floats and doubles.

----

.. c:type:: cfp_array1
.. c:type:: cfp_array2
.. c:type:: cfp_array3
.. c:type:: cfp_array4

  Fictitious types denoting 1D, 2D, 3D, and 4D arrays of any scalar type.

----

.. c:type:: cfp_arrayf
.. c:type:: cfp_arrayd

  Fictitious types denoting any-dimensional arrays of floats and doubles.

----

.. c:type:: cfp_array

  Fictitious type denoting array of any dimensionality and scalar type.

----

.. c:struct:: cfp

  .. c:struct:: array1f
  .. c:struct:: array1d
  .. c:struct:: array2f
  .. c:struct:: array2d
  .. c:struct:: array3f
  .. c:struct:: array3d
  .. c:struct:: array4f
  .. c:struct:: array4d
  .. c:struct:: header

  Nested C "namespaces" for encapsulating the |cfp| API.  The outer
  :c:struct:`cfp` namespace may be redefined at compile-time via the macro
  :c:macro:`CFP_NAMESPACE`, e.g., to avoid symbol clashes.  The inner
  namespaces hold function pointers to the |cfp| wrappers documented below.

----

.. _cfp_ctor:
.. c:function:: cfp_array1f cfp.array1f.ctor(size_t nx, double rate, const float* p, size_t cache_size)
.. c:function:: cfp_array1d cfp.array1d.ctor(size_t nx, double rate, const double* p, size_t cache_size)
.. c:function:: cfp_array2f cfp.array2f.ctor(size_t nx, size_t ny, double rate, const float* p, size_t cache_size)
.. c:function:: cfp_array2d cfp.array2d.ctor(size_t nx, size_t ny, double rate, const double* p, size_t cache_size)
.. c:function:: cfp_array3f cfp.array3f.ctor(size_t nx, size_t ny, size_t nz, double rate, const float* p, size_t cache_size)
.. c:function:: cfp_array3d cfp.array3d.ctor(size_t nx, size_t ny, size_t nz, double rate, const double* p, size_t cache_size)
.. c:function:: cfp_array4f cfp.array4f.ctor(size_t nx, size_t ny, size_t nz, size_t nw, double rate, const float* p, size_t cache_size)
.. c:function:: cfp_array4d cfp.array4d.ctor(size_t nx, size_t ny, size_t nz, size_t nw, double rate, const double* p, size_t cache_size)

  :ref:`Array constructors <array_ctor>`.
  If *p* is not :code:`NULL`, then the array is initialized from uncompressed
  storage; otherwise the array is zero initialized.  *cache_size* is the
  minimum size cache (in bytes) to use.  If *cache_size* is zero, a default
  size is chosen.

----

.. c:function:: cfp_array cfp.array.ctor_default()
  
  Default constructor.  Allocate an empty array that later can be
  :ref:`resized <cfp_resize>` and whose rate and cache size can be
  set by :c:func:`cfp.array.set_rate` and
  :c:func:`cfp.array.set_cache_size`.

----

.. c:function:: cfp_array cfp.array.ctor_copy(const cfp_array src)

  :ref:`Copy constructor <array_ctor_default>`.

----

.. _cfp_ctor_header:
.. c:function:: cfp_array cfp.array.ctor_header(const cfp_header h, const void* buffer, size_t buffer_size_bytes);

  Constructor from metadata given by the :ref:`header <cfp_header>` *h*
  and optionally initialized with compressed data from *buffer* of
  size *buffer_size_bytes*.
  See :ref:`corresponding C++ constructor <array_ctor_header>`.

----

.. c:function:: void cfp.array.dtor(cfp_array self)

  Destructor.  The destructor not only deallocates any compressed data
  owned by the array, but also frees memory for itself, invalidating
  the *self* object upon return.  Note that the user must explicitly
  call the destructor to avoid memory leaks.

----

.. c:function:: void cfp.array.deep_copy(cfp_array self, const cfp_array src)

  Perform a deep copy of *src* analogous to the
  :ref:`C++ assignment operator <array_copy>`.

----

.. _cfp_inspectors:
.. c:function:: float cfp.array1f.get(const cfp_array1f self, size_t i)
.. c:function:: float cfp.array2f.get(const cfp_array2f self, size_t i, size_t j)
.. c:function:: float cfp.array3f.get(const cfp_array3f self, size_t i, size_t j, size_t k)
.. c:function:: float cfp.array4f.get(const cfp_array4f self, size_t i, size_t j, size_t k, size_t l)
.. c:function:: double cfp.array1d.get(const cfp_array1d self, size_t i)
.. c:function:: double cfp.array2d.get(const cfp_array2d self, size_t i, size_t j)
.. c:function:: double cfp.array3d.get(const cfp_array3d self, size_t i, size_t j, size_t k)
.. c:function:: double cfp.array4d.get(const cfp_array4d self, size_t i, size_t j, size_t k, size_t l)

  :ref:`Array inspectors <array_accessor>` via multidimensional indexing.

----

.. _cfp_mutators:
.. c:function:: void cfp.array1f.set(const cfp_array1f self, size_t i, float val)
.. c:function:: void cfp.array2f.set(const cfp_array2f self, size_t i, size_t j, float val)
.. c:function:: void cfp.array3f.set(const cfp_array3f self, size_t i, size_t j, size_t k, float val)
.. c:function:: void cfp.array4f.set(const cfp_array4f self, size_t i, size_t j, size_t k, size_t l, float val)
.. c:function:: void cfp.array1d.set(const cfp_array1d self, size_t i, double val)
.. c:function:: void cfp.array2d.set(const cfp_array2d self, size_t i, size_t j, double val)
.. c:function:: void cfp.array3d.set(const cfp_array3d self, size_t i, size_t j, size_t k, double val)
.. c:function:: void cfp.array4d.set(const cfp_array4d self, size_t i, size_t j, size_t k, size_t l, double val)

  :ref:`Array mutators <lvref>` for assigning values to array elements via
  multidimensional indexing.

----

.. c:function:: float cfp.arrayf.get_flat(const cfp_arrayf self, size_t index)
.. c:function:: double cfp.arrayd.get_flat(const cfp_arrayd self, size_t index)

  Flat index array inspectors; see :cpp:func:`array::operator[]`.

----

.. c:function:: void cfp.arrayf.set_flat(cfp_arrayf self, size_t index, float val)
.. c:function:: void cfp.arrayd.set_flat(cfp_arrayd self, size_t index, double val)

  Flat index array mutators; set array element with flat *index* to *val*.

----

.. c:function:: void cfp.arrayf.get_array(const cfp_arrayf self, float* p)
.. c:function:: void cfp.arrayd.get_array(const cfp_arrayd self, double* p)

  Decompress entire array; see :cpp:func:`array::get`.

----

.. c:function:: void cfp.arrayf.set_array(cfp_arrayf self, const float* p)
.. c:function:: void cfp.arrayd.set_array(cfp_arrayd self, const double* p)

  Initialize entire array; see :cpp:func:`array::set`.

----

.. c:function:: size_t cfp.array2.size_x(const cfp_array2 self)
.. c:function:: size_t cfp.array2.size_y(const cfp_array2 self)
.. c:function:: size_t cfp.array3.size_x(const cfp_array3 self)
.. c:function:: size_t cfp.array3.size_y(const cfp_array3 self)
.. c:function:: size_t cfp.array3.size_z(const cfp_array3 self)
.. c:function:: size_t cfp.array4.size_x(const cfp_array4 self)
.. c:function:: size_t cfp.array4.size_y(const cfp_array4 self)
.. c:function:: size_t cfp.array4.size_z(const cfp_array4 self)
.. c:function:: size_t cfp.array4.size_w(const cfp_array4 self)

  :ref:`Array dimensions <array_dims>`.

----

.. c:function:: size_t cfp.array.size(const cfp_array self)

  See :cpp:func:`array::size`.

----

.. _cfp_resize:
.. c:function:: void cfp.array1.resize(cfp_array1 self, size_t n, zfp_bool clear)
.. c:function:: void cfp.array2.resize(cfp_array2 self, size_t nx, size_t ny, zfp_bool clear)
.. c:function:: void cfp.array3.resize(cfp_array3 self, size_t nx, size_t ny, size_t nz, zfp_bool clear)
.. c:function:: void cfp.array4.resize(cfp_array4 self, size_t nx, size_t ny, size_t nz, size_t nw, zfp_bool clear)

  :ref:`Resize array <array_resize>`.

----

.. c:function:: double cfp.array.rate(const cfp_array self)

  See :cpp:func:`array::rate`.

----

.. c:function:: double cfp.array.set_rate(cfp_array self, double rate)

  See :cpp:func:`array::set_rate`.

----

.. c:function:: size_t cfp.array.cache_size(const cfp_array self)

  See :cpp:func:`array::cache_size`.

----

.. c:function:: void cfp.array.set_cache_size(cfp_array self, size_t cache_size)

  See :cpp:func:`array::set_cache_size`.

----

.. c:function:: void cfp.array.clear_cache(const cfp_array self)

  See :cpp:func:`array::clear_cache`.

----

.. c:function:: void cfp.array.flush_cache(const cfp_array self)

  See :cpp:func:`array::flush_cache`.

----

.. c:function:: size_t cfp.array.size_bytes(const cfp_array self, uint mask)

  See :cpp:func:`array::size_bytes`.

----

.. c:function:: size_t cfp.array.compressed_size(const cfp_array self)

  See :cpp:func:`array::compressed_size`.

----

.. c:function:: void* cfp.array.compressed_data(const cfp_array self)

  See :cpp:func:`array::compressed_data`.

----

.. c:function:: cfp_ref1 cfp.array1.ref(cfp_array1 self, size_t i)
.. c:function:: cfp_ref2 cfp.array2.ref(cfp_array2 self, size_t i, size_t j)
.. c:function:: cfp_ref3 cfp.array3.ref(cfp_array3 self, size_t i, size_t j, size_t k)
.. c:function:: cfp_ref4 cfp.array4.ref(cfp_array4 self, size_t i, size_t j, size_t k, size_t l)

  Reference :ref:`constructor <lvref>` via multidimensional indexing.

----

.. c:function:: cfp_ref cfp.array.ref_flat(cfp_array self, size_t i)

  Reference :ref:`constructor <lvref_idx>` via flat indexing.

----

.. c:function:: cfp_ptr1 cfp.array1.ptr(cfp_array1 self, size_t i)
.. c:function:: cfp_ptr2 cfp.array2.ptr(cfp_array2 self, size_t i, size_t j)
.. c:function:: cfp_ptr3 cfp.array3.ptr(cfp_array3 self, size_t i, size_t j, size_t k)
.. c:function:: cfp_ptr4 cfp.array4.ptr(cfp_array4 self, size_t i, size_t j, size_t k, size_t l)

  Obtain pointer to array element via multidimensional indexing.

----

.. c:function:: cfp_ptr cfp.array.ptr_flat(cfp_array self, size_t i)

  Obtain pointer to array element via flat indexing.

----

.. c:function:: cfp_iter cfp.array.begin(cfp_array self)

  Return iterator to beginning of array;
  see :cpp:func:`array::begin()`.

----

.. c:function:: cfp_iter cfp.array.end(cfp_array self)

  Return iterator to end of array;
  see :cpp:func:`array::end()`.


.. _cfp_serialization:

Serialization
-------------

.. cpp:namespace:: zfp

|zfp| |crpirelease| adds |cfp| array :ref:`serialization <serialization>`.
Like |zfp|'s C++ arrays, |cfp| arrays can be serialized and deserialized to
and from sequential storage.  As with the C++ arrays, (de)serialization is
done with the assistance of a header class, :c:type:`cfp_header`.  Currently,
|cfp| provides no :ref:`factory function <array_factory>`---the caller must
either know which type of array (dimensionality and scalar type) to
:ref:`construct <cfp_ctor>` at compile-time or obtain this information at
run-time from a header :ref:`constructed <cfp_ctor_header>` from a memory
buffer.

.. _cfp_header:

Header
^^^^^^

:c:type:`cfp_header` is a wrapper around :cpp:class:`array::header`.
Although the header type is shared among all array types, the header API
is accessed through the associated array type whose metadata the header
describes.  For example, :code:`cfp.array3f.header.ctor(const cfp_array3f a)`
constructs a header for a :c:type:`cfp_array3f`.  The header is dynamically
allocated and must be explicitly destructed via
:c:func:`cfp.array.header.dtor`.

.. c:type:: cfp_header

  Wrapper around :cpp:class:`array::header`.

----

.. c:function:: cfp_header cfp.array.header.ctor(const cfp_array a);

  :ref:`Construct <header_ctor>` a header that describes the metadata of an
  existing array *a*.

----

.. c:function:: cfp_header cfp.array.header.ctor_buffer(const void* data, size_t size)

  :ref:`Construct <header_ctor_buffer>` a header from header *data* buffer
  of given byte *size*.

----

.. c:function:: void cfp.array.header.dtor(cfp_header self);

  Destructor.  Deallocates all data associated with the header.  The user
  must call the destructor to avoid memory leaks.

----

.. cpp:namespace:: zfp::array

.. c:function:: zfp_type cfp.array.header.scalar_type(const cfp_header self);

  Scalar type associated with array.  See :cpp:func:`header::scalar_type`.

----

.. c:function:: uint cfp.array.header.dimensionality(const cfp_header self);

  Dimensionality associated with array.
  See :cpp:func:`header::dimensionality`.

----

.. c:function:: size_t cfp.array.header.size_x(const cfp_header self);
.. c:function:: size_t cfp.array.header.size_y(const cfp_header self);
.. c:function:: size_t cfp.array.header.size_z(const cfp_header self);
.. c:function:: size_t cfp.array.header.size_w(const cfp_header self);

  :ref:`Array dimensions <header_dims>`.  Unused dimensions have a size of zero.

----

.. c:function:: double cfp.array.header.rate(const cfp_header self);

  Rate in bits/value.  See :cpp:func:`header::rate`.

----

.. c:function:: const void* cfp.array.header.data(const cfp_header self);

  Pointer to header data buffer needed for serializing the header.
  See :cpp:func:`header::data`.

----

.. c:function:: size_t cfp.array.header.size_bytes(const cfp_header self, uint mask);

  When *mask* = :c:macro:`ZFP_DATA_HEADER`, byte size of header data buffer needed
  for serializing the header.  See :cpp:func:`header::size_bytes`.


Array Accessors
---------------

.. cpp:namespace:: zfp::arrayANY
  
|zfp| |crpirelease| adds |cfp| support for proxy
:ref:`references <references>` and :ref:`pointers <pointers>` to individual
array elements, as well as :ref:`iterators <iterators>` for traversing arrays.
These are analogues to the corresponding C++ classes. As with
:ref:`arrays <cfp_arrays>`, fictitious types and namespaces are used to
shorten the documentation.
  
.. _cfp_rpi_value_semantics:
.. note::
  
  Unlike the case of arrays, for which the surrounding struct stores a pointer
  to the underlying array object to allow modifications of the array, the
  |cfp| proxy reference, proxy pointer, and iterator objects are all passed
  by value, and hence none of the functions below modify the *self* argument.
  To increment a pointer, for instance, one should call
  :code:`p = cfp.array.pointer.inc(p)`. Note that while the references,
  pointers, and iterators are not themselves modified, the array elements
  that they reference can be modified.

.. _cfp_references:

References
----------

|cfp| proxy references wrap the C++ :ref:`reference <references>` classes.
References are constructed via :c:func:`cfp.array.ref`, 
:c:func:`cfp.array.pointer.ref`, and :c:func:`cfp.array.iterator.ref` 
(as well as associated :code:`ref_flat` and :code:`ref_at` calls).

.. note::

  |cfp| references exist primarily to provide parity with |zfp| references.
  As references do not exist in C, the preferred way of accessing arrays is
  via :ref:`proxy pointers <cfp_pointers>`, :ref:`iterators <cfp_iterators>`,
  or :ref:`index-based array accessors <cfp_inspectors>`.
  
  |cfp| references do provide the same guarantees as C++ references,
  functioning as aliases to initialized members of the |cfp| wrapped |zfp|
  array. This is with the caveat that they are only accessed via |cfp| API
  calls (use of the :code:`=` C assignment operator to shallow copy a
  :c:type:`cfp_ref` is also allowed in this case).

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

----

.. c:type:: cfp_ref1
.. c:type:: cfp_ref2
.. c:type:: cfp_ref3
.. c:type:: cfp_ref4

  Fictitious types denoting references into 1D, 2D, 3D, and 4D arrays of any
  scalar type.

----

.. c:type:: cfp_reff
.. c:type:: cfp_refd

  Fictitious types denoting references into float or double arrays of any
  dimensionality.

----

.. c:type:: cfp_ref

  Fictitious type denoting reference into array of any dimensionality and
  scalar type.

----

.. c:function:: float  cfp.arrayf.reference.get(const cfp_reff self)
.. c:function:: double cfp.arrayd.reference.get(const cfp_refd self)

  Retrieve value referenced by *self*.

----

.. c:function:: void cfp.arrayf.reference.set(cfp_reff self, float val)
.. c:function:: void cfp.arrayd.reference.set(cfp_refd self, double val)

  Update value referenced by *self*;
  see :cpp:func:`reference::operator=()`.

----

.. c:function:: cfp_ptr cfp.array.reference.ptr(cfp_ref self)

  Obtain proxy pointer to value referenced by *self*;
  see :cpp:func:`reference::operator&()`.

----

.. c:function:: void cfp.array.reference.copy(cfp_ref self, const cfp_ref src)

  Copy value referenced by *src* to value referenced by *self*;
  see :cpp:func:`reference::operator=()`.  This performs a
  deep copy.  This is in contrast to :code:`self = src`, which performs
  only a shallow copy.


.. _cfp_pointers:

Pointers
--------

|cfp| proxy pointers wrap the C++ :ref:`pointer <pointers>` classes.
Pointers are constructed via :c:func:`cfp.array.ptr` and
:c:func:`cfp.array.reference.ptr` (and associated :code:`ptr_flat` and
:code:`ptr_at` calls).  All pointers are
:ref:`passed by value <cfp_rpi_value_semantics>` 
and are themselves not modified by these functions.

.. note::

  As with :cpp:class:`array::pointer`, :c:type:`cfp_ptr` indexing is 
  based on element-wise ordering and is unaware of |zfp| blocks. This 
  may result in a suboptimal access pattern if sequentially 
  accessing array members. To take advantage of |zfp| block 
  traversal optimization, see :ref:`iterators <cfp_iterators>`.

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

----

.. c:type:: cfp_ptr1
.. c:type:: cfp_ptr2
.. c:type:: cfp_ptr3
.. c:type:: cfp_ptr4

  Fictitious types denoting pointers into 1D, 2D, 3D, and 4D arrays of any
  scalar type.

----

.. c:type:: cfp_ptrf
.. c:type:: cfp_ptrd

  Fictitious types denoting pointers into float or double arrays of any
  dimensionality.

----

.. c:type:: cfp_ptr

  Fictitious type denoting pointer into array of any dimensionality and
  scalar type.

----

.. c:function:: float cfp.arrayf.pointer.get(const cfp_ptrf self)
.. c:function:: double cfp.arrayd.pointer.get(const cfp_ptrd self)

  Dereference operator; :code:`*self`.
  See :cpp:func:`pointer::operator*()`.

----

.. c:function:: float cfp.arrayf.pointer.get_at(const cfp_ptrf self, ptrdiff_t d)
.. c:function:: double cfp.arrayd.pointer.get_at(const cfp_ptrd self, ptrdiff_t d)

  Offset dereference operator; :code:`self[d]`.
  See :cpp:func:`pointer::operator[]()`.

----

.. c:function:: void cfp.arrayf.pointer.set(cfp_ptrf self, float val)
.. c:function:: void cfp.arrayd.pointer.set(cfp_ptrd self, double val)

  Dereference operator with assignment; :code:`*self = val`.
  See :cpp:func:`pointer::operator*()`.

----

.. c:function:: void cfp.arrayf.pointer.set_at(cfp_ptrf self, ptrdiff_t d, float val)
.. c:function:: void cfp.arrayd.pointer.set_at(cfp_ptrd self, ptrdiff_t d, double val)

  Offset dereference operator with assignment; :code:`self[d] = val`.
  See :cpp:func:`pointer::operator[]()`.

----

.. c:function:: cfp_ref cfp.array.pointer.ref(cfp_ptr self)

  Get proxy reference to element stored at :code:`*self`.
  See :cpp:func:`pointer::operator*()`.

----

.. c:function:: cfp_ref cfp.array.pointer.ref_at(cfp_ptr self, ptrdiff_t d)

  Get proxy reference to element stored at :code:`self[d]`.
  See :cpp:func:`pointer::operator[]()`.

----

.. c:function:: zfp_bool cfp.array.pointer.lt(const cfp_ptr lhs, const cfp_ptr rhs)
.. c:function:: zfp_bool cfp.array.pointer.gt(const cfp_ptr lhs, const cfp_ptr rhs)
.. c:function:: zfp_bool cfp.array.pointer.leq(const cfp_ptr lhs, const cfp_ptr rhs)
.. c:function:: zfp_bool cfp.array.pointer.geq(const cfp_ptr lhs, const cfp_ptr rhs)
  
  Return true if the two pointers satisfy the given
  :ref:`relationship <ptr_inequalities>`;
  :code:`lhs < rhs`, :code:`lhs > rhs`, :code:`lhs <= rhs`, :code:`lhs >= rhs`.

----

.. c:function:: zfp_bool cfp.array.pointer.eq(const cfp_ptr lhs, const cfp_ptr rhs)

  Compare two proxy pointers for equality; :code:`lhs == rhs`.
  The pointers must be to elements with the same index within the same
  array to satisfy equality.  See :cpp:func:`pointer::operator==()`.

----

.. c:function:: int cfp.array.pointer.neq(const cfp_ptr lhs, const cfp_ptr rhs)

  Compare two proxy pointers for inequality; :code:`lhs != rhs`.
  The pointers are not equal if they point to different arrays or to
  elements with different index within the same array.  See
  :cpp:func:`pointer::operator!=()`.

----

.. c:function:: ptrdiff_t cfp.array.pointer.distance(const cfp_ptr first, const cfp_ptr last)

  Return the difference between two proxy pointers in number of linear array
  elements; :code:`last - first`.  See :cpp:func:`pointer::operator-()`.

----

.. c:function:: cfp_ptr cfp.array.pointer.next(const cfp_ptr p, ptrdiff_t d)

  Return the result of incrementing pointer by *d* elements; :code:`p + d`.
  See :cpp:func:`pointer::operator+()`.

----

.. c:function:: cfp_ptr cfp.array.pointer.prev(const cfp_ptr p, ptrdiff_t d)

  Return the result of decrementing pointer by *d* elements; :code:`p - d`.
  See :cpp:func:`pointer::operator-()`.

----

.. c:function:: cfp_ptr cfp.array.pointer.inc(const cfp_ptr p)

  Return the result of incrementing pointer by one element; :code:`p + 1`.
  See :cpp:func:`pointer::operator++()`.

----

.. c:function:: cfp_ptr cfp.array.pointer.dec(const cfp_ptr p)

  Return the result of decrementing pointer by one element; :code:`p - 1`.
  See :cpp:func:`pointer::operator--()`.


.. _cfp_iterators:

Iterators
---------

|cfp| random-access iterators wrap the C++ :ref:`iterator <iterators>` classes.
All iterators are :ref:`passed by value <cfp_rpi_value_semantics>` and
are themselves not modified by these functions. Iterators are constructed 
similar to C++ iterators via :c:func:`cfp.array.begin` and 
:c:func:`cfp.array.end`. Iterator usage maps closely to equivalent C++ 
iterator syntax. For example, to set an array to all ones::

  // _ and _iter are namespace aliases
  const cfp_array3d_api _ = cfp.array3d; 
  const cfp_iter3d_api _iter = _.iterator;

  cfp_array3d a = _.ctor(nx, ny, nz, rate, 0, 0);
  cfp_iter3d it;

  for (it = _.begin(a); _iter.neq(it, _.end(a)); it = _iter.inc(it))
    _iter.set(it, 1.0);

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

----

.. c:type:: cfp_iter1
.. c:type:: cfp_iter2
.. c:type:: cfp_iter3
.. c:type:: cfp_iter4

  Fictitious types denoting iterators over 1D, 2D, 3D, and 4D arrays of any
  scalar type.

----

.. c:type:: cfp_iterf
.. c:type:: cfp_iterd

  Fictitious types denoting iterators over float or double arrays of any
  dimensionality.

----

.. c:type:: cfp_iter

  Fictitious type denoting iterator over array of any dimensionality and
  scalar type.

----

.. c:function:: float cfp.arrayf.iterator.get(const cfp_iterf self)
.. c:function:: double cfp.arrayd.iterator.get(const cfp_iterd self)

  Return element referenced by iterator; :code:`*self`.
  See :cpp:func:`iterator::operator*()`.

----

.. c:function:: float cfp.array1f.iterator.get_at(const cfp_iter1f self, ptrdiff_t d)
.. c:function:: double cfp.array1d.iterator.get_at(const cfp_iter1d self, ptrdiff_t d)

  Return element *d* elements (may be negative) from iterator; :code:`self[d]`.
  See :cpp:func:`iterator::operator[]()`.

----

.. c:function:: void cfp.arrayf.iterator.set(cfp_iterf self, float val)
.. c:function:: void cfp.arrayd.iterator.set(cfp_iterd self, double val)

  Update element referenced by iterator; :code:`*self = val`.
  See :cpp:func:`iterator::operator*()`.

----

.. c:function:: void cfp.array1f.iterator.set_at(cfp_iter1 self, ptrdiff_t d, float val)
.. c:function:: void cfp.array1d.iterator.set_at(cfp_iter1 self, ptrdiff_t d, double val)

  Update element *d* elements (may be negative) from iterator;
  :code:`self[d] = val`.
  See :cpp:func:`iterator::operator[]()`.

----

.. c:function:: cfp_ref cfp.array.iterator.ref(cfp_iter self)

  Return reference to element referenced by iterator; :code:`*self`.
  See :cpp:func:`iterator::operator*()`.

----

.. c:function:: cfp_ref cfp.array.iterator.ref_at(cfp_iter self, ptrdiff_t d)

  Return reference to an element offset *d* elements (may be negative) from
  iterator; :code:`self[d]`.
  See :cpp:func:`iterator::operator[]()`.

----

.. c:function:: cfp_ptr cfp.array.iterator.ptr(cfp_iter self)

  Return pointer to element referenced by iterator;
  :code:`&*self`.

----

.. c:function:: cfp_ptr cfp.array.iterator.ptr_at(cfp_iter self, ptrdiff_t d)

  Return pointer to element offset *d* elements (may be negative) from 
  iterator; :code:`&self[d]`.

----

.. c:function:: size_t cfp.array.iterator.i(const cfp_iter self)
.. c:function:: size_t cfp.array.iterator.j(const cfp_iter self)
.. c:function:: size_t cfp.array.iterator.k(const cfp_iter self)
.. c:function:: size_t cfp.array.iterator.l(const cfp_iter self)

  Return *i*, *j*, *k*, and *l* component of array element referenced by
  iterator; see :cpp:func:`iterator::i()`, :cpp:func:`iterator::j()`,
  :cpp:func:`iterator::k()`, and :cpp:func:`iterator::l()`.

----

.. c:function:: zfp_bool cfp.array.iterator.lt(const cfp_iter lhs, const cfp_iter rhs)
.. c:function:: zfp_bool cfp.array.iterator.gt(const cfp_iter lhs, const cfp_iter rhs)
.. c:function:: zfp_bool cfp.array.iterator.leq(const cfp_iter lhs, const cfp_iter rhs)
.. c:function:: zfp_bool cfp.array.iterator.geq(const cfp_iter lhs, const cfp_iter rhs)

  Return true if the two iterators satisfy the given
  :ref:`relationship <iter_inequalities>`;
  :code:`lhs < rhs`, :code:`lhs > rhs`, :code:`lhs <= rhs`, :code:`lhs >= rhs`.

----

.. c:function:: zfp_bool cfp.array.iterator.eq(const cfp_iter lhs, const cfp_iter rhs)

  Return whether two iterators are equal; :code:`lhs == rhs`.
  See :cpp:func:`iterator::operator==()`.

----

.. c:function:: zfp_bool cfp.array.iterator.neq(const cfp_iter lhs, const cfp_iter rhs)

  Return whether two iterators are not equal; :code:`lhs != rhs`.
  See :cpp:func:`iterator::operator!=()`.

----

.. c:function:: ptrdiff_t cfp.array.iterator.distance(const cfp_iter first, const cfp_iter last)

  Return the difference between two iterators; :code:`last - first`.
  See :cpp:func:`iterator::operator-()`.

----

.. c:function:: cfp_iter cfp.array.iterator.next(const cfp_iter it, ptrdiff_t d)

  Return the result of advancing iterator by *d* elements; :code:`it + d`.
  See :cpp:func:`iterator::operator+()`.

----

.. c:function:: cfp_iter cfp.array.iterator.prev(const cfp_iter it, ptrdiff_t d)

  Return the result of decrementing iterator by *d* elements; :code:`it - d`.
  See :cpp:func:`iterator::operator-()`.

----

.. c:function:: cfp_iter cfp.array.iterator.inc(const cfp_iter it)

  Return the result of incrementing iterator by one element;
  :code:`it + 1`.  See :cpp:func:`iterator::operator++()`.

----

.. c:function:: cfp_iter cfp.array.iterator.dec(const cfp_iter it)

  Return the result of decrementing iterator by one element;
  :code:`it - 1`.  See :cpp:func:`iterator::operator--()`.
