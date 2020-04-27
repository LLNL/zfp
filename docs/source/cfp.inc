.. index::
   single: cfp
.. _cfp:

C bindings
----------

.. cpp:namespace:: zfp

|zfp| |cfprelease| adds |cfp|: C language bindings for compressed arrays
via wrappers around the C++ classes.

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

Because the underlying C++ objects have no corresponding C representation,
and because C objects are not self aware (they have no implicit :code:`this`
pointer), the C interface interacts with compressed arrays through
*pointers* to opaque types that |cfp| converts to pointers to
the corresponding C++ objects.  As a consequence, |cfp| compressed arrays
must be allocated on the heap and explicitly freed via designated
destructor functions to avoid memory leaks.  The C++ constructors
are mapped to C by allocating objects via C++ :code:`new`.  Moreover, the
C API requires passing an array *self pointer* (wrapped within a cfp array 
struct) in order to manipulate the array.

As with the :ref:`C++ classes <array_classes>`, array elements can be
accessed via multidimensional array indexing, e.g., :code:`get(array, i, j)`,
and via flat, linear indexing, e.g., :code:`get_flat(array, i + nx * j)`.


Arrays
^^^^^^

.. c:type:: cfp_array1f
.. c:type:: cfp_array2f
.. c:type:: cfp_array3f
.. c:type:: cfp_array1d
.. c:type:: cfp_array2d
.. c:type:: cfp_array3d

  Opaque types for 1D, 2D, and 3D compressed arrays of floats and doubles.

.. c:function:: cfp_array1f cfp.array1f.ctor(uint nx, double rate, const float* p, size_t csize)
.. c:function:: cfp_array1d cfp.array1d.ctor(uint nx, double rate, const double* p, size_t csize)
.. c:function:: cfp_array2f cfp.array2f.ctor(uint nx, uint ny, double rate, const float* p, size_t csize)
.. c:function:: cfp_array2d cfp.array2d.ctor(uint nx, uint ny, double rate, const double* p, size_t csize)
.. c:function:: cfp_array3f cfp.array3f.ctor(uint nx, uint ny, uint nz, double rate, const float* p, size_t csize)
.. c:function:: cfp_array3d cfp.array3d.ctor(uint nx, uint ny, uint nz, double rate, const double* p, size_t csize)

  :ref:`Array constructors <array_ctor>`.
  If *p* is not :code:`NULL`, then the array is initialized from uncompressed
  storage; otherwise the array is zero initialized.  *csize* is the minimum
  size cache (in bytes) to use.  If *csize* is zero, a default size is chosen.

.. c:function:: float cfp.array1f.get(const cfp_array1f a, uint i)
.. c:function:: float cfp.array2f.get(const cfp_array2f a, uint i, uint j)
.. c:function:: float cfp.array3f.get(const cfp_array3f a, uint i, uint j, uint k)
.. c:function:: double cfp.array1d.get(const cfp_array1d a, uint i)
.. c:function:: double cfp.array2d.get(const cfp_array2d a, uint i, uint j)
.. c:function:: double cfp.array3d.get(const cfp_array3d a, uint i, uint j, uint k)

  :ref:`Array accessors <array_accessor>` via multidimensional indexing.

.. c:function:: void cfp.array1f.set(const cfp_array1f a, uint i, float val)
.. c:function:: void cfp.array2f.set(const cfp_array2f a, uint i, uint j, float val)
.. c:function:: void cfp.array3f.set(const cfp_array3f a, uint i, uint j, uint k, float val)
.. c:function:: void cfp.array1d.set(const cfp_array1d a, uint i, double val)
.. c:function:: void cfp.array2d.set(const cfp_array2d a, uint i, uint j, double val)
.. c:function:: void cfp.array3d.set(const cfp_array3d a, uint i, uint j, uint k, double val)

  Array mutators for assigning values to array elements via multidimensional
  indexing.

.. c:function:: void cfp.array1f.get_array(const cfp_array1f self, float* p)
.. c:function:: void cfp.array1d.get_array(const cfp_array1d self, double* p)
.. c:function:: void cfp.array2f.get_array(const cfp_array2f self, float* p)
.. c:function:: void cfp.array2d.get_array(const cfp_array2d self, double* p)
.. c:function:: void cfp.array3f.get_array(const cfp_array3f self, float* p)
.. c:function:: void cfp.array3d.get_array(const cfp_array3d self, double* p)

  Decompress entire array; see :cpp:func:`array::get`.

.. c:function:: void cfp.array1f.set_array(cfp_array1f self, const float* p)
.. c:function:: void cfp.array1d.set_array(cfp_array1d self, const double* p)
.. c:function:: void cfp.array2f.set_array(cfp_array2f self, const float* p)
.. c:function:: void cfp.array2d.set_array(cfp_array2d self, const double* p)
.. c:function:: void cfp.array3f.set_array(cfp_array3f self, const float* p)
.. c:function:: void cfp.array3d.set_array(cfp_array3d self, const double* p)

  Initialize entire array; see :cpp:func:`array::set`.

.. c:function:: uint cfp.array2f.size_x(const cfp_array2f self)
.. c:function:: uint cfp.array2f.size_y(const cfp_array2f self)
.. c:function:: uint cfp.array3f.size_x(const cfp_array3f self)
.. c:function:: uint cfp.array3f.size_y(const cfp_array3f self)
.. c:function:: uint cfp.array3f.size_z(const cfp_array3f self)

.. c:function:: uint cfp.array2d.size_x(const cfp_array2d self)
.. c:function:: uint cfp.array2d.size_y(const cfp_array2d self)
.. c:function:: uint cfp.array3d.size_x(const cfp_array3d self)
.. c:function:: uint cfp.array3d.size_y(const cfp_array3d self)
.. c:function:: uint cfp.array3d.size_z(const cfp_array3d self)

  :ref:`Array dimensions <array_dims>`.

.. _cfp_resize:
.. c:function:: void cfp.array1f.resize(cfp_array1f self, uint n, int clear)
.. c:function:: void cfp.array2f.resize(cfp_array2f self, uint nx, uint ny, int clear)
.. c:function:: void cfp.array3f.resize(cfp_array3f self, uint nx, uint ny, uint nz, int clear)
.. c:function:: void cfp.array1d.resize(cfp_array1d self, uint n, int clear)
.. c:function:: void cfp.array2d.resize(cfp_array2d self, uint nx, uint ny, int clear)
.. c:function:: void cfp.array3d.resize(cfp_array3d self, uint nx, uint ny, uint nz, int clear)

  :ref:`Resize array <array_resize>`.

The six array types share many functions that have the same signature.
Below, each instance of :code:`cfp_array` generically refers to one of
those six types (ex. :code:`cfp_array1f`). Similarly, each instance of
:code:`cfp.array` refers to navigating the CFP API namespace for a given cfp
type, to reach the function call (ex. :code:`cfp.array2f`).

.. c:function:: cfp_array cfp.array.ctor_default()
  
  Default constructor.  Allocate an empty array that later can be
  :ref:`resized <cfp_resize>` and whose rate and cache size can be
  set by :c:func:`cfp.array.set_rate` and :c:func:`cfp.array.set_cache_size`.

.. c:function:: cfp_array cfp.array.ctor_copy(const cfp_array src)

  :ref:`Copy constructor <array_ctor_default>`.

.. c:function:: void cfp.array.dtor(cfp_array self)

  Destructor.  The destructor not only deallocates any compressed data
  owned by the array, but also frees memory for itself, invalidating
  the *self* pointer upon return.  Note that the user must explicitly
  call the destructor to avoid memory leaks.

.. c:function:: void cfp.array.deep_copy(cfp_array self, const cfp_array* src)

  Perform a deep copy of *src* analogous to the
  :ref:`C++ assignment operator <array_copy>`.

.. c:function:: double cfp.array.rate(const cfp_array self)

  See :cpp:func:`array::rate`.

.. c:function:: double cfp.array.set_rate(cfp_array self, double rate)

  See :cpp:func:`array::set_rate`.

.. c:function:: size_t cfp.array.cache_size(const cfp_array self)

  See :cpp:func:`array::cache_size`.

.. c:function:: void cfp.array.set_cache_size(cfp_array self, size_t csize)

  See :cpp:func:`array::set_cache_size`.

.. c:function:: void cfp.array.clear_cache(const cfp_array self)

  See :cpp:func:`array::clear_cache`.

.. c:function:: void cfp.array.flush_cache(const cfp_array self)

  See :cpp:func:`array::flush_cache`.

.. c:function:: size_t cfp.array.compressed_size(const cfp_array self)

  See :cpp:func:`array::compressed_size`.

.. c:function:: uchar* cfp.array.compressed_data(const cfp_array self)

  See :cpp:func:`array::compressed_data`.

.. c:function:: size_t cfp.array.size(const cfp_array self)

  See :cpp:func:`array::size`.

.. c:function:: float cfp.array.get_flat(const cfp_array a, uint index)
.. c:function:: double cfp.array.get_flat(const cfp_array a, uint index)

  Flat index array accessors; see :cpp:func:`array::operator[]`.

.. c:function:: void cfp.array.set_flat(const cfp_array a, uint index, float val)
.. c:function:: void cfp.array.set_flat(const cfp_array a, uint index, double val)

  Flat index array mutators; set array element with flat *index* to *val*.



References
^^^^^^^^^^
.. c:type:: cfp_ref1f
.. c:type:: cfp_ref2f
.. c:type:: cfp_ref3f
.. c:type:: cfp_ref1d
.. c:type:: cfp_ref2d
.. c:type:: cfp_ref3d

  Opaque types for references to 1D, 2D, and 3D compressed float or double array elements.

.. c:function:: cfp_ref1f cfp.array1f.ref(cfp_array1f self, uint i)
.. c:function:: cfp_ref1d cfp.array1d.ref(cfp_array1d self, uint i)
.. c:function:: cfp_ref2f cfp.array2f.ref(cfp_array2f self, uint i, uint j)
.. c:function:: cfp_ref2d cfp.array2d.ref(cfp_array2d self, uint i, uint j)
.. c:function:: cfp_ref3f cfp.array3f.ref(cfp_array3f self, uint i, uint j, uint k)
.. c:function:: cfp_ref3d cfp.array3d.ref(cfp_array3d self, uint i, uint j, uint k)

  Reference :ref:`constructor <lvref_bidx>` via multidimensional indexing.

.. c:function:: cfp_ref1f cfp.array1f.flat_ref(cfp_array1f self, uint i)
.. c:function:: cfp_ref1d cfp.array1d.flat_ref(cfp_array1d self, uint i)
.. c:function:: cfp_ref2f cfp.array2f.flat_ref(cfp_array2f self, uint i)
.. c:function:: cfp_ref2d cfp.array2d.flat_ref(cfp_array2d self, uint i)
.. c:function:: cfp_ref3f cfp.array3f.flat_ref(cfp_array3f self, uint i)
.. c:function:: cfp_ref3d cfp.array3d.flat_ref(cfp_array3d self, uint i)

  Reference :ref:`constructor <lvref_idx>` via flat indexing.

.. c:function:: float  cfp.array1f.reference.get(cfp_ref1f self)
.. c:function:: double cfp.array1d.reference.get(cfp_ref1d self)
.. c:function:: float  cfp.array2f.reference.get(cfp_ref2f self)
.. c:function:: double cfp.array2d.reference.get(cfp_ref2d self)
.. c:function:: float  cfp.array3f.reference.get(cfp_ref3f self)
.. c:function:: double cfp.array3d.reference.get(cfp_ref3d self)

  Retrieve value from reference.

.. c:function:: void cfp.array1f.reference.set(cfp_ref1f self, float  val)
.. c:function:: void cfp.array1d.reference.set(cfp_ref1d self, double val)
.. c:function:: void cfp.array2f.reference.set(cfp_ref2f self, float  val)
.. c:function:: void cfp.array2d.reference.set(cfp_ref2d self, double val)
.. c:function:: void cfp.array3f.reference.set(cfp_ref3f self, float  val)
.. c:function:: void cfp.array3d.reference.set(cfp_ref3d self, double val)

  Update value pointed to by reference; see :cpp:func:`reference reference::operator=(Scalar val)`.

.. c:function:: void cfp.array1f.reference.copy(cfp_ref1f self, cfp_ref1f src)
.. c:function:: void cfp.array1d.reference.copy(cfp_ref1d self, cfp_ref1d src)
.. c:function:: void cfp.array2f.reference.copy(cfp_ref2f self, cfp_ref2f src)
.. c:function:: void cfp.array2d.reference.copy(cfp_ref2d self, cfp_ref2d src)
.. c:function:: void cfp.array3f.reference.copy(cfp_ref3f self, cfp_ref3f src)
.. c:function:: void cfp.array3d.reference.copy(cfp_ref3d self, cfp_ref3d src)

  Copy value pointed to by source reference to value of self reference; see :cpp:func:`reference reference::operator=(const reference &ref)`.

.. c:function:: cfp_ptr1f cfp.array1f.reference.ptr(cfp_ref1f self)
.. c:function:: cfp_ptr1d cfp.array1d.reference.ptr(cfp_ref1d self)
.. c:function:: cfp_ptr2f cfp.array2f.reference.ptr(cfp_ref2f self)
.. c:function:: cfp_ptr2d cfp.array2d.reference.ptr(cfp_ref2d self)
.. c:function:: cfp_ptr3f cfp.array3f.reference.ptr(cfp_ref3f self)
.. c:function:: cfp_ptr3d cfp.array3d.reference.ptr(cfp_ref3d self)

  Obtain proxy pointer to reference value; see :cpp:func:`pointer reference::operator&()`.



Pointers
^^^^^^^^
.. c:type:: cfp_ptr1f
.. c:type:: cfp_ptr2f
.. c:type:: cfp_ptr3f
.. c:type:: cfp_ptr1d
.. c:type:: cfp_ptr2d
.. c:type:: cfp_ptr3d

  Opaque types for pointers to 1D, 2D, and 3D compressed float or double array elements.

.. c:function:: cfp_ref1f (*ref)(cfp_ptr1f self)
.. c:function:: cfp_ref1d (*ref)(cfp_ptr1d self)
.. c:function:: cfp_ref2f (*ref)(cfp_ptr2f self)
.. c:function:: cfp_ref2d (*ref)(cfp_ptr2d self)
.. c:function:: cfp_ref3f (*ref)(cfp_ptr3f self)
.. c:function:: cfp_ref3d (*ref)(cfp_ptr3d self)

  Get proxy reference associated with proxy pointer; see :cpp:func:`reference pointer::operator*() const`.

.. c:function:: cfp_ref1f (*offset_ref)(cfp_ptr1f self, int i)
.. c:function:: cfp_ref1d (*offset_ref)(cfp_ptr1d self, int i)
.. c:function:: cfp_ref2f (*offset_ref)(cfp_ptr2f self, int i)
.. c:function:: cfp_ref2d (*offset_ref)(cfp_ptr2d self, int i)
.. c:function:: cfp_ref3f (*offset_ref)(cfp_ptr3f self, int i)
.. c:function:: cfp_ref3d (*offset_ref)(cfp_ptr3d self, int i)

  Get proxy reference associated with an offset from proxy pointer; see :cpp:func:`reference pointer::operator[](ptrdiff_t d) const`.

.. c:function:: int (*eq)(cfp_ptr1f self, cfp_ptr1f src)
.. c:function:: int (*eq)(cfp_ptr1d self, cfp_ptr1d src)
.. c:function:: int (*eq)(cfp_ptr2f self, cfp_ptr2f src)
.. c:function:: int (*eq)(cfp_ptr2d self, cfp_ptr2d src)
.. c:function:: int (*eq)(cfp_ptr3f self, cfp_ptr3f src)
.. c:function:: int (*eq)(cfp_ptr3d self, cfp_ptr3d src)

  Return whether the addresses of two proxy pointers are equal; see :cpp:func:`pointer pointer::operator==(int)` and :cpp:func:`pointer pointer::operator!=(int)`.

.. c:function:: int (*diff)(cfp_ptr1f self, cfp_ptr1f src)
.. c:function:: int (*diff)(cfp_ptr1d self, cfp_ptr1d src)
.. c:function:: int (*diff)(cfp_ptr2f self, cfp_ptr2f src)
.. c:function:: int (*diff)(cfp_ptr2d self, cfp_ptr2d src)
.. c:function:: int (*diff)(cfp_ptr3f self, cfp_ptr3f src)
.. c:function:: int (*diff)(cfp_ptr3d self, cfp_ptr3d src)

  Get the difference between two proxy pointer addresses; see :cpp:func:`ptrdiff_t pointer::operator-(const pointer &q) const`.

.. c:function:: cfp_ptr1f (*shift)(cfp_ptr1f self, int i)
.. c:function:: cfp_ptr1d (*shift)(cfp_ptr1d self, int i)
.. c:function:: cfp_ptr2f (*shift)(cfp_ptr2f self, int i)
.. c:function:: cfp_ptr2d (*shift)(cfp_ptr2d self, int i)
.. c:function:: cfp_ptr3f (*shift)(cfp_ptr3f self, int i)
.. c:function:: cfp_ptr3d (*shift)(cfp_ptr3d self, int i)

  Shift a proxy pointer address by a given offset; see :cpp:func:`pointer pointer::operator+(ptrdiff_t d) const` and :cpp:func:`pointer::operator-(ptrdiff_t d) const`.

.. c:function:: cfp_ptr1f (*inc)(cfp_ptr1f self)
.. c:function:: cfp_ptr1d (*inc)(cfp_ptr1d self)
.. c:function:: cfp_ptr2f (*inc)(cfp_ptr2f self)
.. c:function:: cfp_ptr2d (*inc)(cfp_ptr2d self)
.. c:function:: cfp_ptr3f (*inc)(cfp_ptr3f self)
.. c:function:: cfp_ptr3d (*inc)(cfp_ptr3d self)

  Return proxy pointer incremented by one; see :cpp:func:`pointer pointer::operator++(int)`.

.. c:function:: cfp_ptr1f (*dec)(cfp_ptr1f self)
.. c:function:: cfp_ptr1d (*dec)(cfp_ptr1d self)
.. c:function:: cfp_ptr2f (*dec)(cfp_ptr2f self)
.. c:function:: cfp_ptr2d (*dec)(cfp_ptr2d self)
.. c:function:: cfp_ptr3f (*dec)(cfp_ptr3f self)
.. c:function:: cfp_ptr3d (*dec)(cfp_ptr3d self)

  Return proxy pointer decremented by one; see :cpp:func:`pointer pointer::operator--(int)`.



Iterators
^^^^^^^^^
.. c:type:: cfp_iter1f
.. c:type:: cfp_iter2f
.. c:type:: cfp_iter3f
.. c:type:: cfp_iter1d
.. c:type:: cfp_iter2d
.. c:type:: cfp_iter3d

  Opaque types for block iterators over 1D, 2D, and 3D compressed float or double array elements.

.. c:function:: cfp_iter1f (*dec)(cfp_iter1f self)
.. c:function:: cfp_iter1d (*dec)(cfp_iter1d self)

  Decrement proxy iterator by one (only supported for one dimensional array iterators); see :cpp:func:`iterator iterator::operator--(int)`.

.. c:function:: cfp_iter1f (*shift)(cfp_iter1f self, int i)
.. c:function:: cfp_iter1d (*shift)(cfp_iter1d self, int i)

  Shift proxy iterator by an offset (only supported for one dimensional array iterators); see :cpp:func:`array1::iterator array1::iterator::operator+=(difference_type d)`.

.. c:function:: int (*diff)(cfp_iter1f self, cfp_iter1f src)
.. c:function:: int (*diff)(cfp_iter1d self, cfp_iter1d src)

  Get the difference between two iterators (only supported for one dimensional array iterators); see :cpp:func:`difference_type array1::iterator::operator-(const array1::iterator &it) const`.

.. c:function:: int (*lt)(cfp_iter1f self, cfp_iter1f src)
.. c:function:: int (*lt)(cfp_iter1d self, cfp_iter1d src)
.. c:function:: int (*gt)(cfp_iter1f self, cfp_iter1f src)
.. c:function:: int (*gt)(cfp_iter1d self, cfp_iter1d src)
.. c:function:: int (*leq)(cfp_iter1f self, cfp_iter1f src)
.. c:function:: int (*leq)(cfp_iter1d self, cfp_iter1d src)
.. c:function:: int (*geq)(cfp_iter1f self, cfp_iter1f src)
.. c:function:: int (*geq)(cfp_iter1d self, cfp_iter1d src)

  Return true if the two iterators satisfy the given relationship (only supported for one dimensional array iterators). 

.. c:function:: cfp_ref1f (*offset_ref)(cfp_iter1f self, int i)
.. c:function:: cfp_ref1d (*offset_ref)(cfp_iter1d self, int i)

  Return reference associated with an offset from iterator (only supported for one dimensional array iterators); see :cpp:func:`array1::reference array1::iterator::operator[](difference_type d) const`.


.. c:function:: cfp_ref1f (*ref)(cfp_iter1f self)
.. c:function:: cfp_ref1d (*ref)(cfp_iter1d self)
.. c:function:: cfp_ref2f (*ref)(cfp_iter2f self)
.. c:function:: cfp_ref2d (*ref)(cfp_iter2d self)
.. c:function:: cfp_ref3f (*ref)(cfp_iter3f self)
.. c:function:: cfp_ref3d (*ref)(cfp_iter3d self)

  Return reference associated with iterator; see :cpp:func:`reference iterator::operator*() const`.

.. c:function:: cfp_iter1f (*inc)(cfp_iter1f self)
.. c:function:: cfp_iter1d (*inc)(cfp_iter1d self)
.. c:function:: cfp_iter2f (*inc)(cfp_iter2f self)
.. c:function:: cfp_iter2d (*inc)(cfp_iter2d self)
.. c:function:: cfp_iter3f (*inc)(cfp_iter3f self)
.. c:function:: cfp_iter3d (*inc)(cfp_iter3d self)

  Increment iterator by one; see :cpp:func`iterator iterator::operator++(int)`.

.. c:function:: int (*eq)(cfp_iter1f self, cfp_iter1f src)
.. c:function:: int (*eq)(cfp_iter1d self, cfp_iter1d src)
.. c:function:: int (*eq)(cfp_iter2f self, cfp_iter2f src)
.. c:function:: int (*eq)(cfp_iter2d self, cfp_iter2d src)
.. c:function:: int (*eq)(cfp_iter3f self, cfp_iter3f src)
.. c:function:: int (*eq)(cfp_iter3d self, cfp_iter3d src)

  Return whether two iterators point to the same location; see :cpp:func:`bool iterator::operator==(const iterator &it) const` and :cpp:func:`bool iterator::operator!=(const iterator &it) const`.

.. c:function:: uint (*i)(cfp_iter1f self)
.. c:function:: uint (*i)(cfp_iter1d self)
.. c:function:: uint (*i)(cfp_iter2f self)
.. c:function:: uint (*i)(cfp_iter2d self)
.. c:function:: uint (*i)(cfp_iter3f self)
.. c:function:: uint (*i)(cfp_iter3d self)

  Get i component of iterator index; see :cpp:func:`uint iterator::i() const`.

.. c:function:: uint (*j)(cfp_iter2f self)
.. c:function:: uint (*j)(cfp_iter2d self)
.. c:function:: uint (*j)(cfp_iter3f self)
.. c:function:: uint (*j)(cfp_iter3d self)

  Get j component of iterator index; see :cpp:func:`uint iterator::j() const`.

.. c:function:: uint (*k)(cfp_iter3f self)
.. c:function:: uint (*k)(cfp_iter3d self)

  Get k component of iterator index; see :cpp:func:`uint iterator::k() const`.
