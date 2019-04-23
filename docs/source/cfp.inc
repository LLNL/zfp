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
  cfp_array3d* a = _.ctor(nx, ny, nz, rate, 0, 0);
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
C API requires passing an array *self pointer* in order to manipulate the
array.

As with the :ref:`C++ classes <array_classes>`, array elements can be
accessed via multidimensional array indexing, e.g., :code:`get(array, i, j)`,
and via flat, linear indexing, e.g., :code:`get_flat(array, i + nx * j)`.

.. c:type:: cfp_array1f
.. c:type:: cfp_array2f
.. c:type:: cfp_array3f
.. c:type:: cfp_array1d
.. c:type:: cfp_array2d
.. c:type:: cfp_array3d

  Opaque types for 1D, 2D, and 3D compressed arrays of floats and doubles.

.. c:function:: cfp_array1f* cfp_array1f.ctor(uint nx, double rate, const float* p, size_t csize)
.. c:function:: cfp_array2d* cfp_array1d.ctor(uint nx, double rate, const double* p, size_t csize)
.. c:function:: cfp_array2f* cfp_array2f.ctor(uint nx, uint ny, double rate, const float* p, size_t csize)
.. c:function:: cfp_array2d* cfp_array2d.ctor(uint nx, uint ny, double rate, const double* p, size_t csize)
.. c:function:: cfp_array3f* cfp_array3f.ctor(uint nx, uint ny, uint nz, double rate, const float* p, size_t csize)
.. c:function:: cfp_array3d* cfp_array3d.ctor(uint nx, uint ny, uint nz, double rate, const double* p, size_t csize)

  :ref:`Array constructors <array_ctor>`.
  If *p* is not :code:`NULL`, then the array is initialized from uncompressed
  storage; otherwise the array is zero initialized.  *csize* is the minimum
  size cache (in bytes) to use.  If *csize* is zero, a default size is chosen.

.. c:function:: float cfp_array1f.get(const cfp_array1f* a, uint i)
.. c:function:: float cfp_array2f.get(const cfp_array2f* a, uint i, uint j)
.. c:function:: float cfp_array3f.get(const cfp_array3f* a, uint i, uint j, uint k)
.. c:function:: double cfp_array1d.get(const cfp_array1d* a, uint i)
.. c:function:: double cfp_array2d.get(const cfp_array2d* a, uint i, uint j)
.. c:function:: double cfp_array3d.get(const cfp_array3d* a, uint i, uint j, uint k)

  :ref:`Array accessors <array_accessor>` via multidimensional indexing.

.. c:function:: void cfp_array1f.set(const cfp_array1f* a, uint i, float val)
.. c:function:: void cfp_array2f.set(const cfp_array2f* a, uint i, uint j, float val)
.. c:function:: void cfp_array3f.set(const cfp_array3f* a, uint i, uint j, uint k, float val)
.. c:function:: void cfp_array1d.set(const cfp_array1d* a, uint i, double val)
.. c:function:: void cfp_array2d.set(const cfp_array2d* a, uint i, uint j, double val)
.. c:function:: void cfp_array3d.set(const cfp_array3d* a, uint i, uint j, uint k, double val)

  Array mutators for assigning values to array elements via multidimensional
  indexing.

.. c:function:: void cfp_array1f.get_array(const cfp_array1f* self, float* p)
.. c:function:: void cfp_array1d.get_array(const cfp_array1d* self, double* p)
.. c:function:: void cfp_array2f.get_array(const cfp_array2f* self, float* p)
.. c:function:: void cfp_array2d.get_array(const cfp_array2d* self, double* p)
.. c:function:: void cfp_array3f.get_array(const cfp_array3f* self, float* p)
.. c:function:: void cfp_array3d.get_array(const cfp_array3d* self, double* p)

  Decompress entire array; see :cpp:func:`array::get`.

.. c:function:: void cfp_array1f.set_array(cfp_array1f* self, const float* p)
.. c:function:: void cfp_array1d.set_array(cfp_array1d* self, const double* p)
.. c:function:: void cfp_array2f.set_array(cfp_array2f* self, const float* p)
.. c:function:: void cfp_array2d.set_array(cfp_array2d* self, const double* p)
.. c:function:: void cfp_array3f.set_array(cfp_array3f* self, const float* p)
.. c:function:: void cfp_array3d.set_array(cfp_array3d* self, const double* p)

  Initialize entire array; see :cpp:func:`array::set`.

.. c:function:: uint cfp_array2f.size_x(const cfp_array2f* self)
.. c:function:: uint cfp_array2f.size_y(const cfp_array2f* self)
.. c:function:: uint cfp_array3f.size_x(const cfp_array3f* self)
.. c:function:: uint cfp_array3f.size_y(const cfp_array3f* self)
.. c:function:: uint cfp_array3f.size_z(const cfp_array3f* self)

.. c:function:: uint cfp_array2d.size_x(const cfp_array2d* self)
.. c:function:: uint cfp_array2d.size_y(const cfp_array2d* self)
.. c:function:: uint cfp_array3d.size_x(const cfp_array3d* self)
.. c:function:: uint cfp_array3d.size_y(const cfp_array3d* self)
.. c:function:: uint cfp_array3d.size_z(const cfp_array3d* self)

  :ref:`Array dimensions <array_dims>`.

.. _cfp_resize:
.. c:function:: void cfp_array1f.resize(cfp_array1f* self, uint n, int clear)
.. c:function:: void cfp_array2f.resize(cfp_array2f* self, uint nx, uint ny, int clear)
.. c:function:: void cfp_array3f.resize(cfp_array3f* self, uint nx, uint ny, uint nz, int clear)
.. c:function:: void cfp_array1d.resize(cfp_array1d* self, uint n, int clear)
.. c:function:: void cfp_array2d.resize(cfp_array2d* self, uint nx, uint ny, int clear)
.. c:function:: void cfp_array3d.resize(cfp_array3d* self, uint nx, uint ny, uint nz, int clear)

  :ref:`Resize array <array_resize>`.

The six array types share many functions that have the same signature.
Below, each instance of :code:`cfp_array` generically refers to one of
those six types.

.. c:function:: cfp_array* cfp_array.ctor_default()
  
  Default constructor.  Allocate an empty array that later can be
  :ref:`resized <cfp_resize>` and whose rate and cache size can be
  set by :c:func:`cfp_array.set_rate` and :c:func:`cfp_array.set_cache_size`.
  Return a pointer to the constructed array.

.. c:function:: cfp_array* cfp_array.ctor_copy(const cfp_array* src)

  :ref:`Copy constructor <array_ctor_default>`.  Return a pointer to
  the constructed array.

.. c:function:: void cfp_array.dtor(cfp_array* self)

  Destructor.  The destructor not only deallocates any compressed data
  owned by the array, but also frees memory for itself, invalidating
  the *self* pointer upon return.  Note that the user must explicitly
  call the destructor to avoid memory leaks.

.. c:function:: void cfp_array.deep_copy(cfp_array* self, const cfp_array* src)

  Perform a deep copy of *src* analogous to the
  :ref:`C++ assignment operator <array_copy>`.

.. c:function:: double cfp_array.rate(const cfp_array* self)

  See :cpp:func:`array::rate`.

.. c:function:: double cfp_array.set_rate(cfp_array* self, double rate)

  See :cpp:func:`array::set_rate`.

.. c:function:: size_t cfp_array.cache_size(const cfp_array* self)

  See :cpp:func:`array::cache_size`.

.. c:function:: void cfp_array.set_cache_size(cfp_array* self, size_t csize)

  See :cpp:func:`array::set_cache_size`.

.. c:function:: void cfp_array.clear_cache(const cfp_array* self)

  See :cpp:func:`array::clear_cache`.

.. c:function:: void cfp_array.flush_cache(const cfp_array* self)

  See :cpp:func:`array::flush_cache`.

.. c:function:: size_t cfp_array.compressed_size(const cfp_array* self)

  See :cpp:func:`array::compressed_size`.

.. c:function:: uchar* cfp_array.compressed_data(const cfp_array* self)

  See :cpp:func:`array::compressed_data`.

.. c:function:: size_t cfp_array.size(const cfp_array* self)

  See :cpp:func:`array::size`.

.. c:function:: float cfp_array.get_flat(const cfp_array* a, uint index)
.. c:function:: double cfp_array.get_flat(const cfp_array* a, uint index)

  Flat index array accessors; see :cpp:func:`array::operator[]`.

.. c:function:: void cfp_array.set_flat(const cfp_array* a, uint index, float val)
.. c:function:: void cfp_array.set_flat(const cfp_array* a, uint index, double val)

  Flat index array mutators; set array element with flat *index* to *val*.
