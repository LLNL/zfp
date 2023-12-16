.. note::
  In multidimensional arrays, the order in which dimensions are specified
  is important.  In |zfp|, the memory layout convention is such that *x*
  varies faster than *y*, which varies faster than *z*, and hence *x* should
  map to the innermost (rightmost) array dimension in a C array and to the
  leftmost dimension in a Fortran array.  Getting the order of dimensions
  right is crucial for good compression and accuracy.  See the discussion of
  :ref:`dimensions and strides <indexing>` and FAQ :ref:`#0 <q-layout>` for
  further information.
