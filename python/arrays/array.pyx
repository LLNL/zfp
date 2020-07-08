# Cython Attributes
# ============================
# distutils: language = c++
# cython: language_level = 3

# Cython Dependencies
# ============================
from libcpp cimport bool
from cython.operator cimport dereference as deref

# Define portion of zfp interface needed by zfpy
# ==============================================
cdef extern from "zfparray1.h" namespace "zfp":
    cdef cppclass array1[Scalar]:
        array1()
        array1(unsigned int nx, double rate, const Scalar* p = 0, size_t csize = 0)

cdef extern from "zfpyarray1.h" namespace "zfp":
    cdef cppclass py_array1[Scalar]:
        py_array1(unsigned int nx, double rate, const Scalar* p = 0, size_t csize = 0)
        Scalar get(unsigned int i) const
        void set(unsigned int i, Scalar val)

# zfpy numpy-like array interface
# ===============================
cdef class zfparray1f:
    cdef py_array1[float] *thisptr
    cdef readonly str dtype
    cdef readonly list shape

    def __init__(self, size_t sz, double rate):
        self.shape = [sz]
        self.dtype = "float"
        self.thisptr = new py_array1[float](self.shape[0], rate)

    def __dealloc__(self):
        del self.thisptr

    def __getitem__(self, unsigned int i):
        if i >= self.shape[0]:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        cdef float val = deref(self.thisptr).get(i)
        return val

    def __setitem__(self, unsigned int i, float value):
        if i >= self.shape[0]:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        deref(self.thisptr).set(i, value)
