# Cython Attributes
# ============================
# distutils: language = c++
# cython: language_level = 3

# Cython Dependencies
# ============================
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.string cimport strncpy
from cython.operator cimport dereference as deref

# Define portion of zfp interface needed by zfpy
# ==============================================
cdef extern from "zfparray.h" namespace "zfp":
    cdef cppclass array:
        double rate() const
        double set_rate(double rate)
        size_t compressed_size()
        unsigned char* compressed_data() const

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

    def __cinit__(self, size_t sz, double rate):
        self.shape = [sz]
        self.dtype = "float32"
        self.thisptr = new py_array1[float](self.shape[0], rate)

    def __dealloc__(self):
        del self.thisptr

    #TODO: if isinstance(subscript, slice):
    # see: https://stackoverflow.com/questions/2936863/implementing-slicing-in-getitem
    #TODO: negative index
    #TODO: multi-dimensional bracket operators (e.g. array2d[i][j] = val or arr2d[i] = [val, val, ...])
    def __getitem__(self, long long i):
        if i >= self.shape[0]:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        cdef float val = deref(self.thisptr).get(i)
        return val

    def __setitem__(self, long long i, float value):
        if i >= self.shape[0]:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        deref(self.thisptr).set(i, value)

    #TODO: doing this requires the constructor below takes in a compressed array so we can reconstruct/pickle
    #       this is essentially our serialize/deserialize methods but in a pythonic format
    #def __reduce__(self):
    #    return (self.__class__, (self.shape[0], self.rate()))
    #TODO: once this is done we will also want explicit dump/dumps calls that the user can use. this maps to how ndarrays work

    def compressed_size(self):
        return deref(<array*>self.thisptr).compressed_size()

    def compressed_data(self):
        cdef size_t sz = self.compressed_size()
        cdef unsigned char* buff = <unsigned char*>malloc(sz)
        try:
            strncpy(<char*>buff, <char*>deref(<array*>self.thisptr).compressed_data(), sz)
        except:
            free(buff)
        return buff

    def rate(self):
        return deref(<array*>self.thisptr).rate()

    def set_rate(self, double rate):
        return deref(<array*>self.thisptr).set_rate(rate)


cdef class zfparray1d:
    cdef py_array1[double] *thisptr
    cdef readonly str dtype
    cdef readonly list shape

    def __cinit__(self, size_t sz, double rate):
        self.shape = [sz]
        self.dtype = "float64"
        self.thisptr = new py_array1[double](self.shape[0], rate)

    def __dealloc__(self):
        del self.thisptr

    def __getitem__(self, long long i):
        if i >= self.shape[0]:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        cdef double val = deref(self.thisptr).get(i)
        return val

    def __setitem__(self, long long i, double value):
        if i >= self.shape[0]:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        deref(self.thisptr).set(i, value)

    #TODO: doing this requires the constructor below takes in a compressed array so we can reconstruct/pickle
    #       this is essentially our serialize/deserialize methods but in a pythonic format
    #def __reduce__(self):
    #    return (self.__class__, (self.shape[0], self.rate()))
    #TODO: once this is done we will also want explicit dump/dumps calls that the user can use. this maps to how ndarrays work

    def compressed_size(self):
        return deref(<array*>self.thisptr).compressed_size()

    def compressed_data(self):
        cdef size_t sz = self.compressed_size()
        cdef unsigned char* buff = <unsigned char*>malloc(sz)
        try:
            strncpy(<char*>buff, <char*>deref(<array*>self.thisptr).compressed_data(), sz)
        except:
            free(buff)
        return buff

    def rate(self):
        return deref(<array*>self.thisptr).rate()

    def set_rate(self, double rate):
        return deref(<array*>self.thisptr).set_rate(rate)
