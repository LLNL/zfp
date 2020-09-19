# Cython Attributes
# ============================
# distutils: language = c++
# cython: language_level = 3

# Cython Dependencies
# ============================
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy 
from cython.operator cimport dereference as deref

#TODO: split this up into pxd and pyx files in case someone using cython wants to in turn use our cython module
#       also look at cython inlining where possible
# see: https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html

# Define portion of zfp interface needed by zfpy
# ==============================================

cdef extern from "zfparray.h" namespace "zfp::array":
    cdef cppclass header:
        header()
        size_t size()
        unsigned char* buffer

cdef extern from "zfparray.h" namespace "zfp":
    cdef cppclass array:
        double rate() const
        double set_rate(double rate)
        size_t compressed_size()
        unsigned char* compressed_data() const
        header get_header() const

cdef extern from "zfparray1.h" namespace "zfp":
    cdef cppclass array1[Scalar]:
        array1()
        array1(unsigned int nx, double rate, const Scalar* p = 0, size_t csize = 0)

cdef extern from "zfpyarray1.h" namespace "zfp":
    cdef cppclass py_array1[Scalar]:
        py_array1(unsigned int nx, double rate)
        py_array1(unsigned int nx, double rate, const Scalar* p)
        py_array1(unsigned int nx, double rate, const Scalar* p, size_t csize)
        Scalar get(unsigned int i) const
        void set(unsigned int i, Scalar val)

cdef extern from "zfparray2.h" namespace "zfp":
    cdef cppclass array2[Scalar]:
        array2()
        array2(unsigned int nx, unsigned int ny, double rate, const Scalar* p = 0, size_t csize = 0)

cdef extern from "zfpyarray2.h" namespace "zfp":
    cdef cppclass py_array2[Scalar]:
        py_array2(unsigned int nx, unsigned int ny, double rate)
        py_array2(unsigned int nx, unsigned int ny, double rate, const Scalar*)
        py_array2(unsigned int nx, unsigned int ny, double rate, const Scalar*, size_t csize)
        Scalar get(unsigned int i) const
        Scalar get(unsigned int i, unsigned int j) const
        void set(unsigned int i, Scalar val)
        void set(unsigned int i, unsigned int j, Scalar val)

cdef extern from "zfparray3.h" namespace "zfp":
    cdef cppclass array3[Scalar]:
        array3()
        array3(unsigned int nx, unsigned int ny, unsigned int nz, double rate, const Scalar* p = 0, size_t csize = 0)

cdef extern from "zfpyarray3.h" namespace "zfp":
    cdef cppclass py_array3[Scalar]:
        py_array3(unsigned int nx, unsigned int ny, unsigned int nz, double rate)
        py_array3(unsigned int nx, unsigned int ny, unsigned int nz, double rate, const Scalar*)
        py_array3(unsigned int nx, unsigned int ny, unsigned int nz, double rate, const Scalar*, size_t csize)
        Scalar get(unsigned int i) const
        Scalar get(unsigned int i, unsigned int j, unsigned int k) const
        void set(unsigned int i, Scalar val)
        void set(unsigned int i, unsigned int j, unsigned int k, Scalar val)



# zfpy algorithm interface
# zfpy equivalents to numpy interface
# ===================================
cdef class zfparray:
    def __cinit__(self):
        return

# TODO: numpy like array iterator (good place for block iteration)
# TODO: make sure destructors are getting called properly


# zfpy header interface
# TODO
# =====================
cdef class zfpheader:
    cdef bytearray buff
    cdef int sz

    def __cinit__(self, unsigned char* b, size_t sz):
        cdef buff = bytearray(sz)
        self.sz = sz
        for i in range(sz):
            buff[i] = b[i]

    def size(self):
        return self.sz

    def buffer(self):
        return self.buff

# zfpy numpy-like array interface
# zfpy equivalents to numpy ndarray interface
# ===========================================
cdef class zfparray1f:
    cdef py_array1[float] *thisptr
    cdef readonly str dtype
    cdef readonly list shape

    def __cinit__(self, size_t sz, double rate, size_t csize = 0):
        self.shape = [sz]
        self.dtype = "float32"

        # note: cython needs some help to figure out which overload we want here (hence the explicit casts)
        self.thisptr = new py_array1[float](<unsigned int>self.shape[0], <double>rate, <float*>0, <size_t>csize)

    def __dealloc__(self):
        del self.thisptr

    #TODO: if isinstance(subscript, slice):
    # see: https://stackoverflow.com/questions/2936863/implementing-slicing-in-getitem
    #TODO: negative index
    #TODO: multi-dimensional bracket operators (e.g. array2d[i][j] = val or arr2d[i] = [val, val, ...])
    def __getitem__(self, long long i):
        if i >= self.shape[0] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        cdef float val = deref(self.thisptr).get(i)
        return val

    def __setitem__(self, long long i, float value):
        if i >= self.shape[0] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        deref(self.thisptr).set(i, value)

    def get(self, long long i):
        if i >= self.shape[0] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        cdef float val = deref(self.thisptr).get(i)
        return val

    def set(self, long long i, float value):
        if i >= self.shape[0] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        deref(self.thisptr).set(i, value)

    #TODO: doing this requires the constructor below takes in a compressed array so we can reconstruct/pickle
    #       this is essentially our serialize/deserialize methods but in a pythonic format
    #def __reduce__(self):
    #    return (self.__class__, (self.shape[0], self.rate()))
    #TODO: once this is done we will also want explicit dump/dumps calls that the user can use. this maps to how ndarrays work

    def compressed_size(self):
        return deref(<array*>self.thisptr).compressed_size()

    #TODO: numpy calls this tobytes()
    def compressed_data(self):
        cdef size_t sz = self.compressed_size()
        cdef unsigned char* buff = <unsigned char*>malloc(sz)
        try:
            memcpy(<char*>buff, <char*>deref(<array*>self.thisptr).compressed_data(), sz)
        except:
            free(buff)
        return buff

    def rate(self):
        return deref(<array*>self.thisptr).rate()

    def set_rate(self, double rate):
        return deref(<array*>self.thisptr).set_rate(rate)

    def get_header(self):
        cdef header h = deref(<array*>self.thisptr).get_header()
        return zfpheader(h.buffer, h.size())
        


cdef class zfparray1d:
    cdef py_array1[double] *thisptr
    cdef readonly str dtype
    cdef readonly list shape

    def __cinit__(self, size_t sz, double rate, size_t csize = 0):
        self.shape = [sz]
        self.dtype = "float64"

        # note: cython needs some help to figure out which overload we want here (hence the explicit casts)
        self.thisptr = new py_array1[double](<unsigned int>self.shape[0], <double>rate, <double*>0, <size_t>csize)

    def __dealloc__(self):
        del self.thisptr

    def __getitem__(self, long long i):
        if i >= self.shape[0] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        cdef double val = deref(self.thisptr).get(i)
        return val

    def __setitem__(self, long long i, double value):
        if i >= self.shape[0] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        deref(self.thisptr).set(i, value)

    def get(self, long long i):
        if i >= self.shape[0] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        cdef double val = deref(self.thisptr).get(i)
        return val

    def set(self, long long i, double value):
        if i >= self.shape[0] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0]))
        deref(self.thisptr).set(i, value)

    #TODO: doing this requires the constructor below takes in a compressed array so we can reconstruct/pickle
    #       this is essentially our serialize/deserialize methods but in a pythonic format
    #def __reduce__(self):
    #    return (self.__class__, (self.shape[0], self.rate()))
    #TODO: once this is done we will also want explicit dump/dumps calls that the user can use. this maps to how ndarrays work

    def compressed_size(self):
        return deref(<array*>self.thisptr).compressed_size()

    #TODO: numpy calls this tobytes()
    def compressed_data(self):
        cdef size_t sz = self.compressed_size()
        cdef unsigned char* buff = <unsigned char*>malloc(sz)
        try:
            memcpy(<char*>buff, <char*>deref(<array*>self.thisptr).compressed_data(), sz)
        except:
            free(buff)
        return buff

    def rate(self):
        return deref(<array*>self.thisptr).rate()

    def set_rate(self, double rate):
        return deref(<array*>self.thisptr).set_rate(rate)

    #def get_header(self):
    #    return deref(<array*>self.thisptr).get_header()



cdef class zfparray2f:
    cdef py_array2[float] *thisptr
    cdef readonly str dtype
    cdef readonly list shape

    def __cinit__(self, size_t sz_x, sz_y, double rate, size_t csize = 0):
        self.shape = [sz_x, sz_y]
        self.dtype = "float32"

        # note: cython needs some help to figure out which overload we want here (hence the explicit casts)
        self.thisptr = new py_array2[float](<unsigned int>self.shape[0], <unsigned int>self.shape[1], <double>rate, <float*>0, <size_t>csize)

    def __dealloc__(self):
        del self.thisptr

    #TODO: multi-dimensional and slicing
    def __getitem__(self, int i):
        return self.flat_get(<long long>i)

    #TODO: multi-dimensional and slicing
    def __setitem__(self, int i, float value):
        self.flat_set(<long long>i, value)

    def flat_get(self, long long i):
        if i >= self.shape[0]*self.shape[1] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0] * self.shape[1]))
        cdef float val = deref(self.thisptr).get(i)
        return val

    def get(self, long long i, long long j):
        if i >= self.shape[0] or j >= self.shape[1] or i < 0 or j < 0:
            raise IndexError("index " + str(i) + ", " + str(j) + " is out of bounds for axis 0 with size " + str(self.shape[0]) + ", " + str(self.shape[1]))
        cdef float val = deref(self.thisptr).get(i, j)
        return val

    def flat_set(self, long long i, float value):
        if i >= self.shape[0]*self.shape[1] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0] * self.shape[1]))
        deref(self.thisptr).set(i, value)

    def set(self, long long i, long long j, float value):
        if i >= self.shape[0] or j >= self.shape[1] or i < 0 or j < 0:
            raise IndexError("index " + str(i) + ", " + str(j) + " is out of bounds for axis 0 with size " + str(self.shape[0]) + ", " + str(self.shape[1]))
        deref(self.thisptr).set(i, j, value)

    #TODO: doing this requires the constructor below takes in a compressed array so we can reconstruct/pickle
    #       this is essentially our serialize/deserialize methods but in a pythonic format
    #def __reduce__(self):
    #    return (self.__class__, (self.shape[0], self.rate()))
    #TODO: once this is done we will also want explicit dump/dumps calls that the user can use. this maps to how ndarrays work

    def compressed_size(self):
        return deref(<array*>self.thisptr).compressed_size()

    #TODO: numpy calls this tobytes()
    def compressed_data(self):
        cdef size_t sz = self.compressed_size()
        cdef unsigned char* buff = <unsigned char*>malloc(sz)
        try:
            memcpy(<char*>buff, <char*>deref(<array*>self.thisptr).compressed_data(), sz)
        except:
            free(buff)
        return buff

    def rate(self):
        return deref(<array*>self.thisptr).rate()

    def set_rate(self, double rate):
        return deref(<array*>self.thisptr).set_rate(rate)

    #def get_header(self):
    #    return deref(<array*>self.thisptr).get_header()



cdef class zfparray2d:
    cdef py_array2[double] *thisptr
    cdef readonly str dtype
    cdef readonly list shape

    def __cinit__(self, size_t sz_x, sz_y, double rate, size_t csize = 0):
        self.shape = [sz_x, sz_y]
        self.dtype = "float64"

        # note: cython needs some help to figure out which overload we want here (hence the explicit casts)
        self.thisptr = new py_array2[double](<unsigned int>self.shape[0], <unsigned int>self.shape[1], <double>rate, <double*>0, <size_t>csize)

    def __dealloc__(self):
        del self.thisptr

    #TODO: multi-dimensional and slicing
    def __getitem__(self, int i):
        return self.flat_get(<long long>i)

    #TODO: multi-dimensional and slicing
    def __setitem__(self, int i, double value):
        self.flat_set(<long long>i, value)

    def flat_get(self, long long i):
        if i >= self.shape[0]*self.shape[1] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0] * self.shape[1]))
        cdef double val = deref(self.thisptr).get(i)
        return val

    def get(self, long long i, long long j):
        if i >= self.shape[0] or j >= self.shape[1] or i < 0 or j < 0:
            raise IndexError("index " + str(i) + ", " + str(j) + " is out of bounds for axis 0 with size " + str(self.shape[0]) + ", " + str(self.shape[1]))
        cdef double val = deref(self.thisptr).get(i, j)
        return val

    def flat_set(self, long long i, double value):
        if i >= self.shape[0]*self.shape[1] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0] * self.shape[1]))
        deref(self.thisptr).set(i, value)

    def set(self, long long i, long long j, double value):
        if i >= self.shape[0] or j >= self.shape[1] or i < 0 or j < 0:
            raise IndexError("index " + str(i) + ", " + str(j) + " is out of bounds for axis 0 with size " + str(self.shape[0]) + ", " + str(self.shape[1]))
        deref(self.thisptr).set(i, j, value)

    #TODO: doing this requires the constructor below takes in a compressed array so we can reconstruct/pickle
    #       this is essentially our serialize/deserialize methods but in a pythonic format
    #def __reduce__(self):
    #    return (self.__class__, (self.shape[0], self.rate()))
    #TODO: once this is done we will also want explicit dump/dumps calls that the user can use. this maps to how ndarrays work

    def compressed_size(self):
        return deref(<array*>self.thisptr).compressed_size()

    #TODO: numpy calls this tobytes()
    def compressed_data(self):
        cdef size_t sz = self.compressed_size()
        cdef unsigned char* buff = <unsigned char*>malloc(sz)
        try:
            memcpy(<char*>buff, <char*>deref(<array*>self.thisptr).compressed_data(), sz)
        except:
            free(buff)
        return buff

    def rate(self):
        return deref(<array*>self.thisptr).rate()

    def set_rate(self, double rate):
        return deref(<array*>self.thisptr).set_rate(rate)

    #def get_header(self):
    #    return deref(<array*>self.thisptr).get_header()



cdef class zfparray3f:
    cdef py_array3[float] *thisptr
    cdef readonly str dtype
    cdef readonly list shape

    def __cinit__(self, size_t sz_x, sz_y, sz_z, double rate, size_t csize = 0):
        self.shape = [sz_x, sz_y, sz_z]
        self.dtype = "float32"

        # note: cython needs some help to figure out which overload we want here (hence the explicit casts)
        self.thisptr = new py_array3[float](<unsigned int>self.shape[0], <unsigned int>self.shape[1], <unsigned int>self.shape[2], <double>rate, <float*>0, <size_t>csize)

    def __dealloc__(self):
        del self.thisptr

    #TODO: multi-dimensional and slicing
    def __getitem__(self, int i):
        return self.flat_get(<long long>i)

    #TODO: multi-dimensional and slicing
    def __setitem__(self, int i, float value):
        self.flat_set(<long long>i, value)

    def flat_get(self, long long i):
        if i >= self.shape[0]*self.shape[1]*self.shape[2] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0] * self.shape[1] * self.shape[2]))
        cdef float val = deref(self.thisptr).get(i)
        return val

    def get(self, long long i, long long j, long long k):
        if i >= self.shape[0] or j >= self.shape[1] or k >= self.shape[2] or i < 0 or j < 0 or k < 0:
            raise IndexError("index " + str(i) + ", " + str(j) + ", " + str(k) + " is out of bounds for axis 0 with size " + str(self.shape[0]) + ", " + str(self.shape[1]) + ", " + str(self.shape[2]))
        cdef float val = deref(self.thisptr).get(i, j, k)
        return val

    def flat_set(self, long long i, float value):
        if i >= self.shape[0]*self.shape[1]*self.shape[2] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0] * self.shape[1] * self.shape[2]))
        deref(self.thisptr).set(i, value)

    def set(self, long long i, long long j, long long k, float value):
        if i >= self.shape[0] or j >= self.shape[1] or k >= self.shape[2] or i < 0 or j < 0 or k < 0:
            raise IndexError("index " + str(i) + ", " + str(j) + ", " + str(k) + " is out of bounds for axis 0 with size " + str(self.shape[0]) + ", " + str(self.shape[1]) + ", " + str(self.shape[2]))
        deref(self.thisptr).set(i, j, k, value)

    #TODO: doing this requires the constructor below takes in a compressed array so we can reconstruct/pickle
    #       this is essentially our serialize/deserialize methods but in a pythonic format
    #def __reduce__(self):
    #    return (self.__class__, (self.shape[0], self.rate()))
    #TODO: once this is done we will also want explicit dump/dumps calls that the user can use. this maps to how ndarrays work

    def compressed_size(self):
        return deref(<array*>self.thisptr).compressed_size()

    #TODO: numpy calls this tobytes()
    def compressed_data(self):
        cdef size_t sz = self.compressed_size()
        cdef unsigned char* buff = <unsigned char*>malloc(sz)
        try:
            memcpy(<char*>buff, <char*>deref(<array*>self.thisptr).compressed_data(), sz)
        except:
            free(buff)
        return buff

    def rate(self):
        return deref(<array*>self.thisptr).rate()

    def set_rate(self, double rate):
        return deref(<array*>self.thisptr).set_rate(rate)

    #def get_header(self):
    #    return deref(<array*>self.thisptr).get_header()



cdef class zfparray3d:
    cdef py_array3[double] *thisptr
    cdef readonly str dtype
    cdef readonly list shape

    def __cinit__(self, size_t sz_x, sz_y, sz_z, double rate, size_t csize = 0):
        self.shape = [sz_x, sz_y, sz_z]
        self.dtype = "double32"

        # note: cython needs some help to figure out which overload we want here (hence the explicit casts)
        self.thisptr = new py_array3[double](<unsigned int>self.shape[0], <unsigned int>self.shape[1], <unsigned int>self.shape[2], <double>rate, <double*>0, <size_t>csize)

    def __dealloc__(self):
        del self.thisptr

    #TODO: multi-dimensional and slicing
    def __getitem__(self, int i):
        return self.flat_get(<long long>i)

    #TODO: multi-dimensional and slicing
    def __setitem__(self, int i, double value):
        self.flat_set(<long long>i, value)

    def flat_get(self, long long i):
        if i >= self.shape[0]*self.shape[1]*self.shape[2] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0] * self.shape[1] * self.shape[2]))
        cdef double val = deref(self.thisptr).get(i)
        return val

    def get(self, long long i, long long j, long long k):
        if i >= self.shape[0] or j >= self.shape[1] or k >= self.shape[2] or i < 0 or j < 0 or k < 0:
            raise IndexError("index " + str(i) + ", " + str(j) + ", " + str(k) + " is out of bounds for axis 0 with size " + str(self.shape[0]) + ", " + str(self.shape[1]) + ", " + str(self.shape[2]))
        cdef double val = deref(self.thisptr).get(i, j, k)
        return val

    def flat_set(self, long long i, double value):
        if i >= self.shape[0]*self.shape[1]*self.shape[2] or i < 0:
            raise IndexError("index " + str(i) + " is out of bounds for axis 0 with size " + str(self.shape[0] * self.shape[1] * self.shape[2]))
        deref(self.thisptr).set(i, value)

    def set(self, long long i, long long j, long long k, double value):
        if i >= self.shape[0] or j >= self.shape[1] or k >= self.shape[2] or i < 0 or j < 0 or k < 0:
            raise IndexError("index " + str(i) + ", " + str(j) + ", " + str(k) + " is out of bounds for axis 0 with size " + str(self.shape[0]) + ", " + str(self.shape[1]) + ", " + str(self.shape[2]))
        deref(self.thisptr).set(i, j, k, value)

    #TODO: doing this requires the constructor below takes in a compressed array so we can reconstruct/pickle
    #       this is essentially our serialize/deserialize methods but in a pythonic format
    #def __reduce__(self):
    #    return (self.__class__, (self.shape[0], self.rate()))
    #TODO: once this is done we will also want explicit dump/dumps calls that the user can use. this maps to how ndarrays work

    def compressed_size(self):
        return deref(<array*>self.thisptr).compressed_size()

    #TODO: numpy calls this tobytes()
    def compressed_data(self):
        cdef size_t sz = self.compressed_size()
        cdef unsigned char* buff = <unsigned char*>malloc(sz)
        try:
            memcpy(<char*>buff, <char*>deref(<array*>self.thisptr).compressed_data(), sz)
        except:
            free(buff)
        return buff

    def rate(self):
        return deref(<array*>self.thisptr).rate()

    def set_rate(self, double rate):
        return deref(<array*>self.thisptr).set_rate(rate)

    #def get_header(self):
    #    return deref(<array*>self.thisptr).get_header()
