import cython
cimport libc.stdint as stdint

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
