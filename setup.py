from setuptools import setup, Extension
import numpy as np

setup(
    url="https://zfp.llnl.gov",
    #long_description="zfp is a compressed format for representing multidimensional floating-point and integer arrays. zfp provides compressed-array classes that support high throughput read and write random access to individual array elements. zfp also supports serial and parallel compression of whole arrays using both lossless and lossy compression with error tolerances. zfp is primarily written in C and C++ but also includes Python and Fortran bindings.",
    ext_modules=[Extension("zfpy", ["python/zfpy.pyx"],
                           include_dirs=["include", np.get_include()],
                           libraries=["zfp"], library_dirs=["build/lib64", "build/lib"])],
)
