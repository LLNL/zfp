from setuptools import setup, Extension

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__

setup(
    name="zfpy",
    setup_requires=["numpy", "cython"],
    version="1.0.1",
    author="Peter Lindstrom, Danielle Asher",
    author_email="zfp@llnl.gov",
    url="https://zfp.llnl.gov",
    description="zfp compression in Python",
    long_description="zfp is a compressed format for representing multidimensional floating-point and integer arrays. zfp provides compressed-array classes that support high throughput read and write random access to individual array elements. zfp also supports serial and parallel compression of whole arrays using both lossless and lossy compression with error tolerances. zfp is primarily written in C and C++ but also includes Python and Fortran bindings.",
    ext_modules=[
        Extension(
            "zfpy", 
            sources=["python/zfpy.pyx"],
            include_dirs=["include", str(NumpyImport())],
            libraries=["zfp"], 
            library_dirs=["build/lib64", "build/lib/Release"],
            language_level=3,
            lanugage="c",
        ),
    ],
)
