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
    license="License :: OSI Approved :: BSD License",
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
    classifiers=[
        "Intended Audience :: Developers",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: System :: Archiving :: Compression",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
)
