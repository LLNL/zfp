from setuptools import setup, Extension
import numpy as np

setup(
    name="zfpy",
    version="0.5.5-rc1",
    author="Peter Lindstrom",
    author_email="zfp@llnl.gov",
    url="https://computing.llnl.gov/projects/floating-point-compression",
    description="zfp compression in Python",
    long_description="zfp compression in Python",
    ext_modules=[Extension("zfpy", ["build/python/zfpy.c"],
                           include_dirs=["include", np.get_include()],
                           libraries=["zfp"], library_dirs=["build/lib64", "build/lib/Release"])]
)
