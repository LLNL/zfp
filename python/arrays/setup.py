from distutils.core import setup  
from distutils.extension import Extension  
from Cython.Build import cythonize  

#TODO: this is temporary and should be moved from a setup script to using cmake

from distutils.core import setup

setup(
    ext_modules = cythonize([Extension("zfpy", 
                                       ["/path/to/array.pyx"], 
                                       libraries=["zfp"],
                                       library_dirs=["/path/to/build/lib/"],
                                       include_dirs=["/path/to/zfp/include", "path/to/zfp/array/"]
                                       )])
)
