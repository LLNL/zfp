ZFP
===
[![Travis CI Build Status](https://travis-ci.com/LLNL/zfp.svg?branch=develop)](https://travis-ci.com/LLNL/zfp)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/qb3ld7j11segy52k/branch/develop?svg=true)](https://ci.appveyor.com/project/lindstro/zfp)
[![Documentation Status](https://readthedocs.org/projects/zfp/badge/?version=release0.5.5)](https://zfp.readthedocs.io/en/release0.5.5/?badge=release0.5.5)
[![Code Coverage](https://codecov.io/gh/LLNL/zfp/branch/develop/graph/badge.svg)](https://codecov.io/gh/LLNL/zfp)

zfp is a compressed format for representing multidimensional floating-point
and integer arrays.  zfp provides compressed-array classes that support high
throughput read and write random access to individual array elements.  zfp
also supports serial and parallel (OpenMP and CUDA) compression of whole
arrays, e.g., for applications that read and write large data sets to and
from disk.

zfp uses lossy but optionally error-bounded compression to achieve high
compression ratios.  Bit-for-bit lossless compression is also possible
through one of zfp's compression modes.  zfp works best for 2D, 3D, and 4D
arrays that exhibit spatial correlation, such as continuous fields from
physics simulations, natural images, regularly sampled terrain surfaces, etc.
zfp compression of 1D arrays is possible but generally discouraged.

zfp is freely available as open source and is distributed under a BSD license.
zfp is primarily written in C and C++ but also includes Python and Fortran
bindings.  zfp conforms to various language standards, including C89, C99,
C11, C++98, C++11, and C++14, and is supported on Linux, macOS, and Windows.


Quick Start
-----------

To download zfp, type:

    git clone https://github.com/LLNL/zfp.git

zfp may be built using either [CMake](https://cmake.org/) or
[GNU make](https://www.gnu.org/software/make/).  To use CMake, type:

    cd zfp
    mkdir build
    cd build
    cmake ..
    cmake --build . --config Release
    ctest

This builds the zfp library in the `build/lib` directory and the zfp
command-line executable in the `build/bin` directory.  It then runs
the regression tests.

zfp may also be built using GNU make:

    cd zfp
    make
    make test

Note: GNU builds are less flexible and do not support all available features,
e.g., CUDA support.

For further configuration and build instructions, please consult the
[documentation](https://zfp.readthedocs.io/en/latest/installation.html).


Documentation
-------------

Full HTML [documentation](http://zfp.readthedocs.io/) is available online.
A [PDF](http://readthedocs.org/projects/zfp/downloads/pdf/latest/) version
is also available.


Contributing
------------

The zfp project uses the
[Gitflow](https://nvie.com/posts/a-successful-git-branching-model/)
development model.  Contributions should be made as pull requests on the
`develop` branch.  Although this branch is under continuous development,
it should be robust enough to pass all regression tests.
The `master` branch is updated with each release and reflects the most
recent official release of zfp.  See the
[Releases Page](https://github.com/LLNL/zfp/releases) for a history
of releases.


Authors
-------

zfp was originally developed by [Peter Lindstrom](https://people.llnl.gov/pl)
at [Lawrence Livermore National Laboratory](https://www.llnl.gov/).  Please
see the [Contributors Page](https://github.com/LLNL/zfp/graphs/contributors)
for a full list of contributors.

### Citing zfp

If you use zfp for scholarly research, please cite this paper:

* Peter Lindstrom.
  [Fixed-Rate Compressed Floating-Point Arrays](https://www.researchgate.net/publication/264417607_Fixed-Rate_Compressed_Floating-Point_Arrays).
  IEEE Transactions on Visualization and Computer Graphics, 20(12):2674-2683, December 2014.
  [doi:10.1109/TVCG.2014.2346458](http://doi.org/10.1109/TVCG.2014.2346458).


Additional Resources
--------------------

For more information on zfp, please see the
[zfp website](https://computing.llnl.gov/casc/zfp/).
For bug reports and feature requests, please consult the
[GitHub issue tracker](https://github.com/LLNL/zfp/issues/).
For questions and comments not answered here or in the
[documentation](http://zfp.readthedocs.io),
please send e-mail to [zfp@llnl.gov](mailto:zfp@llnl.gov).


License
-------

zfp is distributed under the terms of the BSD 3-Clause license.  See the
files [LICENSE](https://github.com/LLNL/zfp/blob/develop/LICENSE) and
[NOTICE](https://github.com/LLNL/zfp/blob/develop/NOTICE) for details.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-663824
