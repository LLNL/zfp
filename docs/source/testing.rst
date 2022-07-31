.. include:: defs.rst

Regression Tests
================

The :program:`testzfp` program performs basic regression testing by exercising
a small but important subset of |libzfp| and the compressed-array
classes.  It serves as a sanity check that |zfp| has been built properly.
These tests assume the default compiler settings, i.e., with none of the
settings in :file:`Config` or :file:`CMakeLists.txt` modified.  By default,
small, synthetic floating-point arrays are used in the test.  To test larger
arrays, use the :code:`large` command-line option.  When large arrays are
used, the (de)compression throughput is also measured and reported in number
of uncompressed bytes per second.

More extensive unit and functional tests are available on the |zfp| GitHub
`develop branch <https://github.com/LLNL/zfp/tree/develop>`_ in the
:file:`tests` directory.
