.. include:: defs.rst

Regression Tests
================

The :program:`testzfp` program in the :file:`tests` directory performs
regression testing that exercises most of the functionality of :file:`libzfp`
and the array classes.  The tests assume the default compiler settings,
i.e. with none of the macros in :file:`Config` defined.  By default, small,
pregenerated floating-point arrays are used in the test, since they tend to
have the same binary representation across platforms, whereas it can be
difficult to computationally generate bit-for-bit identical arrays.  To test
larger arrays, use the :code:`medium` or :code:`large` options.
When large arrays are used, the (de)compression throughput is also measured
and reported in number of uncompressed bytes per second.
