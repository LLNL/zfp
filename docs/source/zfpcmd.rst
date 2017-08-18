.. include:: defs.rst

.. _zfpcmd:

File Compressor
===============

The 'zfp' executable in the bin directory is primarily intended for
evaluating the rate-distortion (compression ratio and quality) provided by
the compressor, but since version 0.5.0 also allows reading and writing
compressed data sets.  zfp takes as input a raw, binary array of floats or
doubles in native byte order, and optionally outputs a compressed or
reconstructed array obtained after lossy compression followed by
decompression.  Various statistics on compression ratio and error are also
displayed.

zfp requires a set of command-line options, the most important being the
-i option that specifies that the input is uncompressed.  When present,
"-i <file>" tells zfp to read the uncompressed input file and compress it
to memory.  If desired, the compressed stream can be written to an output
file using "-z <file>".  When -i is absent, on the other hand, -z names
the compressed input (not output) file, which is then decompressed.  In
either case, "-o <file>" can be used to output the reconstructed array
resulting from lossy compression and decompression.

So, to compress a file, use "-i file.in -z file.zfp".  To later decompress
the file, use "-z file.zfp -o file.out".  A single dash "-" can be used in
place of a file name to denote standard input or output.

When reading uncompressed input, the floating-point precision (single or
double) must be specified using either -f (float) or -d (double).  In
addition, the array dimensions must be specified using "-1 nx" (for 1D
arrays), "-2 nx ny" (for 2D arrays), or "-3 nx ny nz" (for 3D arrays).
For multidimensional arrays, x varies faster than y, which in turn varies
faster than z.  That is, a 3D input file should correspond to a flattened
C array declared as a[nz][ny][nx].

Note that "-2 nx ny" is not equivalent to "-3 nx ny 1", even though the
same number of values are compressed.  One invokes the 2D codec, while the
other uses the 3D codec, which in this example has to pad the input to an
*nx* |times| *ny* |times| 4 array since arrays are partitioned into blocks
of dimensions |4powd|.  Such padding usually negatively impacts compression.

Using -h, the array dimensions and type are stored in a header of the
compressed stream so that they do not have to be specified on the command
line during decompression.  The header also stores compression parameters,
which are described below.

zfp accepts several options for specifying how the data is to be compressed.
The most general of these, the -c option, takes four constraint parameters
that together can be used to achieve various effects.  These constraints
are::

    minbits: the minimum number of bits used to represent a block
    maxbits: the maximum number of bits used to represent a block
    maxprec: the maximum number of bit planes encoded
    minexp:  the smallest bit plane number encoded

These parameters are discussed in detail in the section on
:ref:`compression modes <modes>`.  Options -r, -p, and -a provide a simpler
interface to setting all of the above parameters by invoking
:ref:`fixed-rate <mode-fixed-rate>` (-r),
:ref:`-precision <mode-fixed-precision>` (-p), and
:ref:`-accuracy <mode-fixed-accuracy>` (-a).

Usage
-----

**To be added**
