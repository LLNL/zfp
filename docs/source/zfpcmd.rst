.. include:: defs.rst

.. _zfpcmd:

File Compressor
===============

The |zfpcmd| executable in the :file:`bin` directory is primarily
intended for evaluating the rate-distortion (compression ratio and quality)
provided by the compressor, but since version 0.5.0 also allows reading and
writing compressed data sets.  |zfpcmd| takes as input a raw, binary
array of floats, doubles, or integers in native byte order and optionally
outputs a compressed or reconstructed array obtained after lossy compression
followed by decompression.  Various statistics on compression ratio and
error are also displayed.

The uncompressed input and output files should be a flattened, contiguous
sequence of scalars without any header information, generated for instance
by
::

    double* data = new double[nx * ny * nz];
    // populate data
    FILE* file = fopen("data.bin", "wb");
    fwrite(data, sizeof(*data), nx * ny * nz, file);
    fclose(file);


|zfpcmd| requires a set of command-line options, the most important
being the :option:`-i` option that specifies that the input is uncompressed.
When present, :option:`-i` tells |zfpcmd| to read an uncompressed input
file and compress it to memory.  If desired, the compressed stream can be
written to an output file using :option:`-z`.  When :option:`-i` is absent,
on the other hand, :option:`-z` names the compressed input (not output) file,
which is then decompressed.  In either case, :option:`-o` can be used to
output the reconstructed array resulting from lossy compression and
decompression.

So, to compress a file, use :code:`-i file.in -z file.zfp`.  To later
decompress the file, use :code:`-z file.zfp -o file.out`.  A single dash
"-" can be used in place of a file name to denote standard input or output.

When reading uncompressed input, the floating-point precision (single or
double) must be specified using either :option:`-f` (float) or
:option:`-d` (double).  In addition, the array dimensions must be specified
using :option:`-1` (for 1D arrays), :option:`-2` (for 2D arrays), or
:option:`-3` (for 3D arrays).  For multidimensional arrays, *x* varies
faster than *y*, which in turn varies faster than *z*.  That is, a 3D input
file corresponding to a flattened C array :code:`a[nz][ny][nx]` is
specified as :code:`-3 nx ny nz`.

Note that :code:`-2 nx ny` is not equivalent to :code:`-3 nx ny 1`, even
though the same number of values are compressed.  One invokes the 2D codec,
while the other uses the 3D codec, which in this example has to pad the
input to an *nx* |times| *ny* |times| 4 array since arrays are partitioned
into blocks of dimensions |4powd|.  Such padding usually negatively impacts
compression.

Moreover, :code:`-2 nx ny` is not equivalent to :code:`-2 ny nx`, i.e., with
the dimensions transposed.  It is crucial for accuracy and compression ratio
that the array dimensions are listed in the order expected by |zfpcmd| so
that the array layout is correctly interpreted.  See this
:ref:`discussion <p-dimensions>` for more details.

Using :option:`-h`, the array dimensions and type are stored in a header of
the compressed stream so that they do not have to be specified on the command
line during decompression.  The header also stores compression parameters,
which are described below.  The compressor and decompressor must agree on
whether headers are used, and it is up to the user to enforce this.

|zfpcmd| accepts several options for specifying how the data is to be
compressed.  The most general of these, the :option:`-c` option, takes four
constraint parameters that together can be used to achieve various effects.
These constraints are::

    minbits: the minimum number of bits used to represent a block
    maxbits: the maximum number of bits used to represent a block
    maxprec: the maximum number of bit planes encoded
    minexp:  the smallest bit plane number encoded

These parameters are discussed in detail in the section on
:ref:`compression modes <modes>`.  Options :option:`-r`, :option:`-p`,
and :option:`-a` provide a simpler interface to setting all of the above
parameters by invoking
:ref:`fixed-rate <mode-fixed-rate>` (:option:`-r`),
:ref:`-precision <mode-fixed-precision>` (:option:`-p`), and
:ref:`-accuracy <mode-fixed-accuracy>` (:option:`-a`).

Usage
-----

Below is a description of each command-line option accepted by |zfpcmd|.

General options
^^^^^^^^^^^^^^^

.. option:: -h

  Read/write array and compression parameters from/to compressed header.

.. option:: -q

  Quiet mode; suppress diagnostic output.

.. option:: -s

  Evaluate and print the following error statistics:

  * rmse: The root mean square error.
  * nrmse: The root mean square error normalized to the range.
  * maxe: The maximum absolute pointwise error.
  * psnr: The peak signal to noise ratio in decibels.

Input and output
^^^^^^^^^^^^^^^^

.. option:: -i <path>

  Name of uncompressed binary input file.  Use "-" for standard input.

.. option:: -o <path>

  Name of decompressed binary output file.  Use "-" for standard output.
  May be used with either :option:`-i`, :option:`-z`, or both.

.. option:: -z <path>

  Name of compressed input (without :option:`-i`) or output file (with
  :option:`-i`).  Use "-" for standard input or output.

When :option:`-i` is specified, data is read from the corresponding
uncompressed file, compressed, and written to the compressed file
specified by :option:`-z` (when present).  Without :option:`-i`,
compressed data is read from the file specified by :option:`-z`
and decompressed.  In either case, the reconstructed data can be
written to the file specified by :option:`-o`.

Array type and dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. option:: -f

  Single precision (float type).  Shorthand for :code:`-t f32`.

.. option:: -d

  Double precision (double type).  Shorthand for :code:`-t f64`.

.. option:: -t <type>

  Specify scalar type as one of i32, i64, f32, f64 for 32- or 64-bit
  integer or floating scalar type.

.. option:: -1 <nx>

  Dimensions of 1D C array :code:`a[nx]`.

.. option:: -2 <nx> <ny>

  Dimensions of 2D C array :code:`a[ny][nx]`.

.. option:: -3 <nx> <ny> <nz>

  Dimensions of 3D C array :code:`a[nz][ny][nx]`.

When :option:`-i` is used, the scalar type and array dimensions must be
specified.  One of :option:`-f`, :option:`-d`, or :option:`-t` specifies
the input scalar type.  :option:`-1`, :option:`-2`, or :option:`-3`
specifies the array dimensions.  The same parameters must be given when
decompressing data (without :option:`-i`), unless a header was stored
using :option:`-h` during compression.

Compression parameters
^^^^^^^^^^^^^^^^^^^^^^

.. option:: -r <rate>

  Specify fixed rate in terms of number of compressed bits per
  floating-point value.

.. option:: -p <precision>

  Specify fixed precision in terms of number of uncompressed bits per
  value.

.. option:: -a <tolerance>

  Specify fixed accuracy in terms of absolute error tolerance.

.. option:: -c <minbits> <maxbits> <maxprec> <minexp>

  Specify expert mode parameters.

When :option:`-i` is used, the compression parameters must be specified.
The same parameters must be given when decompressing data (without
:option:`-i`), unless a header was stored using :option:`-h` when
compressing.  See the section on :ref:`compression modes <modes>` for a
discussion of these parameters.

Examples
^^^^^^^^

  * :code:`-i file` : read uncompressed file and compress to memory
  * :code:`-z file` : read compressed file and decompress to memory
  * :code:`-i ifile -z zfile` : read uncompressed ifile, write compressed zfile
  * :code:`-z zfile -o ofile` : read compressed zfile, write decompressed ofile
  * :code:`-i ifile -o ofile` : read ifile, compress, decompress, write ofile
  * :code:`-i file -s` : read uncompressed file, compress to memory, print stats
  * :code:`-i - -o - -s` : read stdin, compress, decompress, write stdout, print stats
  * :code:`-f -3 100 100 100 -r 16` : 2x fixed-rate compression of 100 |times| 100 |times| 100 floats
  * :code:`-d -1 1000000 -r 32` : 2x fixed-rate compression of 1,000,000 doubles
  * :code:`-d -2 1000 1000 -p 32` : 32-bit precision compression of 1000 |times| 1000 doubles
  * :code:`-d -1 1000000 -a 1e-9` : compression of 1,000,000 doubles with < 10\ :sup:`-9` max error
  * :code:`-d -1 1000000 -c 64 64 0 -1074` : 4x fixed-rate compression of 1,000,000 doubles
