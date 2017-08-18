.. include:: defs.rst

.. _directions:

Future Directions
=================

|zfp| is actively being developed and plans have been made to add a number of
important features, including:

- Support for 4D arrays, e.g. for compressing time-varying 3D fields.  Although
  the |zfp| compression algorithm trivially generalizes to higher dimensions,
  *d*, the current implementation is hampered by the lack of integer types large
  enough to hold |4powd| bits for *d* > 3.  For now, higher-dimensional data
  should be compressed as collections of independent 3D fields.

- Tagging of missing values.  |zfp| currently assumes that arrays are
  dense, i.e., each array element has an associated value.  In many
  science applications some values are missing.  For instance, in climate
  modeling, ocean temperature is not defined over land.  In other
  applications, the domain is not rectangular but irregular and embedded
  in a rectangular array.  Such examples of sparse arrays suggest the
  need for a mechanism to tag values as missing or indeterminate.
  Current solutions often rely on NaNs or special, often very large sentinel
  values outside the normal range, which can lead to poor compression and
  complete loss of accuracy in nearby valid values.  See
  :ref:`FAQ #7 <q-missing>`.

- Support for NaNs and infinities.  Similar to missing values, some
  applications store special IEEE floating-point values that are not yet
  supported by |zfp|.  In fact, the presence of such values will currently
  force undefined behavior.

- Lossless compression. **To be added**

- Progressive decompression. **To be added**

- Parallel compression. **To be added**

- Thread-safe compressed arrays. **To be added**

- Array views. **To be added**

Please contact `Peter Lindstrom <mailto:pl@llnl.gov>`__ with requests for
features not listed above.
