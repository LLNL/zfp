.. include:: defs.rst

Simultaneous Codec Support
==========================

Each zfp library release contains a single codec used in performing compression
and decompression. Some applications may desire to upgrade to a newer codec,
while supporting existing compressed data under older codecs. This section
describes how to compile multiple zfp releases together, and use their combined
API.

Simultaneous codec support was introduced into zfp at version x.y.z (having
codec y).

The codec is defined with macro :c:macro:`ZFP_CODEC`, or can be accessed through
:c:data:`zfp_codec_version`.

To compile compatible zfp libraries together, navigate to the latest zfp
directory and call the CMake command with flag :c:macro:`ZFP_VX_DIR`

For example, if codecs 5 and 6 are to be compiled together, navigate to the
directory for zfp codec 6 and pass
:c:macro:`ZFP_V5_DIR` ="/path/to/zfpCodec5/project/directory"

Then in the application, include :file:`zfpApi.h` rather than :file:`zfp.h`

Combined API
------------

Prefix Mechanism
^^^^^^^^^^^^^^^^

To be able to compile multiple zfp libraries together, the symbols must be
unique across each library version. We accomplish this by applying a prefix to
each type, function, and constant. The prefix, v5 for codec 5, is inserted in
each (snake-cased) name, after the initial zfp prefix. These are automatically
applied after passing an older zfp project path in the CMake command. Macros
perform the substitutions, allowing the source code to remain almost entirely
the same (without prefixes).

Macros:    :c:macro:`ZFP_VERSION_STRING` becomes :c:macro:`ZFP_V5_VERSION_STRING`

Types:     :c:type:`zfp_stream` becomes :c:type:`zfp_v5_stream`

Functions: :c:func:`zfp_stream_close` becomes :c:func:`zfp_v5_stream_close`

Exception: The bitstream API is never prefixed, as it is not expected to change.

This allows users to explicitly call API functions from a specific
codec/release. All functions from the :ref:`high-level API <hl-api>` and
:ref:`low-level API <ll-api>` are prefixed.

Preserving Original Unprefixed API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The original API set forth by :file:`zfp.h` is still available. One last set of
macros bind the unprefixed API to the latest codec's library/behavior.

Calling :c:macro:`zfp_stream_open()` will actually call
:c:macro:`zfp_vx_stream_open()`, where x is the latest codec in this compiled
bundle.

Redefined Functions
-------------------

Some functions are redefined to automatically route behavior to the proper
codec's API function. This becomes possible with the addition of the
:c:macro:`codec_version` field in :c:type:`zfp_stream`. This is initialized in
:c:func:`zfp_stream_open`, but can also be manually set with
:c:func:`zfp_stream_set_codec_version`, and queried with
:c:func:`zfp_stream_codec_version`.

Note that the prefixed versions of these functions are still available. Only
some unprefixed behavior changes.

:c:macro:`size_t zfp_stream_maximum_size(const zfp_stream* stream, const zfp_field* field)`

  Calls the proper prefixed :c:func:`zfp_stream_maximum_size` depending on the value of
  :c:data:`zfp->codec_version`.

:c:macro:`size_t zfp_write_header(zfp_stream* stream, const zfp_field* field, uint mask)`

  Calls the proper prefixed :c:func:`zfp_write_header` depending on the value of
  :c:data:`stream->codec_version`.

:c:macro:`size_t zfp_read_header(zfp_stream* stream, zfp_field* field, uint mask)`

  If :c:macro:`stream->codec_version` has value :c:macro:`ZFP_CODEC_WILDCARD`,
  this function attempts each prefixed :c:func:`zfp_read_header`. Returns zero
  if none were successful. On success, :c:data:`stream->codec_version` is
  updated with the corresponding codec.

  If :c:macro:`stream->codec_version` has a specific codec's value, this
  function only attempts that prefixed :c:func:`zfp_read_header`. This enables
  users to filter compressed data to those only with a specific codec. If the
  specified codec is not supported in the combined library, it will always fail
  (return zero).

:c:macro:`size_t zfp_compress(zfp_stream* stream, const zfp_field* field)`

  Calls the proper prefixed :c:func:`zfp_compress` depending on the value of
  :c:data:`stream->codec_version`.

:c:macro:`int zfp_decompress(zfp_stream* stream, zfp_field* field)`

  Calls the proper prefixed :c:func:`zfp_decompress` depending on the value of
  :c:data:`stream->codec_version`.

Example Use Case
^^^^^^^^^^^^^^^^

An application has written compressed data with a specific codec, and desires to
upgrade to a newer codec, while being able to decompress with the older codec,
and compress with only the newer codec.

First, compile both codecs together and include :file:`zfpApi.h` instead of
:file:`zfp.h`

After creating a :c:macro:`zfp_stream` through :c:func:`zfp_stream_open`, set
the :c:macro:`codec_version` to :c:macro:`ZFP_CODEC_WILDCARD` through
:c:func:`zfp_stream_set_codec_version`. This prepares
:c:macro:`zfp_read_header()` to attempt all possible codecs. If
:c:macro:`codec_version` was left to its initialized value,
:c:macro:`zfp_read_header()` would only attempt the latest codec's
:c:func:`zfp_read_header` (:c:macro:`zfp_vx_read_header()`).

Upon calling :c:macro:`zfp_read_header()`, both codecs'
:c:macro:`zfp_vx_read_header()` will be attempted, and if either is successful,
:c:macro:`zfp_stream->codec_version` will be updated with that codec value.
Next, :c:macro:`zfp_decompress()` will automatically route to the function from
the same codec.

Finally, it's possible to only compress with the newer codec with
:c:macro:`zfp_vx_compress()`. It's also possible to manually choose the codec
to compress with, through :c:func:`zfp_stream_set_codec_version()` and
:c:macro:`zfp_compress()`.

The purpose of these redefined unprefixed functions is to make it easier for
developers to handle multiple codecs, by reducing code they would write
(converting between types across versions, creating conditionals to call
specific prefixed functions).

Compiling a single version with prefixes
----------------------------------------

It is possible to compile a single version with prefixes applied. Compile with
:c:macro:`ZFP_WITH_VERSION_PREFIX`, and build as usual. After including
:file:`zfp.h` in an application, note that the unprefixed counterparts are
automatically bound to the prefixed ones with macros.
