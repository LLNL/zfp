#!/usr/bin/env python

import unittest

import zfpy
import test_utils
import numpy as np
try:
    from packaging.version import parse as version_parse
except ImportError:
    version_parse = None


class TestNumpy(unittest.TestCase):
    def lossless_round_trip(self, orig_array):
        compressed_array = zfpy.compress_numpy(orig_array, write_header=True)
        decompressed_array = zfpy.decompress_numpy(compressed_array)
        self.assertIsNone(np.testing.assert_array_equal(decompressed_array, orig_array))

    def test_different_dimensions(self):
        for dimensions in range(1, 5):
            shape = [5] * dimensions
            c_array = np.random.rand(*shape)
            self.lossless_round_trip(c_array)

            shape = range(2, 2 + dimensions)
            c_array = np.random.rand(*shape)
            self.lossless_round_trip(c_array)

    def test_different_dtypes(self):
        shape = (5, 5)
        num_elements = shape[0] * shape[1]

        for dtype in [np.float32, np.float64]:
            elements = np.random.random_sample(num_elements)
            elements = elements.astype(dtype, casting="same_kind")
            array = np.reshape(elements, newshape=shape)
            self.lossless_round_trip(array)

        if (version_parse is not None and
            (version_parse(np.__version__) >= version_parse("1.11.0"))
        ):
            for dtype in [np.int32, np.int64]:
                array = np.random.randint(2**30, size=shape, dtype=dtype)
                self.lossless_round_trip(array)
        else:
            array = np.random.randint(2**30, size=shape)
            self.lossless_round_trip(array)

    def test_advanced_decompression_checksum(self):
        ndims = 2
        ztype = zfpy.type_float
        random_array = test_utils.getRandNumpyArray(ndims, ztype)
        mode = zfpy.mode_fixed_accuracy
        compress_param_num = 1
        compression_kwargs = {
            "tolerance": test_utils.computeParameterValue(
                mode,
                compress_param_num
            ),
        }
        compressed_array = zfpy.compress_numpy(
            random_array,
            write_header=False,
            **compression_kwargs
        )

        # Decompression using the "advanced" interface which enforces no header,
        # and the user must provide all the metadata
        decompressed_array = np.empty_like(random_array)
        zfpy._decompress(
            compressed_array,
            ztype,
            random_array.shape,
            out=decompressed_array,
            **compression_kwargs
        )
        decompressed_array_dims = decompressed_array.shape + tuple(0 for i in range(4 - decompressed_array.ndim))
        decompressed_checksum = test_utils.getChecksumDecompArray(
            decompressed_array_dims,
            ztype,
            mode,
            compress_param_num
        )
        actual_checksum = test_utils.hashNumpyArray(
            decompressed_array
        )
        self.assertEqual(decompressed_checksum, actual_checksum)

    def test_memview_advanced_decompression_checksum(self):
        ndims = 2
        ztype = zfpy.type_float
        random_array = test_utils.getRandNumpyArray(ndims, ztype)
        mode = zfpy.mode_fixed_accuracy
        compress_param_num = 1
        compression_kwargs = {
            "tolerance": test_utils.computeParameterValue(
                mode,
                compress_param_num
            ),
        }
        compressed_array_tmp = zfpy.compress_numpy(
            random_array,
            write_header=False,
            **compression_kwargs
        )
        mem = memoryview(compressed_array_tmp)
        compressed_array = np.array(mem, copy=False)
        # Decompression using the "advanced" interface which enforces no header,
        # and the user must provide all the metadata
        decompressed_array = np.empty_like(random_array)
        zfpy._decompress(
            compressed_array,
            ztype,
            random_array.shape,
            out=decompressed_array,
            **compression_kwargs
        )
        decompressed_array_dims = decompressed_array.shape + tuple(0 for i in range(4 - decompressed_array.ndim))
        decompressed_checksum = test_utils.getChecksumDecompArray(
            decompressed_array_dims,
            ztype,
            mode,
            compress_param_num
        )
        actual_checksum = test_utils.hashNumpyArray(
            decompressed_array
        )
        self.assertEqual(decompressed_checksum, actual_checksum)

    def test_advanced_decompression_nonsquare(self):
        for dimensions in range(1, 5):
            shape = range(2, 2 + dimensions)
            random_array = np.random.rand(*shape)

            decompressed_array = np.empty_like(random_array)
            compressed_array = zfpy.compress_numpy(
                random_array,
                write_header=False,
            )
            zfpy._decompress(
                compressed_array,
                zfpy.dtype_to_ztype(random_array.dtype),
                random_array.shape,
                out= decompressed_array,
            )
            self.assertIsNone(np.testing.assert_array_equal(decompressed_array, random_array))

    def test_utils(self):
        for ndims in range(1, 5):
            for ztype, ztype_str in [
                    (zfpy.type_float,  "float"),
                    (zfpy.type_double, "double"),
                    (zfpy.type_int32,  "int32"),
                    (zfpy.type_int64,  "int64"),
            ]:
                orig_random_array = test_utils.getRandNumpyArray(ndims, ztype)
                orig_random_array_dims = orig_random_array.shape + tuple(0 for i in range(4 - orig_random_array.ndim))
                orig_checksum = test_utils.getChecksumOrigArray(orig_random_array_dims, ztype)
                actual_checksum = test_utils.hashNumpyArray(orig_random_array)
                self.assertEqual(orig_checksum, actual_checksum)

                for stride_str, stride_config in [
                        ("as_is", test_utils.stride_as_is),
                        ("permuted", test_utils.stride_permuted),
                        ("interleaved", test_utils.stride_interleaved),
                        #("reversed", test_utils.stride_reversed),
                ]:
                    # permuting a 1D array is not supported
                    if stride_config == test_utils.stride_permuted and ndims == 1:
                        continue
                    random_array = test_utils.generateStridedRandomNumpyArray(
                        stride_config,
                        orig_random_array
                    )
                    random_array_dims = random_array.shape + tuple(0 for i in range(4 - random_array.ndim))
                    self.assertTrue(np.equal(orig_random_array, random_array).all())

                    for compress_param_num in range(3):
                        modes = [(zfpy.mode_fixed_accuracy, "tolerance"),
                                 (zfpy.mode_fixed_precision, "precision"),
                                 (zfpy.mode_fixed_rate, "rate")]
                        if ztype in [zfpy.type_int32, zfpy.type_int64]:
                            modes = [modes[-1]] # only fixed-rate is supported for integers
                        for mode, mode_str in modes:
                            # Compression
                            compression_kwargs = {
                                mode_str: test_utils.computeParameterValue(
                                    mode,
                                    compress_param_num
                                ),
                            }

                            compressed_array = zfpy.compress_numpy(
                                random_array,
                                write_header=False,
                                **compression_kwargs
                            )
                            compressed_checksum = test_utils.getChecksumCompArray(
                                random_array_dims,
                                ztype,
                                mode,
                                compress_param_num
                            )
                            actual_checksum = test_utils.hashCompressedArray(
                                compressed_array
                            )
                            self.assertEqual(compressed_checksum, actual_checksum)

                            # Decompression
                            decompressed_checksum = test_utils.getChecksumDecompArray(
                                random_array_dims,
                                ztype,
                                mode,
                                compress_param_num
                            )

                            # Decompression using the "public" interface
                            # requires a header, so re-compress with the header
                            # included in the stream
                            compressed_array_tmp = zfpy.compress_numpy(
                                random_array,
                                write_header=True,
                                **compression_kwargs
                            )
                            mem = memoryview(compressed_array_tmp)
                            compressed_array = np.array(mem, copy=False)
                            decompressed_array = zfpy.decompress_numpy(
                                compressed_array,
                            )
                            actual_checksum = test_utils.hashNumpyArray(
                                decompressed_array
                            )
                            self.assertEqual(decompressed_checksum, actual_checksum)


if __name__ == "__main__":
    unittest.main(verbosity=2)
