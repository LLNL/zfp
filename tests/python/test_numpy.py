#!/usr/bin/env python

import unittest

import zfp
import test_utils
import numpy as np


class TestNumpy(unittest.TestCase):
    def test_utils(self):
        for ndims in range(1, 5):
            for ztype, ztype_str in [
                    (zfp.type_float,  "float"),
                    (zfp.type_double, "double"),
                    (zfp.type_int32,  "int32"),
                    (zfp.type_int64,  "int64"),
            ]:
                orig_random_array = test_utils.getRandNumpyArray(ndims, ztype)
                orig_checksum = test_utils.getChecksumOrigArray(ndims, ztype)
                actual_checksum = test_utils.hashNumpyArray(orig_random_array)
                self.assertEqual(orig_checksum, actual_checksum)

                for stride_str, stride_config in [
                        ("as_is", test_utils.stride_as_is),
                        ("permuted", test_utils.stride_permuted),
                        ("interleaved", test_utils.stride_interleaved),
                        #("reversed", test_utils.stride_reversed),
                ]:
                    if stride_config == test_utils.stride_permuted and ndims == 1:
                        continue
                    random_array = test_utils.generateStridedRandomNumpyArray(
                        stride_config,
                        orig_random_array
                    )
                    self.assertTrue(np.equal(orig_random_array, random_array).all())

                    for compress_param_num in range(3):
                        modes = [(zfp.mode_fixed_accuracy, "tolerance"),
                                 (zfp.mode_fixed_precision, "precision"),
                                 (zfp.mode_fixed_rate, "rate")]
                        if ztype in [zfp.type_int32, zfp.type_int64]:
                            modes = [modes[-1]] # only fixed-rate is supported for integers
                        for mode, mode_str in modes:
                            # Compression
                            compression_kwargs = {
                                mode_str: test_utils.computeParameterValue(
                                    mode,
                                    compress_param_num
                                ),
                            }

                            compressed_array = zfp.compress_numpy(
                                random_array,
                                write_header=False,
                                **compression_kwargs
                            )
                            compressed_checksum = test_utils.getChecksumCompArray(
                                ndims,
                                ztype,
                                mode,
                                compress_param_num
                            )
                            actual_checksum = test_utils.hashCompressedArray(
                                compressed_array
                            )
                            self.assertEqual(compressed_checksum, actual_checksum)

                            # Decompression
                            zshape = [int(x) for x in random_array.shape]
                            decompression_kwargs = dict(
                                compression_kwargs,
                                ztype=ztype,
                                shape=zshape,
                            )
                            if stride_config == test_utils.stride_permuted:
                                # test decompressing into a numpy array
                                # created by the "user"
                                decompressed_array = np.empty_like(random_array)
                                decompression_kwargs['out'] = decompressed_array

                            decompressed_array = zfp.decompress_numpy(
                                compressed_array,
                                **decompression_kwargs
                            )
                            actual_checksum = test_utils.hashNumpyArray(
                                decompressed_array
                            )
                            decompressed_checksum = test_utils.getChecksumDecompArray(
                                ndims,
                                ztype,
                                mode,
                                compress_param_num
                            )
                            self.assertEqual(decompressed_checksum, actual_checksum)

if __name__ == "__main__":
    unittest.main(verbosity=2)
