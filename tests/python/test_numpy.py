#!/usr/bin/env python

import unittest

import zfp
import numpy as np


class TestNumpy(unittest.TestCase):
    def tolerance_round_trip(self, orig_array):
        compressed_array = zfp.compress_numpy(orig_array, tolerance=1e-4)
        decompressed_array = zfp.decompress_numpy(compressed_array)
        self.assertIsNone(np.testing.assert_allclose(decompressed_array, orig_array, atol=1e-3))

    def test_large_zeros_array(self):
        zero_array = np.zeros((1000,1000), dtype=np.float64)
        self.tolerance_round_trip(zero_array)

    def test_different_dimensions(self):
        for dimensions in range(1, 5):
            shape = range(5, 5 + dimensions)

            c_array = np.random.rand(*shape)
            self.tolerance_round_trip(c_array)

            f_array = np.asfortranarray(c_array)
            self.tolerance_round_trip(f_array)

    def test_different_dtypes(self):
        shape = (5, 10)
        num_elements = 50

        for dtype in [np.float32, np.float64]:
            elements = np.random.random_sample(num_elements)
            elements = elements.astype(dtype, casting="same_kind")
            array = np.reshape(elements, newshape=shape)
            self.tolerance_round_trip(array)

        for dtype in [np.int32, np.int64]:
            array = np.random.randint(2**30, size=shape, dtype=dtype)
            with self.assertRaises(NotImplementedError):
                self.tolerance_round_trip(array)

    def test_fixed_rate(self):
        c_array = np.random.rand(25, 25, 25)

        decompressed_array = zfp.decompress_numpy(zfp.compress_numpy(c_array, rate=15))
        self.assertIsNone(np.testing.assert_allclose(decompressed_array, c_array, atol=1e-3))

        decompressed_array = zfp.decompress_numpy(zfp.compress_numpy(c_array, rate=1))
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(decompressed_array, c_array, atol=1e-3)

    def test_fixed_precision(self):
        c_array = np.random.rand(25, 25, 25)

        decompressed_array = zfp.decompress_numpy(zfp.compress_numpy(c_array, precision=24))
        self.assertIsNone(np.testing.assert_allclose(decompressed_array, c_array, atol=1e-3))

        decompressed_array = zfp.decompress_numpy(zfp.compress_numpy(c_array, precision=1))
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(decompressed_array, c_array, atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
