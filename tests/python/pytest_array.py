#!/usr/bin/env python

import unittest
from random import random

import zfpy
import test_utils
try:
    from packaging.version import parse as version_parse
except ImportError:
    version_parse = None

#TODO: checks for compressed_data
#TODO: multi-d get/set checks
#TODO: checksumming
#TODO: paramaterized tests

class TestArray1f(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.array1f(1000, 64)
        self.assertEqual(zfpArray.dtype, "float32")
        self.assertEqual(zfpArray.shape, (1000,))
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [random() for idx in range(1000)]
        zfpArray = zfpy.array1f(1000, 64)
        for idx in range(1000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.array1f(1000, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.array1f(1000, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)


class TestArray1d(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.array1d(1000, 64)
        self.assertEqual(zfpArray.dtype, "float64")
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [random() for idx in range(1000)]
        zfpArray = zfpy.array1d(1000, 64)
        for idx in range(1000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.array1d(1000, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.array1d(1000, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)


class TestArray2f(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.array2f(100, 100, 64)
        self.assertEqual(zfpArray.dtype, "float32")
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [[random() for idx in range(100)] for idx in range(100)]
        zfpArray = zfpy.array2f(100, 100, 64)
        for j in range(100):
            for i in range(100):
                zfpArray.set(i, j, inputArray[i][j])
                self.assertAlmostEqual(zfpArray.get(i, j), inputArray[i][j], places=6)

    def test_array_flat_get_set(self):
        inputArray = [random() for idx in range(10000)]
        zfpArray = zfpy.array2f(100, 100, 64)
        for idx in range(10000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.array2f(100, 100, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.array2f(100, 100, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)


class TestArray2d(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.array2d(100, 100, 64)
        self.assertEqual(zfpArray.dtype, "float64")
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [[random() for idx in range(100)] for idx in range(100)]
        zfpArray = zfpy.array2d(100, 100, 64)
        for j in range(100):
            for i in range(100):
                zfpArray.set(i, j, inputArray[i][j])
                self.assertAlmostEqual(zfpArray.get(i, j), inputArray[i][j], places=6)

    def test_array_flat_get_set(self):
        inputArray = [random() for idx in range(10000)]
        zfpArray = zfpy.array2d(100, 100, 64)
        for idx in range(10000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.array2d(100, 100, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.array2d(100, 100, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)


class TestArray3f(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.array3f(10, 10, 10, 64)
        self.assertEqual(zfpArray.dtype, "float32")
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [[[random() for idx in range(10)] for idx in range(10)] for idx in range(10)]
        zfpArray = zfpy.array3f(10, 10, 10, 64)
        for k in range(10):
            for j in range(10):
                for i in range(10):
                    zfpArray.set(i, j, k, inputArray[i][j][k])
                    self.assertAlmostEqual(zfpArray.get(i, j, k), inputArray[i][j][k], places=6)

    def test_array_flat_get_set(self):
        inputArray = [random() for idx in range(10000)]
        zfpArray = zfpy.array3f(10, 10, 10, 64)
        for idx in range(1000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.array3f(10, 10, 10, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.array2f(10, 10, 10, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)


class TestArray3d(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.array3d(10, 10, 10, 64)
        self.assertEqual(zfpArray.dtype, "float64")
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [[[random() for idx in range(10)] for idx in range(10)] for idx in range(10)]
        zfpArray = zfpy.array3d(10, 10, 10, 64)
        for k in range(10):
            for j in range(10):
                for i in range(10):
                    zfpArray.set(i, j, k, inputArray[i][j][k])
                    self.assertAlmostEqual(zfpArray.get(i, j, k), inputArray[i][j][k], places=6)

    def test_array_flat_get_set(self):
        inputArray = [random() for idx in range(1000)]
        zfpArray = zfpy.array3d(10, 10, 10, 64)
        for idx in range(1000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.array3d(10, 10, 10, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.array3d(10, 10, 10, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)



if __name__ == "__main__":
    unittest.main(verbosity=2)
