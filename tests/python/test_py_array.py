#!/usr/bin/env python

import unittest
from random import random

import zfpyarray as zfpy
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
        zfpArray = zfpy.zfparray1f(1000, 64)
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [random() for idx in range(1000)]
        zfpArray = zfpy.zfparray1f(1000, 64)
        for idx in range(1000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray1f(1000, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray1f(1000, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)


class TestArray1d(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.zfparray1d(1000, 64)
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [random() for idx in range(1000)]
        zfpArray = zfpy.zfparray1d(1000, 64)
        for idx in range(1000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray1d(1000, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray1d(1000, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)


class TestArray2f(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.zfparray2f(100, 100, 64)
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [random() for idx in range(10000)]
        zfpArray = zfpy.zfparray2f(100, 100, 64)
        for idx in range(10000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray2f(100, 100, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray2f(100, 100, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)


class TestArray2d(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.zfparray2d(100, 100, 64)
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [random() for idx in range(10000)]
        zfpArray = zfpy.zfparray2d(100, 100, 64)
        for idx in range(10000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray2d(100, 100, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray2d(100, 100, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)


class TestArray3f(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.zfparray3f(10, 10, 10, 64)
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [random() for idx in range(10000)]
        zfpArray = zfpy.zfparray3f(10, 10, 10, 64)
        for idx in range(1000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray3f(10, 10, 10, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray2f(10, 10, 10, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)


class TestArray3d(unittest.TestCase):
    def setUp(self):
        self.rates = [i*16 for i in range(1,20)]

    def test_array_default_init(self):
        zfpArray = zfpy.zfparray3d(10, 10, 10, 64)
        for val in zfpArray:
            self.assertAlmostEqual(val, 0.0, places=12)

    def test_array_get_set(self):
        inputArray = [random() for idx in range(1000)]
        zfpArray = zfpy.zfparray3d(10, 10, 10, 64)
        for idx in range(1000):
            zfpArray[idx] = inputArray[idx]
            self.assertAlmostEqual(zfpArray[idx], inputArray[idx], places=6)

    def test_rate_init(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray3d(10, 10, 10, rate)
            self.assertEqual(zfpArray.rate(), rate)

    def test_rate_set(self):
        for rate in self.rates:
            zfpArray = zfpy.zfparray3d(10, 10, 10, 1)
            zfpArray.set_rate(rate)
            self.assertEqual(zfpArray.rate(), rate)



if __name__ == "__main__":
    unittest.main(verbosity=2)
