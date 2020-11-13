#! /usr/bin/python

import numpy as np
arr = np.arange(0, 27, dtype=np.float32)
arr.tofile("out.dat")
