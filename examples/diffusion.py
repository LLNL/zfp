#!/usr/bin/python3
import zfpy
import numpy as np
import math


class Constants:
    def __init__(self, nx, ny, nt):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.x0 = (nx - 1) / 2
        self.y0 = (ny - 1) / 2
        self.k = 0.04
        self.dx = 2.0 / (max(nx, ny) - 1)
        self.dy = 2.0 / (max(nx, ny) - 1)
        self.dt = 0.5 * (self.dx ** 2 + self.dy ** 2) / (8 * self.k)
        self.tfinal = nt * self.dt if nt != 0 else 1.0
        self.pi = math.pi


def total(u):
    s = 0.0
    nx = u.shape[0]
    ny = u.shape[1]
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            s = s + u.get(x, y)
    return s


def error(u, c, t):
    e = 0.0
    for y in range(1, c.ny - 1):
        py = c.dy * (y - c.y0)
        for x in range(1, c.nx - 1):
            px = c.dx * (x - c.x0)
            f = u.get(x, y)
            g = c.dx * c.dy * math.e ** (-1*(px ** 2 + py ** 2) / (4 * c.k * t)) / (4 * c.pi * c.k * t)
            e = e + (f - g) ** 2
    return math.sqrt(e / ((c.nx - 2) * (c.ny - 2)))


def time_step_indexed(u, c):
    du = zfpy.zfparray2d(c.nx, c.ny, u.rate())
    for y in range(1, c.ny - 1):
        for x in range(1, c.nx - 1):
            uxx = (u.get(x - 1, y) - 2 * u.get(x, y) + u.get(x + 1, y)) / (c.dx ** 2)
            uyy = (u.get(x, y - 1) - 2 * u.get(x, y) + u.get(x, y + 1)) / (c.dy ** 2)
            du.set(x, y, c.dt * c.k * (uxx + uyy))
    for i in range(0, u.shape[0]*u.shape[1] - 1):
        u.flat_set(i, u.flat_get(i) + du.flat_get(i))


def solve_indexed(u, c):
    u.set(c.x0, c.y0, 1)
    for t in np.arange(0, c.tfinal, c.dt):
        print("t=" + "{:8.6f}".format(t))
        time_step_indexed(u, c)
    return t


def main():
    nx = 100
    ny = 100
    nt = 100
    rate = 64
    cache = 4
    c = Constants(nx, ny, nt+1) 
    u = zfpy.zfparray2d(nx, ny, rate)
    rate = u.rate()
    t = solve_indexed(u, c)
    usum = total(u) 
    uerr = error(u, c, t)
    print("rate=" + str(rate) + " sum=" + str(usum) + " error=" + str(uerr))


if __name__ == "__main__":
    main()
