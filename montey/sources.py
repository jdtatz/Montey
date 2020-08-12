import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Collection
from numba.cuda.random import xoroshiro128p_type, xoroshiro128p_uniform_float32
from .vector import Vector

LaunchFunctionType = Callable[[xoroshiro128p_type[::1], int], Tuple[Vector, Vector]]
Pi = np.float32(np.pi)
Pi_2 = np.float32(np.pi * 2)


class Source(ABC):
    @abstractmethod
    def create_launch_function(self) -> LaunchFunctionType:
        pass


class Pencil(Source):
    def __init__(self, srcpos: Vector, srcdir: Vector):
        self.srcpos = srcpos
        self.srcdir = srcdir

    def create_launch_function(self) -> LaunchFunctionType:
        srcpos = self.srcpos.numba_compat()
        srcdir = self.srcdir.numba_compat()

        def launch(_rng, _idx):
            return srcpos, srcdir
        return launch


class UniformDisk(Source):
    def __init__(self, srcpos: Vector, srcdir: Vector, radius: float):
        self.srcpos = srcpos
        self.srcdir = srcdir
        self.radius = radius

    def create_launch_function(self) -> LaunchFunctionType:
        srcpos = self.srcpos.numba_compat()
        srcdir = self.srcdir.numba_compat()
        radius = np.float32(self.radius)
        # Use Gram-Schmidt Process to generate orthogonal vector to disk normal
        z = Vector(0.0, 0.0, 1.0).numba_compat()
        n1 = (z - srcdir * (srcdir @ z)).numba_compat()
        # use cross product to generate second orthogonal vector to disk normal
        n2 = Vector(*np.cross(srcdir, n1)).numba_compat()

        def launch(rng, idx):
            r = radius * math.sqrt(xoroshiro128p_uniform_float32(rng, idx))
            theta = xoroshiro128p_uniform_float32(rng, idx) * Pi_2
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            p = srcpos + n1 * x + n2 * y
            return p, srcdir
        return launch


class UniformDiskArray(Source):
    def __init__(self, srcpos: Collection[Vector], srcdir: Collection[Vector], radius: Collection[float]):
        assert len(srcpos) == len(srcdir)
        assert len(srcpos) == len(radius)
        self.srcpos = srcpos
        self.srcdir = srcdir
        self.radius = radius

    def create_launch_function(self) -> LaunchFunctionType:
        srcpos = tuple(p.numba_compat() for p in self.srcpos)
        srcdir = tuple(v.numba_compat() for v in self.srcdir)
        radii = tuple(np.float32(r) for r in self.radius)
        N = np.int32(len(radii))
        # Use Gram-Schmidt Process to generate orthogonal vector to disk normal
        z = Vector(0.0, 0.0, 1.0)
        n1s = tuple((z - n * (n @ z)).numba_compat() for n in self.srcdir)
        # use cross product to generate second orthogonal vector to disk normal
        n2s = tuple(Vector(*np.cross(n, n1)).numba_compat() for n, n1 in zip(srcdir, n1s))

        def launch(rng, idx):
            i = np.int32(xoroshiro128p_uniform_float32(rng, idx) * N)
            r = radii[i] * math.sqrt(xoroshiro128p_uniform_float32(rng, idx))
            theta = xoroshiro128p_uniform_float32(rng, idx) * Pi_2
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            p = srcpos[i] + n1s[i] * x + n2s[i] * y
            return p, srcdir[i]
        return launch
