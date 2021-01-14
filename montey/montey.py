from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.resources import path
from numbers import Real
from typing import Tuple, Sequence, TypeVar, Generic, Optional, Union, Type

import cupy as cu
import numpy as np
import xarray as xr
from numba.cuda.random import create_xoroshiro128p_states
from pint import UnitRegistry, get_application_registry

ArrayScalar = TypeVar('ArrayScalar', bound=np.generic)
DTypeLike = Union[np.dtype, None, Type[ArrayScalar]]

T = TypeVar("T", bound=Real)


@dataclass
class Vector(Generic[T]):
    x: T
    y: T
    z: T

    def __iter__(self):
        return self.x, self.y, self.z

    @staticmethod
    def dtype(scalar: DTypeLike) -> np.dtype:
        return np.dtype([("x", scalar), ("y", scalar), ("z", scalar)])

    def as_record(self, scalar: DTypeLike) -> np.recarray:
        return np.rec.array([(self.x, self.y, self.z)], dtype=self.dtype(scalar))

    def __neg__(self) -> Vector[T]:
        return Vector(x=-self.x, y=-self.y, z=-self.z)

    def __add__(self, rhs: Vector[T]) -> Vector[T]:
        return Vector(x=self.x + rhs.x, y=self.y + rhs.y, z=self.z + rhs.z)

    def __sub__(self, rhs: Vector[T]) -> Vector[T]:
        return Vector(x=self.x - rhs.x, y=self.y - rhs.y, z=self.z - rhs.z)

    def __mul__(self, rhs: T) -> Vector[T]:
        return Vector(x=self.x * rhs, y=self.y * rhs, z=self.z * rhs)

    def __div__(self, rhs: T) -> Vector[T]:
        return Vector(x=self.x / rhs, y=self.y / rhs, z=self.z / rhs)

    def dot(self, rhs: Vector[T]) -> T:
        return self.x * rhs.x + self.y * rhs.y + self.z * rhs.z

    def cross(self, rhs: Vector[T]) -> Vector[T]:
        return Vector(
            x=self.y * rhs.z - self.z * rhs.y,
            y=self.z * rhs.x - self.x * rhs.z,
            z=self.x * rhs.y - self.y * rhs.x,
        )


@dataclass
class State:
    mua: Real
    mus: Real
    g: Real
    n: Real

    @staticmethod
    def dtype(scalar: DTypeLike) -> np.dtype:
        return np.dtype(
            [("mua", scalar), ("mus", scalar), ("g", scalar), ("n", scalar)]
        )

    def as_record(self, scalar: DTypeLike) -> np.recarray:
        return np.rec.array(
            [(self.mua, self.mus, self.g, self.n)], dtype=self.dtype(scalar)
        )


@dataclass
class Detector:
    position: Vector[Real]
    radius: Real

    @staticmethod
    def dtype(scalar: DTypeLike) -> np.dtype:
        return np.dtype([("position", Vector.dtype(scalar)), ("radius", scalar)])

    def as_record(self, scalar: DTypeLike) -> np.recarray:
        return np.rec.array(
            [(self.position.as_record(scalar), self.radius)], dtype=self.dtype(scalar)
        )


@dataclass
class Specification:
    nphoton: int
    voxel_dim: Tuple[Real, Real, Real]
    lifetime_max: Real
    dt: Real
    lightspeed: Real
    freq: Real

    @staticmethod
    def dtype(scalar: DTypeLike) -> np.dtype:
        return np.dtype(
            [
                ("nphoton", np.uint32),
                ("voxel_dim", Vector.dtype(scalar)),
                ("lifetime_max", scalar),
                ("dt", scalar),
                ("lightspeed", scalar),
                ("freq", scalar),
            ]
        )

    def as_record(self, scalar: DTypeLike) -> np.recarray:
        d = Vector(*self.voxel_dim).as_record(scalar)
        return np.rec.array(
            [(self.nphoton, d, self.lifetime_max, self.dt, self.lightspeed, self.freq)],
            dtype=self.dtype(scalar),
        )


class Source(ABC):
    @abstractmethod
    def kernel_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def dtype(self, scalar: DTypeLike) -> np.dtype:
        raise NotImplementedError

    @abstractmethod
    def as_record(self, scalar: DTypeLike) -> np.recarray:
        raise NotImplementedError


S = TypeVar("S", bound=Source)


class SourceArray(Source, Generic[S]):
    def __init__(self, sources: Sequence[S]):
        if len(sources) == 0:
            raise TypeError(
                "SourceArray sources must be a non-empty sequence of sources"
            )
        s = type(sources[0])
        if not all(s == type(src) for src in sources):
            raise TypeError("SourceArray sources must all be the same type")
        self.sources = sources

    def kernel_name(self) -> str:
        return f"{self.sources[0].kernel_name()}_array"

    def dtype(self, scalar: DTypeLike) -> np.dtype:
        return self.sources[0].dtype(scalar)

    def as_record(self, scalar: DTypeLike) -> np.recarray:
        return np.stack([src.as_record(scalar) for src in self.sources]).view(np.recarray)


@dataclass
class Pencil(Source):
    position: Vector[Real]
    direction: Vector[Real]

    def kernel_name(self) -> str:
        return "pencil"

    def dtype(self, scalar: DTypeLike) -> np.dtype:
        return np.dtype(
            [("position", Vector.dtype(scalar)), ("direction", Vector.dtype(scalar))]
        )

    def as_record(self, scalar: DTypeLike) -> np.rec.array:
        return np.rec.array(
            [(self.position.as_record(scalar), self.direction.as_record(scalar))],
            dtype=self.dtype(scalar),
        )


class Disk(Source):
    def __init__(self, position: Vector[Real], direction: Vector[Real], radius: Real):
        self.position = position
        self.direction = direction
        self.radius = radius
        z = Vector(0.0, 0.0, 1.0)
        x_vec = z - direction * direction.dot(z)
        y_vec = direction.cross(x_vec)
        self.orthonormal_basis = (x_vec, y_vec)

    def kernel_name(self) -> str:
        return "disk"

    def dtype(self, scalar: DTypeLike) -> np.dtype:
        return np.dtype(
            [
                ("position", Vector.dtype(scalar)),
                ("direction", Vector.dtype(scalar)),
                ("orthonormal_basis", (Vector.dtype(scalar), 2)),
                ("radius", scalar),
            ]
        )

    def as_record(self, scalar: DTypeLike) -> np.rec.array:
        return np.rec.array(
            [
                (
                    self.position.as_record(scalar),
                    self.direction.as_record(scalar),
                    (
                        self.orthonormal_basis[0].as_record(scalar),
                        self.orthonormal_basis[1].as_record(scalar),
                    ),
                    self.radius,
                )
            ],
            dtype=self.dtype(scalar),
        )


@cu.memoize(for_each_device=True)
def load_module():
    with path(__package__, "kernel.ptx") as kernel_ptx_path:
        return cu.RawModule(path=str(kernel_ptx_path.absolute()))


@cu.memoize(for_each_device=True)
def load_kernel(kernel_name: str):
    return load_module().get_function(kernel_name)


def monte_carlo(
    spec: Specification,
    source: Source,
    states: Sequence[State],
    detectors: Sequence[Detector],
    media: np.uint8[:, :, ::1],
    seed: int = 12345,
    nwarp: int = 4,
    nblock: int = 512,
    ureg: Optional[UnitRegistry] = None,
) -> xr.Dataset:
    if ureg is None:
        ureg = get_application_registry()
    nthread = nblock * nwarp * 32
    pcount = nthread * spec.nphoton

    rng_states = create_xoroshiro128p_states(nthread, seed)

    ndet = len(detectors)
    ntof = int(np.ceil(spec.lifetime_max / spec.dt))
    nmedia = len(states) - 1
    fluence = cu.zeros((*media.shape, ntof), np.float32)
    phi_td = cu.zeros((nthread, ndet, ntof), np.float32)
    phi_phase = cu.zeros((nthread, ndet), np.float32)
    phi_dist = cu.zeros((nthread, ndet, ntof, nmedia), np.float32)
    mom_dist = cu.zeros((nthread, ndet, ntof, nmedia), np.float32)
    photon_counter = cu.zeros((nthread, ndet, ntof), np.uint64)

    args = (
        cu.asarray(spec.as_record(np.float32).view(np.uint32)),
        cu.asarray(source.as_record(np.float32).view(np.uint32)),
        np.uint32(nmedia),
        cu.asarray(np.stack([s.as_record(np.float32) for s in states]).view(np.uint32)),
        np.uint32(media.shape[0]),
        np.uint32(media.shape[1]),
        np.uint32(media.shape[2]),
        cu.asarray(media),
        cu.asarray(rng_states),
        np.uint32(ndet),
        cu.asarray(np.stack([d.as_record(np.float32) for d in detectors]).view(np.uint32)),
        fluence,
        phi_td,
        phi_phase,
        phi_dist,
        mom_dist,
        photon_counter,
    )

    kernel = load_kernel(source.kernel_name())
    start_event = cu.cuda.Event()
    start_event.record()
    kernel(
        args=args,
        block=(nwarp * 32, 1, 1),
        grid=(nblock, 1),
        shared_mem=nwarp * 32 * 2 * nmedia * 4,
    )
    end_event = cu.cuda.Event(block=True)
    end_event.record()
    end_event.synchronize()
    dt = cu.cuda.get_elapsed_time(start_event, end_event)
    print(f"Time: {dt}ms")
    print(f"Photon count: {pcount}")
    print(f"Throughput: {pcount / dt} photons/ms")

    return xr.Dataset(
        {
            "Photons": (
                ["detector", "time"],
                photon_counter.sum(axis=0, dtype=np.uint64),
            ),
            "Phi": (
                ["detector", "time"],
                phi_td.sum(axis=0, dtype=np.float64) / pcount,
                {"long_name": "Φ"},
            ),
            "PhiPhase": (
                ["detector"],
                phi_phase.sum(axis=0, dtype=np.float64)
                / phi_td.sum(axis=(0, 2), dtype=np.float64),
                {"units": ureg.rad, "long_name": "Φ Phase"},
            ),
            "PhiDist": (
                ["detector", "time", "layer"],
                phi_dist.sum(axis=0, dtype=np.float64)
                / phi_td.sum(axis=(0, 2), dtype=np.float64)[:, None, None],
                {"long_name": "Φ Distribution"},
            ),
            "MomDist": (
                ["detector", "time", "layer"],
                mom_dist.sum(axis=0, dtype=np.float64)
                / phi_td.sum(axis=(0, 2), dtype=np.float64)[:, None, None],
                {"long_name": "Φ-Weighted Momentum Transfer Distribution"},
            ),
            "Fluence": (["x", "y", "z", "time"], fluence),
        },
        coords={
            "time": (
                ["time"],
                (np.arange(ntof) + 0.5) * spec.dt,
            ),
        },
    )
