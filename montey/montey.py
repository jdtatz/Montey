from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.resources import path
import numpy as np
import xarray as xr
import cupy as cu
from numba.cuda.random import create_xoroshiro128p_states
from typing import Tuple, Sequence, TypeVar, Generic, Optional
from numbers import Real
from pint import UnitRegistry, get_application_registry

T = TypeVar("T")


@dataclass
class Vector(Generic[T]):
    x: T
    y: T
    z: T

    def __iter__(self):
        return self.x, self.y, self.z

    @staticmethod
    def dtype(scalar: np.dtype) -> np.dtype:
        return np.dtype([("x", scalar), ("y", scalar), ("z", scalar)])

    def as_record(self, scalar: np.dtype) -> np.rec.array:
        return np.rec.array([(self.x, self.y, self.z)], dtype=self.dtype(scalar))


@dataclass
class State:
    mua: Real
    mus: Real
    g: Real
    n: Real

    @staticmethod
    def dtype(scalar: np.dtype) -> np.dtype:
        return np.dtype([("mua", scalar), ("mus", scalar), ("g", scalar), ("n", scalar)])

    def as_record(self, scalar: np.dtype) -> np.rec.array:
        return np.rec.array([(self.mua, self.mus, self.g, self.n)], dtype=self.dtype(scalar))


@dataclass
class Detector:
    position: Vector[Real]
    radius: Real

    @staticmethod
    def dtype(scalar: np.dtype) -> np.dtype:
        return np.dtype([("position", Vector.dtype(scalar)), ("radius", scalar)])

    def as_record(self, scalar: np.dtype) -> np.rec.array:
        return np.rec.array([(self.position.as_record(scalar), self.radius)], dtype=self.dtype(scalar))


@dataclass
class Specification:
    nphoton: int
    voxel_dim: Tuple[Real, Real, Real]
    lifetime_max: Real
    dt: Real
    lightspeed: Real
    freq: Real

    @staticmethod
    def dtype(scalar: np.dtype) -> np.dtype:
        return np.dtype([
            ("nphoton", np.uint32),
            ("voxel_dim", Vector.dtype(scalar)),
            ("lifetime_max", scalar),
            ("dt", scalar),
            ("lightspeed", scalar),
            ("freq", scalar),
        ])

    def as_record(self, scalar: np.dtype) -> np.rec.array:
        d = Vector(*self.voxel_dim).as_record(scalar)
        return np.rec.array([(self.nphoton, d, self.lifetime_max, self.dt, self.lightspeed, self.freq)], dtype=self.dtype(scalar))


class Source(ABC):
    @abstractmethod
    def kernel_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def dtype(self, scalar: np.dtype) -> np.dtype:
        raise NotImplementedError

    @abstractmethod
    def as_record(self, scalar: np.dtype) -> np.rec.array:
        raise NotImplementedError


@dataclass
class Pencil(Source):
    position: Vector[Real]
    direction: Vector[Real]

    def kernel_name(self) -> str:
        return "pencil"

    def dtype(self, scalar: np.dtype) -> np.dtype:
        return np.dtype([("position", Vector.dtype(scalar)), ("direction", Vector.dtype(scalar))])

    def as_record(self, scalar: np.dtype) -> np.rec.array:
        return np.rec.array([(self.position.as_record(scalar), self.direction.as_record(scalar))], dtype=self.dtype(scalar))


@cu.memoize(for_each_device=True)
def load_module():
    with path(__package__, "kernel.ptx") as kernel_ptx_path:
        return cu.RawModule(path=str(kernel_ptx_path.absolute()))


@cu.memoize(for_each_device=True)
def load_kernel(kernel_name: str):
    return load_module().get_function(kernel_name)


def monte_carlo(spec: Specification, source: Source, states: Sequence[State], detectors: Sequence[Detector],
                media: np.uint8[:, :, ::1], seed: int = 12345, nwarp: int = 4, nblock: int = 512,
                ureg: Optional[UnitRegistry] = None) -> xr.Dataset:
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
    photon_counter = cu.zeros((nthread, ndet, ntof), np.uint64)

    states = np.stack([s.as_record(np.float32) for s in states])
    detectors = np.stack([d.as_record(np.float32) for d in detectors])

    args = (
        cu.asarray(spec.as_record(np.float32).view(np.uint32)),
        cu.asarray(source.as_record(np.float32).view(np.uint32)),
        np.uint32(nmedia),
        cu.asarray(states.view(np.uint32)),
        np.uint32(media.shape[0]),
        np.uint32(media.shape[1]),
        np.uint32(media.shape[2]),
        cu.asarray(media),
        cu.asarray(rng_states),
        np.uint32(ndet),
        cu.asarray(detectors.view(np.uint32)),
        fluence,
        phi_td,
        phi_phase,
        phi_dist,
        photon_counter,
    )

    kernel = load_kernel(source.kernel_name())
    start_event = cu.cuda.Event()
    start_event.record()
    kernel(args=args, block=(nwarp * 32, 1, 1), grid=(nblock, 1), shared_mem=nwarp * 32 * nmedia * 4)
    end_event = cu.cuda.Event(block=True)
    end_event.record()
    end_event.synchronize()
    dt = cu.cuda.get_elapsed_time(start_event, end_event)
    print(f"Time: {dt}ms")
    print(f"Photon count: {pcount}")
    print(f"Throughput: {pcount / dt} photons/ms")

    return xr.Dataset(
        {
            "Photons": (["detector", "time"], photon_counter.sum(axis=0, dtype=np.uint64)),
            "PhiTD": (["detector", "time"], phi_td.sum(axis=0, dtype=np.float64) / pcount, {"long_name": "Φ"}),
            "PhiPhase": (["detector"], phi_phase.sum(axis=0, dtype=np.float64) / phi_td.sum(axis=(0, 2), dtype=np.float64), {"units": ureg.rad, "long_name": "Φ Phase"}),
            "PhiDist": (["detector", "time", "layer"], phi_dist.sum(axis=0, dtype=np.float64) / phi_td.sum(axis=(0, 2), dtype=np.float64)[:, None, None], {"long_name": "Φ Distribution"}),
            "Fluence": (["x", "y", "z", "time"], fluence),
        },
        coords={
            "time": (["time"], (np.arange(ntof) + 0.5) * spec.dt, {"units": ureg.second}),
        }
    )
