from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import xarray as xr
import cupy as cu
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
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


@dataclass
class MonteCarloResult:
    fluence: np.float32[:, :, :, ::1]
    phi_td: np.float64[:, ::1]
    phi_phase: np.float64[::1]
    phi_dist: np.float64[:, :, ::1]
    photon_counter: np.uint64[:, ::1]


@cu.memoize(for_each_device=True)
def load_module():
    return cu.RawModule(path="src/kernel.ptx")


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


res = monte_carlo(
    Specification(
        nphoton=10,
        voxel_dim=(1.0, 1.0, 1.0),
        lifetime_max=5000,
        dt=100,
        lightspeed=0.2998,
        freq=110e-6,
    ),
    source=Pencil(
        position=Vector(8, 100, 100),
        direction=Vector(1, 0, 0)
    ),
    states=[
        State(mua=0, mus=0, g=1, n=1),
        State(mua=1e-2, mus=3, g=0.9, n=1.4),
    ],
    detectors=[
        Detector(position=Vector(0, 100, 100), radius=10),
        Detector(position=Vector(0, 100, 100), radius=20),
        Detector(position=Vector(0, 100, 100), radius=30),
    ],
    media=np.ones((200, 200, 200), np.uint8),
)

print(res)

for k, v in res.data_vars.items():
    v.data = cu.asnumpy(v.data)

ndet = len(res.coords["detector"])
ntof = len(res.coords["time"])

# Plot Photon Counter
print(res["Photons"])
print(res["Photons"].sum(dim="time"))
fig, axs = plt.subplots(ndet, 2)
for ((ax1, ax2), ps) in zip(axs, res["Photons"]):
    ax1.bar(np.arange(ntof), ps, log=True)
    ax2.bar(np.arange(ntof), ps)
fig.tight_layout()
fig.savefig("photons.png", dpi=300)

# Plot Phi Time Domain
print('phi_td', np.nansum(res["PhiTD"], axis=1))
print('phi_td', np.nansum(res["PhiTD"]))
fig, axs = plt.subplots(1, ndet, sharey='all')
for (ax, td) in zip(axs, res["PhiTD"]):
    ax.semilogy(td, '*--')
fig.tight_layout()
fig.savefig("phitd.png", dpi=300)

# Plot Phi Distr
print('phi_dist', np.nansum(res["PhiDist"], axis=(1, 2)))
fig, axs = plt.subplots(2, ndet)
for ((ax1, ax2), distr) in zip(axs.T, res["PhiDist"]):
    ax1.matshow(distr.T, extent=[0, 8, 0, 1])
    ax2.matshow(np.log(distr).T, extent=[0, 8, 0, 1])
fig.tight_layout()
fig.savefig("phidistr.png", dpi=300)

# Plot Phase
fig, ax = plt.subplots(1)
ax.plot(np.rad2deg(res["PhiPhase"]), '*--')
fig.tight_layout()
fig.savefig("phase.png", dpi=300)


# Plot Fd
fd = np.exp(1j * res["PhiPhase"]) * res["PhiTD"].sum(axis=1)
print(res["PhiPhase"])
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(fd.real)
ax2.plot(fd.imag)
fig.tight_layout()
fig.savefig("fd.png", dpi=300)

# Plot Fluence
sliced = res["Fluence"].data[:, :, 100]
_, _, nt = sliced.shape

sliced = np.log(sliced)
sliced[~np.isfinite(sliced)] = np.nan
norm = colors.Normalize(vmin=np.nanmin(sliced), vmax=np.nanmax(sliced))

print(np.nanmin(sliced), np.nanmax(sliced))

fig, ax = plt.subplots()

mat = ax.matshow(sliced[:, :, 0].T, origin='lower', animated=True, norm=norm)
clb = fig.colorbar(mat)


def animate(i):
    mat.set_data(sliced[:, :, i].T)
    return mat,


anim = FuncAnimation(fig, animate, frames=nt, interval=100, blit=True)
anim.save("fluence.gif", writer='imagemagick', dpi=150)
