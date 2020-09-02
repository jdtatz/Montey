from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from numba.cuda.random import xoroshiro128p_dtype, init_xoroshiro128p_states_cpu
from typing import Tuple, Sequence, TypeVar, Generic
from numbers import Real

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


def monte_carlo(spec: Specification, source: Source, states: Sequence[State], detectors: Sequence[Detector],
                media: np.uint8[:, :, ::1], seed: int = 12345, nwarp: int = 4, nblock: int = 512,
                gpu_id: int = 0) -> MonteCarloResult:
    nthread = nblock * nwarp * 32
    pcount = nthread * spec.nphoton
    print(nthread, pcount)

    rng_states = np.empty(shape=nthread, dtype=xoroshiro128p_dtype)
    init_xoroshiro128p_states_cpu(rng_states, seed, subsequence_start=0)
    print(rng_states)
    rng_states = rng_states.view(np.uint64)

    ndet = len(detectors)
    ntof = int(np.ceil(spec.lifetime_max / spec.dt))
    nmedia = len(states) - 1
    fluence = np.zeros((*media.shape, ntof), np.float32)
    phi_td = np.zeros((nthread, ndet, ntof), np.float32)
    phi_phase = np.zeros((nthread, ndet), np.float32)
    phi_dist = np.zeros((nthread, ndet, ntof, nmedia), np.float32)
    photon_counter = np.zeros((nthread, ndet, ntof), np.uint64)

    print(states)
    print(detectors)

    print(states[0].as_record(np.float32))
    print(detectors[0].as_record(np.float32))

    states = np.stack([s.as_record(np.float32) for s in states])
    detectors = np.stack([d.as_record(np.float32) for d in detectors])

    print(states)
    print(detectors)
    print(spec.as_record(np.float32))
    print(source.as_record(np.float32))

    dev = drv.Device(gpu_id)
    print(f"Using gpu: {dev.name()}")
    ctxt: drv.Context = dev.make_context()

    args = [
        gpuarray.to_gpu(spec.as_record(np.float32).view(np.uint32)),
        gpuarray.to_gpu(source.as_record(np.float32).view(np.uint32)),
        np.uint32(nmedia),
        gpuarray.to_gpu(states.view(np.uint32)),
        np.uint32(media.shape[0]),
        np.uint32(media.shape[1]),
        np.uint32(media.shape[2]),
        drv.In(media),
        drv.In(rng_states),
        np.uint32(ndet),
        gpuarray.to_gpu(detectors.view(np.uint32)),
        drv.InOut(fluence),
        drv.InOut(phi_td),
        drv.InOut(phi_phase),
        drv.InOut(phi_dist),
        drv.InOut(photon_counter),
    ]
    with open("src/kernel.ptx", "rb") as f:
        module = drv.module_from_buffer(f.read())
    kernel = module.get_function(source.kernel_name())

    smem = nwarp * 32 * nmedia * 4
    dt = kernel(*args, block=(nwarp * 32, 1, 1), grid=(nblock, 1), shared=smem, time_kernel=True)
    dt *= 1000
    print(f"Time: {dt}ms")
    print(f"Photon count: {pcount}")
    print(f"Throughput: {pcount / dt} photons/ms")

    ctxt.pop()
    return MonteCarloResult(
        fluence=fluence,
        phi_td=phi_td.sum(axis=0, dtype=np.float64) / pcount,
        phi_phase=phi_phase.sum(axis=0, dtype=np.float64) / phi_td.sum(axis=(0, 2), dtype=np.float64),
        phi_dist=phi_dist.sum(axis=0, dtype=np.float64) / phi_td.sum(axis=(0, 2), dtype=np.float64)[:, None, None],
        photon_counter=photon_counter.sum(axis=0, dtype=np.uint64),
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

# Plot Photon Counter
fig, axs = plt.subplots(res.photon_counter.shape[0], 2)
for ((ax1, ax2), ps) in zip(axs, res.photon_counter):
    ax1.bar(np.arange(len(ps)), ps, log=True)
    ax2.bar(np.arange(len(ps)), ps)
fig.tight_layout()
fig.savefig("photons.png", dpi=300)

# Plot Phi Time Domain
print('phi_td', np.nansum(res.phi_td, axis=1))
print('phi_td', np.nansum(res.phi_td))
fig, axs = plt.subplots(1, res.phi_td.shape[0], sharey='all')
for (ax, td) in zip(axs, res.phi_td):
    ax.semilogy(td, '*--')
fig.tight_layout()
fig.savefig("phitd.png", dpi=300)

# Plot Phi Distr
print('phi_dist', np.nansum(res.phi_dist, axis=(1, 2)))
fig, axs = plt.subplots(2, res.phi_dist.shape[0])
for ((ax1, ax2), distr) in zip(axs.T, res.phi_dist):
    ax1.matshow(distr.T, extent=[0, 8, 0, 1])
    ax2.matshow(np.log(distr).T, extent=[0, 8, 0, 1])
fig.tight_layout()
fig.savefig("phidistr.png", dpi=300)

# Plot Phase
fig, ax = plt.subplots(1)
ax.plot(np.rad2deg(res.phi_phase), '*--')
fig.tight_layout()
fig.savefig("phase.png", dpi=300)


# Plot Fd
fd = np.exp(1j * res.phi_phase) * res.phi_td.sum(axis=1)
print(res.phi_phase)
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(fd.real)
ax2.plot(fd.imag)
fig.tight_layout()
fig.savefig("fd.png", dpi=300)

# Plot Fluence
sliced = res.fluence[:, :, 100]
_, _, nt = sliced.shape

sliced = np.log(sliced)
sliced[~np.isfinite(sliced)] = np.nan
norm = colors.Normalize(vmin=np.nanmin(sliced), vmax=np.nanmax(sliced))

print(np.nanmin(sliced), np.nanmax(sliced))
print(res.photon_counter)
print(res.photon_counter.sum(axis=1))

fig, ax = plt.subplots()

mat = ax.matshow(sliced[:, :, 0].T, origin='lower', animated=True, norm=norm)
clb = fig.colorbar(mat)


def animate(i):
    mat.set_data(sliced[:, :, i].T)
    return mat,


anim = FuncAnimation(fig, animate, frames=nt, interval=100, blit=True)
anim.save("fluence.gif", writer='imagemagick', dpi=150)
