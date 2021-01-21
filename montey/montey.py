from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass, fields
from importlib.resources import path
from numbers import Real
from typing import Tuple, List, Sequence, TypeVar, Generic, Optional, Union, Type, Dict, get_type_hints
try:
    from typing import get_args, get_origin
except ImportError:
    def get_args(ty):
        # Pyton < 3.8 compat polyfill
        return getattr(ty, "__args__", default=())

    def get_origin(ty):
        # Pyton < 3.8 compat polyfill
        return getattr(ty, "__origin__", default=None)

import cupy as cu
import numpy as np
import xarray as xr
from numba.cuda.random import create_xoroshiro128p_states
from pint import UnitRegistry, get_application_registry

ArrayScalar = TypeVar('ArrayScalar', bound=np.generic)
DTypeLike = Union[np.dtype, None, Type[ArrayScalar]]


def _to_dtype(v, ty, generic_args: Tuple[DTypeLike, ...] = ()) -> np.dtype:
    orig = get_origin(ty)
    if orig is not None:
        return _to_dtype(v, orig, get_args(ty))
    
    if hasattr(v, "_to_dtype"):
        return v._to_dtype(generic_args)
    if hasattr(v, "resolve_generic_args"):
        generic_args = v.resolve_generic_args()

    if ty is int:
        assert len(generic_args) == 0
        return np.uint32
    if issubclass(ty, Real):
        assert len(generic_args) == 0
        return np.float32
    if issubclass(ty, Tuple):
        if len(generic_args) == 0:
            raise TypeError("Only Non-Empty Tuples are allowed")
        elem_ty = generic_args[0]
        if not all(elem_ty == arg_ty for arg_ty in generic_args[1:]):
            raise TypeError("Only Homogenous Tuples are allowed")
        return np.dtype((_to_dtype(v[0], elem_ty), len(generic_args)))
    if issubclass(ty, Sequence):
        elem_ty, = generic_args
        return np.dtype((_to_dtype(v[0], elem_ty), len(v)))
    if is_dataclass(ty):
        generic_params = getattr(ty, "__parameters__", ())
        assert len(generic_params) == len(generic_args)
        if len(generic_params) != len(generic_args):
            raise TypeError(f"Dataclass has {len(generic_params)} generic parameters, but {len(generic_args)} generic arguments were given.")
        generics = dict(zip(generic_params, generic_args))
        resolved = {f: generics.get(ty, ty) for f, ty in get_type_hints(ty).items()}
        return np.dtype([
            (f.name, _to_dtype(getattr(v, f.name), resolved[f.name]))
            for f in fields(ty)
        ])
    raise NotImplementedError


def _to_record(v, ty, generic_args: Tuple[DTypeLike, ...] = ()) -> np.dtype:
    orig = get_origin(ty)
    if orig is not None:
        return _to_record(v, orig, get_args(ty))

    if hasattr(v, "_to_record"):
        return v._to_record(generic_args)
    if hasattr(v, "resolve_generic_args"):
        generic_args = v.resolve_generic_args()

    if ty is int:
        return v
    if issubclass(ty, Real):
        return v
    if issubclass(ty, Tuple):
        elem_ty = generic_args[0]
        return tuple(_to_record(vi, elem_ty) for vi in v)
    if issubclass(ty, Sequence):
        elem_ty = generic_args[0]
        return tuple(_to_record(vi, elem_ty) for vi in v)
    if is_dataclass(ty):
        generic_params = getattr(ty, "__parameters__", ())
        if len(generic_params) != len(generic_args):
            raise TypeError(f"Dataclass has {len(generic_params)} generic parameters, but {len(generic_args)} generic arguments were given.")
        generics = dict(zip(generic_params, generic_args))
        resolved = {f: generics.get(ty, ty) for f, ty in get_type_hints(ty).items()}
        return np.rec.array([tuple(
            _to_record(getattr(v, f.name), resolved[f.name])
            for f in fields(ty)
        )], dtype=_to_dtype(v, ty, generic_args))
    raise NotImplementedError


class CudaCompat(ABC):
    # @abstractmethod
    def to_dtype(self, generic_args: Tuple[DTypeLike, ...] = ()) -> np.dtype:
        return _to_dtype(self, type(self), generic_args)

    # @abstractmethod
    def to_record(self, generic_args: Tuple[DTypeLike, ...] = ()) -> np.recarray:
        return _to_record(self, type(self), generic_args)


T = TypeVar("T", bound=Real)



@dataclass
class Vector(CudaCompat, Generic[T]):
    x: T
    y: T
    z: T

    def __iter__(self):
        return iter((self.x, self.y, self.z))

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
class State(CudaCompat):
    mua: Real
    mus: Real
    g: Real
    n: Real


@dataclass
class Detector(CudaCompat):
    position: Vector[Real]
    radius: Real


@dataclass
class Specification(CudaCompat):
    nphoton: int
    lifetime_max: Real
    dt: Real
    lightspeed: Real
    freq: Real

class Source(ABC):
    @abstractmethod
    def kernel_name(self) -> str:
        raise NotImplementedError

    def extra_args(self) -> Sequence[np.scalar]:
        return ()

S = TypeVar("S", bound=Source)


class SourceArray(Source, CudaCompat, Sequence[S], Generic[S]):
    def __init__(self, sources: Sequence[S]):
        if len(sources) == 0:
            raise TypeError(
                "SourceArray sources must be a non-empty sequence of sources"
            )
        s = type(sources[0])
        if not all(s == type(src) for src in sources):
            raise TypeError("SourceArray sources must all be the same type")
        self.sources = sources

    def __iter__(self):
        return iter(self.sources)

    def kernel_name(self) -> str:
        return f"{self.sources[0].kernel_name()}_array"

    def resolve_generic_args(self):
        return type(self.sources[0]),

    def extra_args(self) -> Sequence[np.scalar]:
        return np.uint32(len(self.sources)),


@dataclass
class Pencil(Source, CudaCompat):
    position: Vector[Real]
    direction: Vector[Real]

    def kernel_name(self) -> str:
        return "pencil"


@dataclass
class Disk(Source, CudaCompat):
    position: Vector[Real]
    direction: Vector[Real]
    orthonormal_basis: Tuple[Vector[Real], Vector[Real]] = field(init=False)
    radius: Real = field()

    def __post_init__(self):
        z = Vector(0.0, 0.0, 1.0)
        x_vec = z - self.direction * self.direction.dot(z)
        y_vec = self.direction.cross(x_vec)
        self.orthonormal_basis = (x_vec, y_vec)

    def kernel_name(self) -> str:
        return "disk"


class Geometry(ABC):
    @abstractmethod
    def kernel_prefix(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fluence_dim(self, time_dim: int) -> Tuple[List[str], Tuple[int, ...]]:
        raise NotImplementedError

    def extra_args(self) -> Sequence[np.scalar]:
        return ()


@dataclass
class VoxelGeometry(Geometry, CudaCompat):
    voxel_dim: Tuple[Real, Real, Real]
    media_dim: Tuple[int, int, int]

    @staticmethod
    def kernel_prefix() -> str:
        return ''

    def fluence_dim(self, time_dim: int) -> Tuple[List[str], Tuple[int, ...]]:
        return ["x", "y", "z", "time"], (*self.media_dim, time_dim)



@dataclass
class AxialSymetricGeometry(Geometry, CudaCompat):
    voxel_dim: Tuple[Real, Real]
    media_dim: Tuple[int, int]

    @staticmethod
    def kernel_prefix() -> str:
        return 'axial_'

    def fluence_dim(self, time_dim: int) -> Tuple[List[str], Tuple[int, ...]]:
        return ["r", "z", "time"], (*self.media_dim, time_dim)



@dataclass
class FreeSpaceGeometry(Geometry, CudaCompat):
    @staticmethod
    def kernel_prefix() -> str:
        return 'free_space_'

    def fluence_dim(self, time_dim: int) -> Tuple[List[str], Tuple[int, ...]]:
        return [], ()


G = TypeVar("G", bound=Geometry)


@dataclass
class LayeredGeometry(Geometry, CudaCompat, Generic[G]):
    inner: G
    layers: Sequence[Real]

    def kernel_prefix(self) -> str:
        return f'layered_{self.inner.kernel_prefix()}'

    def fluence_dim(self, time_dim: int) -> Tuple[List[str], Tuple[int, ...]]:
        return self.inner.fluence_dim(time_dim)

    def resolve_generic_args(self):
        return type(self.inner),

    def extra_args(self) -> Sequence[np.scalar]:
        return np.uint32(len(self.layers)),


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
    geom: Geometry,
    media: np.uint8[::1],
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
    fluence_keys, fluence_shape = geom.fluence_dim(ntof)
    fluence = cu.zeros(fluence_shape, np.float32)
    phi_td = cu.zeros((nthread, ndet, ntof), np.float32)
    phi_phase = cu.zeros((nthread, ndet), np.float32)
    phi_dist = cu.zeros((nthread, ndet, ntof, nmedia), np.float32)
    mom_dist = cu.zeros((nthread, ndet, ntof, nmedia), np.float32)
    photon_counter = cu.zeros((nthread, ndet, ntof), np.uint64)

    args = (
        cu.asarray(spec.to_record().view(np.uint32)),
        cu.asarray(source.to_record().view(np.uint32)),
        *source.extra_args(),
        np.uint32(nmedia),
        cu.asarray(np.stack([s.to_record() for s in states]).view(np.uint32)),
        cu.asarray(media),
        cu.asarray(geom.to_record().view(np.uint32)),
        *geom.extra_args(),
        cu.asarray(rng_states),
        np.uint32(ndet),
        cu.asarray(np.stack([d.to_record() for d in detectors]).view(np.uint32)),
        fluence,
        phi_td,
        phi_phase,
        phi_dist,
        mom_dist,
        photon_counter,
    )

    kernel = load_kernel(geom.kernel_prefix() + source.kernel_name())
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
            "Fluence": (list(fluence_keys), fluence),
        },
        coords={
            "time": (
                ["time"],
                (np.arange(ntof) + 0.5) * spec.dt,
            ),
        },
    )
