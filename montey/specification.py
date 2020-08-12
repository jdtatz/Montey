from __future__ import annotations
import numpy as np
import numba as nb
from numba import cuda
from typing import TypeVar, Generic, NamedTuple
from .vector import Vector

T = TypeVar('T')


class Specification(NamedTuple, Generic[T]):
    nphoton: int
    voxel_size: Vector[T]
    lifetime_max: T
    dt: T
    lightspeed: T
    freq: T
    # 2 𝜋 / λ
    wavenumber: T
    isflu: bool
    isdet: bool

    @staticmethod
    def numpy_dtype():
        return np.dtype([
            ('nphoton', 'i4'),
            ('voxel_size', ('f4', 3)),
            ('lifetime_max', 'f4'),
            ('dt', 'f4'),
            ('lightspeed', 'f4'),
            ('freq', 'f4'),
            ('wavenumber', 'f4'),
            ('isflu', np.bool_),
            ('isdet', np.bool_)
        ])

    def as_record(self):
        dtype = self.numpy_dtype()
        record_arr = np.rec.array([self._asdict()[k] for k in dtype.fields], dtype=dtype)
        return record_arr[()]

    @staticmethod
    def numba_type():
        return nb.types.NamedTuple((nb.i4, Vector.numba_type(nb.f4), nb.f4, nb.f4, nb.f4, nb.f4, nb.f4, nb.bool_, nb.bool_), Specification)


@cuda.jit(Specification.numba_type()(nb.from_dtype(Specification.numpy_dtype())), device=True)
def specification_from_record(record):
    return Specification(
        nphoton=record.nphoton,
        voxel_size=Vector(record.voxel_size[0], record.voxel_size[1], record.voxel_size[2]),
        lifetime_max=record.lifetime_max,
        dt=record.dt,
        lightspeed=record.lightspeed,
        freq=record.freq,
        wavenumber=record.wavenumber,
        isflu=record.isflu,
        isdet=record.isdet,
    )


state_dtype = np.dtype([('mua', 'f4'), ('mus', 'f4'), ('g', 'f4'), ('n', 'f4'), ('BFi', 'f4')])
