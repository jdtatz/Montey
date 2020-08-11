from __future__ import annotations
import operator
import numpy as np
import numba as nb
from numba import cuda, types
from numba.core.typing import templates
from numba.core.typing.templates import infer as register_cpu, infer_getattr as register_cpu_attr, infer_global as register_cpu_global
from numba.core.imputils import lower_builtin as lower_cpu
from numba.cuda.cudadecl import register as register_gpu, register_attr as register_gpu_attr, register_global as register_gpu_global
from numba.cuda.cudaimpl import lower as lower_gpu
from typing import NamedTuple, TypeVar, Generic

T = TypeVar("T")


class Vector(NamedTuple, Generic[T]):
    x: T
    y: T
    z: T

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __add__(self, rhs: Vector[T]) -> Vector[T]:
        return Vector(x=self.x + rhs.x, y=self.y + rhs.y, z=self.z + rhs.z)

    def __sub__(self, rhs: Vector[T]) -> Vector[T]:
        return Vector(x=self.x - rhs.x, y=self.y - rhs.y, z=self.z - rhs.z)

    def __mul__(self, rhs: T) -> Vector[T]:
        return Vector(x=self.x * rhs, y=self.y * rhs, z=self.z * rhs)

    def __truediv__(self, rhs: T) -> Vector[T]:
        return Vector(x=self.x / rhs, y=self.y / rhs, z=self.z / rhs)

    def __matmul__(self, rhs: Vector[T]) -> T:
        return self.x * rhs.z + self.y * rhs.y + self.z * rhs.z

    def multiply_add(self, b: T, c: Vector[T]) -> Vector[T]:
        """Imprecise implementation"""
        return Vector(x=(self.x * b + c.x), y=(self.y * b + c.y), z=(self.z * b + c.z))

    @staticmethod
    def numba_type(scalar):
        return types.NamedUniTuple(scalar, 3, Vector)

    def numba_compat(self):
        return Vector(x=np.float32(self.x), y=np.float32(self.y), z=np.float32(self.z))


_fvec = Vector.numba_type(types.f4)
_ivec = Vector.numba_type(types.i4)


@register_cpu
@register_gpu
class VectorAdd(templates.ConcreteTemplate):
    key = operator.add
    cases = [
        templates.signature(_fvec, _fvec, _fvec),
        templates.signature(_ivec, _ivec, _ivec),
    ]


register_cpu_global(operator.add, types.Function(VectorAdd))
register_gpu_global(operator.add, types.Function(VectorAdd))


@lower_cpu(operator.add, _fvec, _fvec)
@lower_cpu(operator.add, _ivec, _ivec)
@lower_gpu(operator.add, _fvec, _fvec)
@lower_gpu(operator.add, _ivec, _ivec)
def impl_vector_add(context, builder, sig, args):
    return context.compile_internal(builder, Vector.__add__, sig, args)


@register_cpu
@register_gpu
class VectorSub(templates.ConcreteTemplate):
    key = operator.sub
    cases = [
        templates.signature(_fvec, _fvec, _fvec),
        templates.signature(_ivec, _ivec, _ivec),
    ]


register_cpu_global(operator.sub, types.Function(VectorSub))
register_gpu_global(operator.sub, types.Function(VectorSub))


@lower_cpu(operator.sub, _fvec, _fvec)
@lower_cpu(operator.sub, _ivec, _ivec)
@lower_gpu(operator.sub, _fvec, _fvec)
@lower_gpu(operator.sub, _ivec, _ivec)
def impl_vector_sub(context, builder, sig, args):
    return context.compile_internal(builder, Vector.__sub__, sig, args)


@register_cpu
@register_gpu
class VectorMul(templates.ConcreteTemplate):
    key = operator.mul
    cases = [
        templates.signature(_fvec, _fvec, nb.f4),
        templates.signature(_fvec, nb.f4, _fvec),
        templates.signature(_ivec, _ivec, nb.i4),
        templates.signature(_ivec, nb.i4, _ivec),
    ]


register_cpu_global(operator.mul, types.Function(VectorMul))
register_gpu_global(operator.mul, types.Function(VectorMul))


@lower_cpu(operator.mul, _fvec, nb.f4)
@lower_cpu(operator.mul, nb.f4, _fvec)
@lower_cpu(operator.mul, _ivec, nb.i4)
@lower_cpu(operator.mul, nb.i4, _ivec)
@lower_gpu(operator.mul, _fvec, nb.f4)
@lower_gpu(operator.mul, nb.f4, _fvec)
@lower_gpu(operator.mul, _ivec, nb.i4)
@lower_gpu(operator.mul, nb.i4, _ivec)
def impl_vector_mul(context, builder, sig, args):
    return context.compile_internal(builder, Vector.__mul__, sig, args)


@register_cpu
@register_gpu
class VectorDiv(templates.ConcreteTemplate):
    key = operator.truediv
    cases = [
        templates.signature(_fvec, _fvec, nb.f4),
        templates.signature(_ivec, _ivec, nb.i4),
    ]


register_cpu_global(operator.truediv, types.Function(VectorDiv))
register_gpu_global(operator.truediv, types.Function(VectorDiv))


@lower_cpu(operator.truediv, _fvec, nb.f4)
@lower_cpu(operator.truediv, _ivec, nb.i4)
@lower_gpu(operator.truediv, _fvec, nb.f4)
@lower_gpu(operator.truediv, _ivec, nb.i4)
def impl_vector_div(context, builder, sig, args):
    return context.compile_internal(builder, Vector.__truediv__, sig, args)


@register_cpu
@register_gpu
class VectorMatMul(templates.ConcreteTemplate):
    key = operator.matmul
    cases = [
        templates.signature(nb.f4, _fvec, _fvec),
        templates.signature(nb.i4, _ivec, _ivec),
    ]


register_cpu_global(operator.matmul, types.Function(VectorMatMul))
register_gpu_global(operator.matmul, types.Function(VectorMatMul))


@lower_cpu(operator.matmul, _fvec, _fvec)
@lower_cpu(operator.matmul, _ivec, _ivec)
@lower_gpu(operator.matmul, _fvec, _fvec)
@lower_gpu(operator.matmul, _ivec, _ivec)
def impl_vector_matmul(context, builder, sig, args):
    return context.compile_internal(builder, Vector.__matmul__, sig, args)


# @register_cpu
@register_gpu
class VectorFMA(templates.ConcreteTemplate):
    key = (_fvec, "multiply_add")
    is_method = True
    cases = [
        templates.signature(_fvec, nb.f4, _fvec, recvr=_fvec),
    ]


# @register_cpu_attr
@register_gpu_attr
class VectorAttrs(templates.AttributeTemplate):
    key = _fvec

    def resolve_x(self, vec):
        return vec[0]

    def resolve_y(self, vec):
        return vec[1]

    def resolve_z(self, vec):
        return vec[2]

    def resolve_multiply_add(self, vec):
        return types.BoundFunction(VectorFMA, vec)


# register_cpu_global((_fvec, "multiply_add"), types.BoundFunction(VectorFMA, _fvec))
register_gpu_global((_fvec, "multiply_add"), types.BoundFunction(VectorFMA, _fvec))


@lower_gpu((_fvec, 'multiply_add'), _fvec, nb.f4, _fvec)
def impl_vector_fma(context, builder, sig, args):
    return context.compile_internal(builder, lambda a, b, c: Vector(x=cuda.fma(a.x, b, c.x), y=cuda.fma(a.y, b, c.y), z=cuda.fma(a.z, b, c.z)), sig, args)


@lower_cpu((_fvec, 'multiply_add'), _fvec, nb.f4, _fvec)
def impl_vector_fma(context, builder, sig, args):
    return context.compile_internal(builder, Vector.multiply_add, sig, args)


@cuda.jit(device=True)
def vmulv(lhs, rhs):
    return Vector(x=lhs.x * rhs.x, y=lhs.y * rhs.y, z=lhs.z * rhs.z)


@cuda.jit(Vector.numba_type(nb.f4)(Vector.numba_type(nb.i4)), device=True)
def vi2f(v):
    return Vector(x=np.float32(v.x), y=np.float32(v.y), z=np.float32(v.z))


@cuda.jit(Vector.numba_type(nb.i4)(Vector.numba_type(nb.f4)), device=True)
def vf2i(v):
    return Vector(x=np.int32(v.x), y=np.int32(v.y), z=np.int32(v.z))
