from __future__ import annotations
import math
import time
import numpy as np
import numba as nb
from numba import types, cuda
import numba.cuda.random
from .vector import Vector, vmulv, vf2i, vi2f
from .specification import Specification, specification_from_record
from .sources import Source


inital_weight = np.float32(1)
roulette_const = np.float32(10)
recip_roulette_const = np.float32(1 / roulette_const)
roulette_threshold = np.float32(1e-4)
i0 = np.int32(0)
i1 = np.int32(1)
i2 = np.int32(2)
f0 = np.float32(0)
f1 = np.float32(1)
f2 = np.float32(2)
f4 = np.float32(4)
PI_2 = np.float32(2 * math.pi)
EPS_N_1 = np.float32(1 - 1e-6)


@nb.cuda.jit(nb.f4(nb.f4), device=True)
def sqr(x):
    return x * x


@nb.cuda.jit(nb.i4(nb.f4), device=True)
def signum(x):
    if x > f0:
        return i1
    elif x < f0:
        return -i1
    else:
        return i0


@nb.cuda.jit(nb.types.Tuple((nb.f4, nb.i4))(nb.i4, Vector.numba_type(nb.f4), Vector.numba_type(nb.f4), Vector.numba_type(nb.f4)), device=True)
def intersect(prev, p, v, voxel_size):
    dx = (voxel_size.x if prev == i0 else (voxel_size.x - p.x if v.x > f0 else p.x))
    dy = (voxel_size.y if prev == i1 else (voxel_size.y - p.y if v.y > f0 else p.y))
    dz = (voxel_size.z if prev == i2 else (voxel_size.z - p.z if v.z > f0 else p.z))
    hx = np.float32(abs(dx/v.x)) if v.x != f0 else f0
    hy = np.float32(abs(dy/v.y)) if v.y != f0 else f0
    hz = np.float32(abs(dz/v.z)) if v.z != f0 else f0
    if hx != 0 and hx < hy and hx < hz:
        return hx, i0
    elif hy != 0 and hy < hz:
        return hy, i1
    else:
        return hz, i2


@nb.cuda.jit(nb.f4(nb.f4, nb.f4), device=True)
def henyey_greenstein_phase(g, rand):
    if g != 0:
        g2 = sqr(g)
        return (f1 / (f2 * g)) * (f1 + g2 - sqr((f1 - g2) / (f1 - g + f2 * g * rand)))
    else:
        return f1 - f2 * rand


def create_monte_carlo(source: Source):
    vec_type = Vector.numba_type(nb.f4)
    spec_type = Specification.numba_type()
    rng_type = nb.cuda.random.xoroshiro128p_type[::1]
    launch = source.create_launch_function()
    launch = nb.cuda.jit(types.Tuple((vec_type, vec_type))(spec_type, rng_type, nb.i4), device=True)(launch)

    @nb.cuda.jit(fastmath=True)
    def monte_carlo(counter, spec, states, media, rng, detpos, fluence, detp):
        spec = specification_from_record(spec)
        buffer = nb.cuda.shared.array(0, nb.f4)
        nmedia = states.shape[0] - 1
        gid = nb.cuda.grid(1)
        tid = nb.cuda.threadIdx.x
        ppath_ind = tid*nmedia
        count = i0
        weight = f0
        reset = True
        n_out = states[0].n
        while True:
            if reset:
                reset = False
                outofbounds = False
                count += i1
                if count >= spec.nphoton:
                    return
                weight = inital_weight
                p, v = launch(spec, rng, gid)
                idx = vf2i(Vector(p.x // spec.voxel_size.x, p.y // spec.voxel_size.y, p.z // spec.voxel_size.z))
                nscat = i0
                t = f0
                mid = media[idx.x, idx.y, idx.z]
                state = states[mid]
                if n_out != state.n:
                    weight *= f4*state.n*n_out/sqr(state.n+n_out)
                buffer[ppath_ind:ppath_ind+nmedia] = f0
            # move
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
            s = -math.log(rand) / (state.mua + state.mus)
            dist, boundry = intersect(-i1, p - vmulv(vi2f(idx), spec.voxel_size), v, spec.voxel_size)
            while s > dist:
                p = v.multiply_add(dist, p)
                t += dist * state.n / spec.lightspeed
                s -= dist
                buffer[ppath_ind + mid - 1] += dist
                if boundry == i0:
                    # idx = vadd(idx, Vector(signum(v.x), i0, i0))
                    ix = np.int32(idx.x + signum(v.x))
                    idx = Vector(ix, idx.y, idx.z)
                    if not (i0 <= idx.x < np.int32(media.shape[0])):
                        outofbounds = True
                        s = f0
                        break
                elif boundry == i1:
                    # idx = vadd(idx, Vector(i0, signum(v.y), i0))
                    iy = np.int32(idx.y + signum(v.y))
                    idx = Vector(idx.x, iy, idx.z)
                    if not (i0 <= idx.y < np.int32(media.shape[1])):
                        outofbounds = True
                        s = f0
                        break
                else:
                    # idx = vadd(idx, Vector(i0, i0, signum(v.z)))
                    iz = np.int32(idx.z + signum(v.z))
                    idx = Vector(idx.x, idx.y, iz)
                    if not (i0 <= idx.z < np.int32(media.shape[2])):
                        outofbounds = True
                        s = f0
                        break
                mid, old_mid = media[idx.x, idx.y, idx.z], mid
                if mid == 0:
                    break
                if mid != old_mid:
                    old_mut = (state.mua + state.mus)
                    state = states[mid]
                    s *= old_mut / (state.mua + state.mus)
                dist, boundry = intersect(boundry, (p - vmulv(vi2f(idx), spec.voxel_size)), v, spec.voxel_size)
            p = v.multiply_add(s, p)
            t += s * state.n / spec.lightspeed
            if outofbounds or mid == 0 or t > spec.tend:
                if spec.isdet:
                    for i in range(detpos.shape[0]):
                        if sqr(detpos[i, 0] - p.x) + sqr(detpos[i, 1] - p.y) + sqr(detpos[i, 2] - p.z) < sqr(detpos[i, 3]):
                            if detpos[i, 3] > 0:
                                detid = nb.cuda.atomic.add(counter, 0, 1)
                                detp[detid, 0] = i
                                detp[detid, 1] = weight
                                for i in range(nmedia):
                                    detp[detid, 2+i] = buffer[ppath_ind+i]
                                detp[detid, 2+nmedia:8+nmedia] = p.x, p.y, p.z, v.x, v.y, v.z
                            break
                reset = True
                continue
            buffer[ppath_ind + mid - 1] += s
            # absorb
            delta_weight = weight * state.mua / (state.mua + state.mus)
            if spec.isflu:
                # nb.cuda.atomic.add(fluence, (ix, iy, iz), delta_weight)
                nb.cuda.atomic.add(fluence, (idx.x, idx.y, idx.z, np.int32(t//spec.tstep)), delta_weight)
            weight -= delta_weight
            # scatter
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
            ct = henyey_greenstein_phase(state.g, rand)
            st = math.sqrt(f1 - sqr(ct))
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
            phi = PI_2 * rand
            sp = math.sin(phi)
            cp = math.cos(phi)
            if abs(v.z) < EPS_N_1:
                denom = math.sqrt(f1 - sqr(v.z))
                v = Vector(st*(v.x*v.z*cp-v.y*sp) / denom + v.x*ct, st*(v.y*v.z*cp+v.x*sp) / denom + v.y*ct, -denom*st*cp + v.z*ct)
            else:
                v = Vector(st * cp, st * sp, ct * math.copysign(f1, v.z))
            nscat += i1
            # roulette
            if weight < roulette_threshold:
                rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
                reset = rand > recip_roulette_const
                weight *= roulette_const

    def wrapped(spec: Specification, states: np.ndarray, media: np.ndarray, detpos: np.ndarray,
                warps_per_block: int = 8, blocks: int = 256, gpu_id: int = 0):
        threads_per_block = warps_per_block * 32
        pcount = blocks * threads_per_block * spec.nphoton
        nmedia = (states.shape[0] - 1)
        fluence = np.zeros((200, 200, 200, int(np.ceil(spec.tend / spec.tstep))), np.float32)
        detp = np.empty((pcount, 8 + nmedia), dtype=np.float32)
        counter = np.zeros(10, np.int32)
        with cuda.gpus[gpu_id]:
            rng_states = cuda.random.create_xoroshiro128p_states(threads_per_block * blocks, seed=42)
            t1 = time.time()
            monte_carlo[blocks, threads_per_block, None, 4 * threads_per_block * nmedia](
                counter,
                spec.as_record(),
                states,
                media,
                rng_states,
                detpos,
                fluence,
                detp
            )
            dt = time.time() - t1
        nphoton = counter[0]
        return fluence, detp[:nphoton], dt
    return monte_carlo, wrapped
