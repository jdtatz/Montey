from __future__ import annotations
import math
import cmath
import time
import numpy as np
from numba import cuda, types
from numba.core.types import int32, int64, float32, float64, complex64, complex128
import numba.cuda.random
from typing import NamedTuple, Tuple, Callable, Any, Optional
from .vector import Vector, vmulv, vf2i, vi2f
from .specification import Specification, specification_from_record
from .sources import Source


class MonteCarloResults(NamedTuple):
    fluence: float64[:, :, :, :1]
    phiCW: float64[::1]
    phiFD: complex128[::1]
    phiTD: float64[:, ::1]
    phiDist: float64[:, :, ::1]
    g1: float64[:, ::1]
    photonCounter: int64[:, ::1]


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
c0 = np.complex64(0)
c1j = np.complex64(1j)
PI_2 = np.float32(2 * math.pi)
EPS_N_1 = np.float32(1 - 1e-6)


@cuda.jit(float32(float32), device=True)
def sqr(x):
    return x * x


@cuda.jit(int32(float32), device=True)
def signum(x):
    if x > f0:
        return i1
    elif x < f0:
        return -i1
    else:
        return i0


@cuda.jit(types.Tuple((float32, int32))(int32, Vector.numba_type(float32), Vector.numba_type(float32), Vector.numba_type(float32)), device=True)
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


@cuda.jit(float32(float32, float32), device=True)
def henyey_greenstein_phase(g, rand):
    if g != 0:
        g2 = sqr(g)
        return (f1 / (f2 * g)) * (f1 + g2 - sqr((f1 - g2) / (f1 - g + f2 * g * rand)))
    else:
        return f1 - f2 * rand


def create_monte_carlo(source: Source) -> Tuple[Any, Callable[[Specification, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[int], Optional[int], Optional[int]], Tuple[MonteCarloResults, float]]]:
    vec_type = Vector.numba_type(float32)
    rng_type = cuda.random.xoroshiro128p_type[::1]
    launch = source.create_launch_function()
    launch = cuda.jit(types.Tuple((vec_type, vec_type))(rng_type, int32), device=True)(launch)

    @cuda.jit(fastmath=True)
    def monte_carlo(spec, states, media, rng, detpos, fluence, phi_td, phi_fd, tau, g1_top, photon_counter, phi_dist):
        spec = specification_from_record(spec)
        buffer = cuda.shared.array(0, float32)
        nmedia = states.shape[0] - 1
        gid = cuda.grid(1)
        tid = cuda.threadIdx.x
        ppath_ind = tid*nmedia
        count = i0
        weight = f0
        t = f0
        reset = True
        outofbounds = False
        n_out = states[0].n
        while True:
            if reset:
                reset = False
                outofbounds = False
                count += i1
                if count > spec.nphoton:
                    return
                weight = inital_weight
                p, v = launch(rng, gid)
                idx = vf2i(Vector(p.x // spec.voxel_size.x, p.y // spec.voxel_size.y, p.z // spec.voxel_size.z))
                if not (i0 <= idx.x < np.int32(media.shape[0]) and i0 <= idx.y < np.int32(media.shape[1]) and i0 <= idx.z < np.int32(media.shape[2])):
                    reset = True
                    continue
                t = f0
                mid = media[idx.x, idx.y, idx.z]
                state = states[mid]
                if n_out != state.n:
                    weight *= f4*state.n*n_out/sqr(state.n+n_out)
                buffer[ppath_ind:ppath_ind+nmedia] = f0
            # move
            rand = cuda.random.xoroshiro128p_uniform_float32(rng, gid)
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
            if outofbounds or mid == 0 or t > spec.lifetime_max:
                if spec.isdet:
                    for i in range(detpos.shape[0]):
                        sqr_dist = sqr(detpos[i, 0] - p.x) + sqr(detpos[i, 1] - p.y) + sqr(detpos[i, 2] - p.z)
                        if sqr_dist < sqr(detpos[i, 3]):
                            # optical path length
                            opl = f0
                            # frequency-domain optical path length
                            opl_fd = c0
                            g1_prep = f0
                            totaldist = f0
                            temp1 = c1j * PI_2 * spec.freq / spec.lightspeed
                            for j in range(nmedia):
                                opl += buffer[ppath_ind + j] * (-states[1 + j].mua)
                                opl_fd += buffer[ppath_ind + j] * (-states[1 + j].mua - temp1 * states[1 + j].n)
                                totaldist += buffer[ppath_ind + j] * states[1 + j].n
                                # k = n k0 = 2 𝜋 n / λ  ; wavenumber of light in the media
                                # BFi = blood flow index = α D_b
                                # ⟨Δr^2(τ)⟩ = 6 D_b τ  => ⟨Δr^2(τ)⟩ = 6 α D_b τ = 6 BFi τ
                                # g1(τ) = 1/N Sum_i^N { exp[ Sum_j^nmedia { -(1/3) Y k^2 ⟨Δr^2(τ)⟩ } + opl ] }]
                                k = sqr(spec.wavenumber * states[1 + j].n)
                                g1_prep += buffer[ppath_ind + j] * (-f2 * k * states[1 + j].BFi)
                            time_id = min(np.int32(t // spec.dt), phi_td.shape[2] - 1)
                            phi = math.exp(opl)
                            phi_td[gid, i, time_id] += phi
                            phi_fd[gid, i] += cmath.exp(opl_fd)
                            for j in range(len(tau)):
                                g1_top[gid, i, j] += math.exp(g1_prep * tau[j] + opl)
                            photon_counter[gid, i, time_id] += 1
                            for j in range(nmedia):
                                layerdist = buffer[ppath_ind + j] * states[1 + j].n
                                phi_dist[gid, i, time_id, j] += phi * layerdist / totaldist
                            break
                reset = True
                continue
            buffer[ppath_ind + mid - 1] += s
            # absorb
            delta_weight = weight * state.mua / (state.mua + state.mus)
            if spec.isflu:
                # cuda.atomic.add(fluence, (ix, iy, iz), delta_weight)
                cuda.atomic.add(fluence, (idx.x, idx.y, idx.z, np.int32(t//spec.dt)), delta_weight)
            weight -= delta_weight
            # scatter
            rand = cuda.random.xoroshiro128p_uniform_float32(rng, gid)
            ct = henyey_greenstein_phase(state.g, rand)
            st = math.sqrt(f1 - sqr(ct))
            rand = cuda.random.xoroshiro128p_uniform_float32(rng, gid)
            phi = PI_2 * rand
            sp = math.sin(phi)
            cp = math.cos(phi)
            if abs(v.z) < EPS_N_1:
                denom = math.sqrt(f1 - sqr(v.z))
                v = Vector(
                    x=st*(v.x*v.z*cp-v.y*sp) / denom + v.x*ct,
                    y=st*(v.y*v.z*cp+v.x*sp) / denom + v.y*ct,
                    z=-denom*st*cp + v.z*ct
                )
            else:
                v = Vector(st * cp, st * sp, ct * math.copysign(f1, v.z))
            # roulette
            if weight < roulette_threshold:
                rand = cuda.random.xoroshiro128p_uniform_float32(rng, gid)
                reset = rand > recip_roulette_const
                weight *= roulette_const

    def wrapped(spec: Specification, states: np.ndarray, media: np.ndarray,
                detpos: np.ndarray, tau: np.ndarray, detector_area: np.ndarray,
                warps_per_block: int = 8, blocks: int = 256, gpu_id: int = 0) \
            -> Tuple[MonteCarloResults, float]:
        threads_per_block = warps_per_block * 32
        nthread = blocks * threads_per_block
        nmedia = (states.shape[0] - 1)
        ntof = int(np.ceil(spec.lifetime_max / spec.dt))
        ndet = detpos.shape[0]
        fluence = np.zeros((*media.shape, ntof), np.float32)
        phi_td = np.zeros((nthread, ndet, ntof), np.float32)
        phi_fd = np.zeros((nthread, ndet), np.complex64)
        g1_top = np.zeros((nthread, ndet, len(tau)), np.float32)
        photon_counter = np.zeros((nthread, ndet, ntof), np.int64)
        phi_dist = np.zeros((nthread, ndet, ntof, nmedia), np.float32)
        print(f"Fluence size: {fluence.nbytes / 1e6}Mb")
        print(f"PhiTD size: {phi_td.nbytes / 1e6}Mb")
        print(f"PhiFD size: {phi_fd.nbytes / 1e6}Mb")
        print(f"g1_numerator size: {g1_top.nbytes / 1e6}Mb")
        print(f"Photon counter size: {photon_counter.nbytes / 1e6}Mb")
        print(f"PhiDist size: {phi_dist.nbytes / 1e6}Mb")
        mem_usage = (
                fluence.nbytes +
                phi_td.nbytes +
                phi_fd.nbytes +
                g1_top.nbytes +
                photon_counter.nbytes +
                phi_dist.nbytes
        )
        print(f"Total GPU Memory usage: {mem_usage / 1e6}Mb")

        with cuda.gpus[gpu_id]:
            rng_states = cuda.random.create_xoroshiro128p_states(threads_per_block * blocks, seed=42)
            t1 = time.time()
            monte_carlo[blocks, threads_per_block, None, 4 * threads_per_block * nmedia](
                spec.as_record(),
                states,
                media,
                rng_states,
                detpos,
                fluence,
                phi_td,
                phi_fd,
                tau,
                g1_top,
                photon_counter,
                phi_dist,
            )
            dt = time.time() - t1
        pcount = nthread * spec.nphoton
        phiCW = phi_td.sum(axis=(0, 2), dtype=np.float64) / (detector_area * pcount)
        phiFD = phi_fd.sum(axis=0, dtype=np.complex128) / (detector_area * pcount)
        phiTD = phi_td.sum(axis=0, dtype=np.float64) / (spec.dt * detector_area[:, np.newaxis] * pcount)
        # TODO verify normalization for phiDist & g1
        phiDist = phi_dist.sum(axis=0, dtype=np.float64) / phi_td.sum(axis=(0, 2), dtype=np.float64)[:, np.newaxis, np.newaxis]
        g1 = g1_top.sum(axis=0, dtype=np.float64) / phi_td.sum(axis=(0, 2), dtype=np.float64)[:, np.newaxis]
        res = MonteCarloResults(
            fluence=fluence,
            phiCW=phiCW,
            phiFD=phiFD,
            phiTD=phiTD,
            phiDist=phiDist,
            g1=g1,
            photonCounter=photon_counter.sum(axis=0),
        )
        return res, dt
    return monte_carlo, wrapped
