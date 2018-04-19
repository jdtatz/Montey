#!/usr/bin/env python3
import os
import time
import math
import numpy as np
import numba as nb
import numba.cuda
import numba.cuda.random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice'

inital_weight = np.float32(1)
roulette_const = np.float32(10)
roulette_threshold = np.float32(1e-4)


@nb.cuda.jit(device=True)
def launch(spec, rng):
    x, y, z = spec.srcpos
    vx, vy, vz = spec.srcdir
    return x, y, z, vx, vy, vz


@nb.cuda.jit(nb.types.Tuple((nb.f4, nb.i4))(nb.i4, nb.f4, nb.f4, nb.f4, nb.f4, nb.f4, nb.f4), device=True)
def intersect(prev, fx, fy, fz, vx, vy, vz):
    dx = (np.float32(1) if prev == np.int32(0) else (np.float32(1) - fx if vx > 0 else fx))
    dy = (np.float32(1) if prev == np.int32(1) else (np.float32(1) - fy if vy > 0 else fy))
    dz = (np.float32(1) if prev == np.int32(2) else (np.float32(1) - fz if vz > 0 else fz))
    hx = np.float32(abs(dx/vx)) if vx != 0 else np.float32(0)
    hy = np.float32(abs(dy/vy)) if vy != 0 else np.float32(0)
    hz = np.float32(abs(dz/vz)) if vz != 0 else np.float32(0)
    if hx != 0 and hx < hy and hx < hz:
        return hx, np.int32(0)
    elif hy != 0 and hy < hz:
        return hy, np.int32(1)
    else:
        return hz, np.int32(2)

@nb.cuda.jit(nb.f4(nb.f4, nb.f4), device=True)
def henyey_greenstein_phase(g, rand):
    if g != 0:
        return (np.float32(1) / (np.float32(2) * g)) * (np.float32(1) + g ** 2 -((np.float32(1) - g ** 2) / (np.float32(1) - g + np.float32(2) * g * rand)) ** 2)
    else:
        return np.float32(1) - np.float32(2) * rand


@nb.cuda.jit(fastmath=True)
def monte_carlo(counter, spec, states, media, rng, detpos, fluence, detp):
    buffer = nb.cuda.shared.array(0, nb.f4)
    nmedia = states.shape[0] - 1
    gid = nb.cuda.grid(1)
    tid = nb.cuda.threadIdx.x
    ppath_ind = tid*nmedia
    count = np.int32(0)
    weight = np.float32(0)
    reset = True
    n_out = states[0].n
    while True:
        if reset:
            reset = False
            outofbounds = False
            count += np.int32(1)
            if count >= spec.nphoton:
                return
            weight = inital_weight
            x, y, z, vx, vy, vz = launch(spec, rng)
            ix, iy, iz = np.int32(x), np.int32(y), np.int32(z)
            nscat = np.int32(0)
            t = np.float32(0)
            mid = media[ix, iy, iz]
            state = states[mid]
            if n_out != state.n:
                weight *= np.float32(4)*state.n*n_out/(state.n+n_out)**2
            buffer[ppath_ind:ppath_ind+nmedia] = 0
        # move
        rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
        s = -math.log(rand) / (state.mua + state.mus)
        dist, boundry = intersect(np.int32(-1), x-np.float32(ix), y-np.float32(iy), z-np.float32(iz), vx, vy, vz)
        while s > dist:
            x += vx*dist
            y += vy*dist
            z += vz*dist
            t += dist * state.n / spec.lightspeed
            s -= dist
            buffer[ppath_ind + mid - 1] += dist
            if boundry == np.int32(0):
                ix += np.int32(math.copysign(np.float32(1), vx))
                if not (np.int32(0) <= ix < np.int32(media.shape[0])):
                    outofbounds = True
                    s = np.float32(0)
                    break
            elif boundry == np.int32(1):
                iy += np.int32(math.copysign(np.float32(1), vy))
                if not (np.int32(0) <= iy < np.int32(media.shape[1])):
                    outofbounds = True
                    s = np.float32(0)
                    break
            else:
                iz += np.int32(math.copysign(np.float32(1), vz))
                if not (np.int32(0) <= iz < np.int32(media.shape[2])):
                    outofbounds = True
                    s = np.float32(0)
                    break
            mid, old_mid = media[ix, iy, iz], mid
            if mid == 0:
                break
            if mid != old_mid:
                old_mut = (state.mua + state.mus)
                state = states[mid]
                s *= old_mut / (state.mua + state.mus)
            dist, boundry = intersect(boundry, x-np.float32(ix), y-np.float32(iy), z-np.float32(iz), vx, vy, vz)
        x += vx*s
        y += vy*s
        z += vz*s
        t += s * state.n / spec.lightspeed
        if outofbounds or mid == 0 or t > spec.tend:
            if spec.isdet:
                for i in range(detpos.shape[0]):
                    if (detpos[i, 0] - x)**2 + (detpos[i, 1] - y)**2 + (detpos[i, 2] - z)**2 < detpos[i, 3]**2:
                        if detpos[i, 3] > 0:
                            detid = nb.cuda.atomic.add(counter, 0, 1)
                            detp[detid, 0] = i
                            detp[detid, 1] = weight
                            for i in range(nmedia):
                                detp[detid, 2+i] = buffer[ppath_ind+i]
                            detp[detid, 2+nmedia:8+nmedia] = x, y, z, vx, vy, vz
                        break
            reset = True
            continue
        buffer[ppath_ind + mid - 1] += s
        # absorb
        delta_weight = weight * state.mua / (state.mua + state.mus)
        if spec.isflu:
            # nb.cuda.atomic.add(fluence, (ix, iy, iz), delta_weight)
            nb.cuda.atomic.add(fluence, (np.int32(ix), np.int32(iy), np.int32(iz), np.int32(t//spec.tstep)), delta_weight)
        weight -= delta_weight
        # scatter
        rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
        ct = henyey_greenstein_phase(state.g, rand)
        st = math.sqrt(np.float32(1) - ct**2)
        rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
        phi = np.float32(2 * math.pi) * rand
        sp = math.sin(phi)
        cp = math.cos(phi)
        if abs(vz) < np.float32(1 - 1e-6):
            denom = math.sqrt(np.float32(1) - vz**2)
            vx, vy, vz = st*(vx*vz*cp-vy*sp) / denom + vx*ct, st*(vy*vz*cp+vx*sp) / denom + vy*ct, -denom*st*cp + vz*ct
        else:
            vx, vy, vz = st * cp, st * sp, ct * math.copysign(np.float32(1), vz)
        nscat += np.int32(1)
        # roulette
        if weight < roulette_threshold:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
            reset = rand > np.float32(1 / roulette_const)
            weight *= roulette_const


spec_dtype = np.dtype([('nphoton', 'i4'), ('srcpos', 'f4', 3), ('srcdir', 'f4', 3), ('tend', 'f4'), ('tstep', 'f4'), ('lightspeed', 'f4'), ('isflu', 'b'), ('isdet', 'b')])
spec = np.rec.array([(500, [100, 100, 0], [0, 0, 1], 5000, 100, 0.2998, True, True)], dtype=spec_dtype)[0]

state_dtype = np.dtype([('mua', 'f4'), ('mus', 'f4'), ('g', 'f4'), ('n', 'f4')])
states = np.rec.array([(0, 0, 1, 1), (1e-2, 3, 0.9, 1.4), (1e-3, 2.4, 0.9, 1.4), (1.3e-2, 5, 0.9, 1.4)], dtype=state_dtype)
nmedia = (states.shape[0]-1)

media = np.empty((200, 200, 200), np.int8)
media[:, :, :] = 1
#media[:, :, 4:7] = 2
#media[:, :, 7:] = 3

threads_per_block = 256
blocks = 256
pcount = blocks*threads_per_block*spec.nphoton

#fluence = np.zeros((200, 200, 200), np.float32)
fluence = np.zeros((200, 200, 200, int(np.ceil(spec.tend / spec.tstep))), np.float32)

radii = np.array([10, 20, 30, 40, 50])
detpos = np.array([[*spec.srcpos, r] for r in radii], dtype=np.float32)

detp = np.empty((pcount, 8+nmedia), dtype=np.float32)

counter = np.zeros(10, np.int32)

with numba.cuda.gpus[1]:
    rng_states = nb.cuda.random.create_xoroshiro128p_states(threads_per_block * blocks, seed=42)
    t1 = time.time()
    monte_carlo[blocks, threads_per_block, None, 4*threads_per_block*nmedia](counter, spec, states, media, rng_states, detpos, fluence, detp)
    dt = time.time() - t1

with open('typed_montey.txt', 'w') as f:
    monte_carlo.inspect_types(f)
with open('llvm_montey.ll', 'w') as f:
    f.write(monte_carlo.inspect_llvm().popitem()[1])
with open('ptx_montey.ptx', 'w') as f:
    f.write(monte_carlo.inspect_asm().popitem()[1])

print("max # photons: {}, photons/ms: {}, time(s): {}".format(pcount, (pcount)/(1000*dt), dt))
if spec.isdet:
    nphoton = counter[0]
    print("det # photons: {}, mean det: {}".format(nphoton, np.mean(detp[:nphoton, 0])))
    
    area = np.pi*radii**2
    area[1:] -= area[:-1]
    tof_domain = np.append(np.arange(0, spec.tend, spec.tstep), spec.tend)

    ndet, ntof = len(radii), len(tof_domain) - 1

    c = spec.lightspeed
    detBins = detp[:nphoton, 0].astype(np.int32)
    tofBins = np.minimum(np.digitize(detp[:nphoton, 2:(2+nmedia)] @ states[1:].n, c * tof_domain), ntof) - 1

    phiTD = np.zeros((ndet, ntof), np.float64)
    np.add.at(phiTD, (detBins, tofBins), np.exp(detp[:nphoton, 2:(2+nmedia)] @ -states[1:].mua))
    phiTD /= (area[:, np.newaxis] * nphoton)
    phiCW = np.sum(phiTD, axis=1)

    plt.semilogy(radii, phiCW, '*')
    plt.savefig('phicw.png')


if spec.isflu:
    fig = plt.figure()
    fig.suptitle('Fluence')
    frames = [[plt.imshow(fluence[:, 100, :, i], animated=True)] for i in range(fluence.shape[3])]
    ani = animation.ArtistAnimation(fig, frames, blit=True)
    ani.save('fluence.gif', writer='imagemagick', fps=10)
    plt.clf()
    #plt.imshow(np.log(fluence[:, 100, :]))
    #plt.savefig('fluence.png')
    #plt.clf()
