from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from numba.cuda.random import xoroshiro128p_dtype, init_xoroshiro128p_states_cpu

vector_dtype = np.dtype([
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4")
])
spec_dtype = np.dtype([
    ("nphoton", "u8"),
    ("voxel_dim", vector_dtype),
    ("lifetime_max", "f4"),
    ("dt", "f4"),
    ("lightspeed", "f4"),
    ("freq", "f4"),
])
state_dtype = np.dtype([
    ("mua", "f4"),
    ("mus", "f4"),
    ("g", "f4"),
    ("n", "f4")
])
detector_dtype = np.dtype([
    ("position", vector_dtype),
    ("radius", "f4"),
])
pencil_src_dtype = np.dtype([
    ("src_pos", vector_dtype),
    ("src_dir", vector_dtype),
])
disk_src_dtype = np.dtype([
    ("src_pos", vector_dtype),
    ("src_dir", vector_dtype),
    ("orthonormal_basis", vector_dtype, 2),
    ("radius", "f4"),
])


spec = np.zeros(1, dtype=spec_dtype).view(np.recarray)
spec.nphoton = int(50)
spec.voxel_dim.x = 1.0
spec.voxel_dim.y = 1.0
spec.voxel_dim.z = 1.0
spec.lifetime_max = 5000
spec.dt = 100
spec.lightspeed = 0.2998
spec.freq = 110e-6
print(spec)
print(bytes(spec))

seed = 12345
nwarp = 4
nblock = 512
nthread = nblock * nwarp * 32
pcount = nthread * spec.nphoton
print(nthread, pcount)

rng_states = np.empty(shape=nthread, dtype=xoroshiro128p_dtype)
init_xoroshiro128p_states_cpu(rng_states, seed, subsequence_start=0)
print(rng_states)
rng_states = rng_states.view(np.uint64)

source = np.zeros(1, dtype=pencil_src_dtype).view(np.recarray)
source.src_pos = 8, 100, 100
source.src_dir = 1, 0, 0
print(source)
media = np.ones((200, 200, 200), dtype=np.uint8)
nmedia = 1
states = np.zeros(1 + nmedia, dtype=state_dtype).view(np.recarray)
states[0] = 0, 0, 1, 1
states[1] = 1e-2, 3, 0.9, 1.4
print(states)
detectors = np.zeros(3, dtype=detector_dtype).view(np.recarray)
detectors[0] = (0, 100, 100), 10
detectors[1] = (0, 100, 100), 20
detectors[2] = (0, 100, 100), 30
ndet = detectors.shape[0]
ntof = int(np.ceil(spec.lifetime_max / spec.dt))
fluence = np.zeros((*media.shape, ntof), np.float32)
phi_td = np.zeros((nthread, ndet, ntof), np.float32)
phi_fd = np.zeros((nthread, ndet), np.complex64)
phi_dist = np.zeros((nthread, ndet, ntof, nmedia), np.float32)
photon_counter = np.zeros((nthread, ndet, ntof), np.uint64)

dev = drv.Device(0)
print(dev.name())
ctxt: drv.Context = dev.make_context()
ctxt.push()

args = [
    gpuarray.to_gpu(spec.view(np.uint32)),
    gpuarray.to_gpu(source.view(np.uint32)),
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
    drv.InOut(phi_fd),
    drv.InOut(phi_dist),
    drv.InOut(photon_counter),
]
print(args[0].dtype)

with open("src/kernel.ptx", "rb") as f:
    module = drv.module_from_buffer(f.read())
print(module)
pencil = module.get_function("pencil")
print(pencil)
pencil.prepare([
    'P',
    'P',
    np.uint32,
    'P',
    np.uint32,
    np.uint32,
    np.uint32,
    np.uint8,
    np.uint64,
    np.uint32,
    'P',
    np.float32,
    np.float32,
    np.complex64,
    np.float32,
    np.uint64,
])
smem = nwarp * 32 * nmedia * 4
dt = pencil(*args, block=(nwarp * 32, 1, 1), grid=(nblock, 1), shared=smem, time_kernel=True)
dt *= 1000
print(dt)
print(pcount)
print(pcount / dt)

sliced = fluence[:, :, 100]
_, _, nt = sliced.shape

sliced = np.log(sliced)
sliced[~np.isfinite(sliced)] = np.nan
norm = colors.Normalize(vmin=np.nanmin(sliced), vmax=np.nanmax(sliced))

print(np.nanmin(sliced), np.nanmax(sliced))
print(photon_counter)
print(photon_counter.sum(axis=0))

fig, ax = plt.subplots()

mat = ax.matshow(sliced[:, :, 0].T, origin='lower', animated=True, norm=norm)
clb = fig.colorbar(mat)


def animate(i):
    mat.set_data(sliced[:, :, i].T)
    return mat,


anim = FuncAnimation(fig, animate, frames=nt, interval=100, blit=True)
anim.save("anim.gif", writer='imagemagick')
