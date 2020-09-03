import numpy as np
import cupy as cu
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from montey import *

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
