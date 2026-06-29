import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
N = 101          # number of lattice sites
a = 1.0          # lattice spacing
J = 1.0          # hopping
center = N // 2

sites = np.arange(N) - center

# -------------------------
# Hamiltonian
# -------------------------
def tilted_hamiltonian(F_over_J, N=101, J=1.0, a=1.0):
    F = F_over_J * J

    H = np.zeros((N, N))

    # hopping
    for n in range(N - 1):
        H[n, n + 1] = -J
        H[n + 1, n] = -J

    # tilt
    for n in range(N):
        H[n, n] = F * a * (n - N//2)

    return H

# -------------------------
# RMS width
# -------------------------
def rms_width(psi, positions):
    prob = np.abs(psi)**2

    x_mean = np.sum(prob * positions)
    x2_mean = np.sum(prob * positions**2)

    return np.sqrt(x2_mean - x_mean**2)

# -------------------------
# Plot eigenstates
# -------------------------
FJ_values = [0.2, 1.0, 5.0]

fig, axes = plt.subplots(len(FJ_values), 1, figsize=(8, 8))

for ax, FJ in zip(axes, FJ_values):

    H = tilted_hamiltonian(FJ, N, J, a)

    E, V = np.linalg.eigh(H)

    # choose eigenstate near middle of spectrum
    idx = len(E)//2
    psi = V[:, idx]

    ax.plot(sites, np.abs(psi)**2)
    ax.set_title(f"F/J = {FJ}")
    ax.set_ylabel(r"$|\psi_n|^2$")

axes[-1].set_xlabel("site n")
plt.tight_layout()
plt.show()

# -------------------------
# Localization length vs F/J
# -------------------------
FJ_scan = np.logspace(-1, 1.2, 30)

widths = []

for FJ in FJ_scan:

    H = tilted_hamiltonian(FJ, N, J, a)

    E, V = np.linalg.eigh(H)

    # average RMS width over all eigenstates
    sigma = []

    for k in range(N):
        sigma.append(rms_width(V[:, k], sites))

    widths.append(np.mean(sigma))

plt.figure(figsize=(6,4))
plt.loglog(FJ_scan, widths, 'o-')
plt.xlabel(r"$F/J$")
plt.ylabel("mean RMS width")
plt.title("Localization length of Wannier-Stark states")
plt.grid(True)
plt.show()