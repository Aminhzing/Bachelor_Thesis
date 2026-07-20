import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
from scipy.sparse.linalg import expm_multiply, expm

U = 5
a = 1
G = 2*np.pi
dim = 50
k_L = np.pi
band_cutoff = 5
sites = 10
stepsize = 0.001
cutoff = 50000

m_vals = np.arange(-dim, dim+1)
r_vals = np.linspace(-20*a,20*a,4001)
k_vals = np.linspace(-np.pi/a,np.pi/a,1000, endpoint=False)
band_vals = np.arange(0, band_cutoff)

def time_this(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.6f}s")
        return result
    return wrapper

def calc_c_k_m():
    main_diag = np.zeros((len(k_vals), len(m_vals)))
    sub_diag = np.full(2*dim, -U / 2)
    Matrix = []
    gs_c_k_m = []
    for idk, k in enumerate(k_vals):
        for idj, m in enumerate(m_vals):
            main_diag[idk][idj] = ((k+m*G)/k_L)**2
        Matrix.append(np.diag(main_diag[idk]) + np.diag(sub_diag, 1) + np.diag(sub_diag, -1))
        eigvals, eigvecs = np.linalg.eigh(Matrix[idk])
        gs_c_k_m.append(eigvecs[:, 0])
    return gs_c_k_m

@time_this
def calc_c_k_n_m():  #produces c_k_n_m[k][n][m]
    main_diag = np.zeros((len(k_vals), len(m_vals)))
    sub_diag = np.full(2*dim, -U / 2)
    Matrix = []
    c_k_n_m = []
    for idk, k in enumerate(k_vals):
        c_n_m = []
        for idm, m in enumerate(m_vals):
            main_diag[idk][idm] = ((k+m*G)/k_L)**2
        Matrix.append(np.diag(main_diag[idk]) + np.diag(sub_diag, 1) + np.diag(sub_diag, -1))
        _, eigvecs = np.linalg.eigh(Matrix[idk])

        for idn in range(band_cutoff):
            c_n_m.append(eigvecs[:, idn].astype(complex))
        c_k_n_m.append(c_n_m)

    for idn in range(band_cutoff):  #Gauge smoothing
       for idk in range(len(k_vals)):
            overlap = np.vdot(c_k_n_m[idk - 1][idn], c_k_n_m[idk][idn])

            c_k_n_m[idk][idn] *= np.exp(-1j * np.angle(overlap))

    return c_k_n_m

def calc_E_k():
    Eigenenergies = []
    main_diag = np.zeros((len(k_vals), len(m_vals)))
    sub_diag = np.full(2 * dim, -U / 2)
    Matrix = []
    for idk, k in enumerate(k_vals):
        for idj, m in enumerate(m_vals):
            main_diag[idk][idj] = ((k+m*G)/k_L)**2
        Matrix.append(np.diag(main_diag[idk]) + np.diag(sub_diag, 1) + np.diag(sub_diag, -1))
        Eigenenergies.append(np.linalg.eigvalsh(Matrix[idk])[0])
    return Eigenenergies

def calc_u_k():
    c_k_m = calc_c_k_m()
    u_k = np.zeros((len(k_vals), len(r_vals)), dtype= complex)
    for idk in range(len(k_vals)):
        for idr, r in enumerate(r_vals):
            x = 0j
            for idm, m in enumerate(m_vals):
                x += np.exp(1j*m*G*r) * c_k_m[idk][idm]
            u_k[idk][idr] = x
    return u_k

@time_this
def calc_u_n_k():
    c_k_n_m = calc_c_k_n_m()
    phase = np.exp(1j * np.outer(m_vals * G, r_vals))

    u_n_k = np.einsum('knm,mr->nkr', c_k_n_m, phase) #u_n_k[n][k][r]

    return u_n_k

def calc_w_0():
    u_k = calc_u_k()
    w_0 = np.zeros( len(r_vals), dtype= complex)
    for idr, r in enumerate(r_vals):
        for idk, k in enumerate(k_vals):
            w_0[idr] += np.exp(1j * k  * r) * u_k[idk][idr] / len(k_vals)
    return w_0

def calc_w_n_0(): #produces w_n_0[n][r]
    u_n_k = calc_u_n_k()
    w_n_0 = np.zeros((len(band_vals), len(r_vals)), dtype= complex)
    for idn in range(len(band_vals)):
        for idr, r in enumerate(r_vals):
            for idk, k in enumerate(k_vals):
                w_n_0[idn][idr] += np.exp(1j * k  * r) * u_n_k[idn][idk][idr] / len(k_vals)
    return w_n_0

def calc_pot():
    A_0 = U
    A_p = -U/10
    Rms_p = 0.03
    V_0 = []
    Pert = []
    for idr, r in enumerate(r_vals):
        V_0.append(-A_0*np.cos(G*r))
    for idr, r in enumerate(r_vals):
        Pert.append(A_p * np.exp(-((r - a / 2) ** 2) / (2 * Rms_p)))
    return V_0, Pert

def calc_mu_t():
    def disprel(k, mu, t):
        y = 2*t*np.cos(k*a) - mu
        return y

    parameters, covariance = curve_fit(disprel, k_vals, calc_E_k())
    fit_mu = parameters[0]
    fit_t = parameters[1]
    fit_disprel = disprel(k_vals, fit_mu, fit_t)

    SE = np.sqrt(np.diag(covariance))
    SE_mu = SE[0]
    SE_t = SE[1]
    print(f"mu = ({fit_mu:.5f} ± {SE_mu:.5f}) E_rec")
    print(f"t  = ({fit_t:.5f} ± {SE_t:.5f}) E_rec")
    return fit_disprel

def plot_disp():
    E_k = calc_E_k()
    fit = calc_mu_t()
    plt.figure()
    plt.plot(k_vals, E_k, '-', label='E_k')
    plt.plot(k_vals, fit, '-', label='fit')
    plt.legend()
    plt.ylabel("U/E$_{rec}$")
    plt.xlabel("k/k$_L$")
    plt.show()

@time_this
def calc_w_n_i(): #produces w_n_i[n][i][r]
    u_n_k = calc_u_n_k() #u_n_k[n][k][r]
    site_vals = np.arange(sites)
    phase = np.exp(1j * k_vals[:, None, None] *(r_vals[None, None, :] - site_vals[None, :, None]))
    w_n_i = np.einsum('kir,nkr->nir',phase, u_n_k) / len(k_vals)

    for n in range(band_cutoff):
        w = w_n_i[n, 0]

        # find point with largest amplitude
        idx = np.argmax(np.abs(w))

        # rotate point to be real
        phase = np.angle(w[idx])

        w_n_i[n, :] *= np.exp(-1j * phase)
    return w_n_i

@time_this
def calc_mu_t_n():
    w_n_i = calc_w_n_i() #want to produce w_n_i[n][i][r]
    H = np.zeros((len(band_vals), len(band_vals), sites, sites), dtype = complex) #want to produce H[n][m][i][j]
    V_0, V_1 = calc_pot()
    for idm in range(len(band_vals)):
        for idj in range(sites):
            kin = -np.gradient( np.gradient(w_n_i[idm][idj],r_vals),r_vals) / k_L ** 2
            for idn in range(len(band_vals)):
                for idi in range(sites):
                    H[idn][idm][idi][idj] = simpson(    np.conj(w_n_i[idn][idi])    *      (kin + V_0* w_n_i[idm][idj] + V_1 * w_n_i[idm][idj]) , r_vals       )
    return H


def calc_dmu_dt():
    dt = 0
    dr = r_vals[1] - r_vals[0]
    indlenofsite = len(r_vals) // (r_vals[len(r_vals)-1] - r_vals[0])
    w_0 = calc_w_0()
    _ , pert = calc_pot()
    w_1 = np.roll(w_0, indlenofsite)
    for idr in range(int(indlenofsite)):
        w_1[idr] = 0
    dmu = simpson(np.abs(w_0)**2*pert, r_vals)
    print( 'dmu =', dmu, 'E_rec')
    for idr in range(len(r_vals)):
        dt += dr * np.conj(w_0[idr]) * w_1[idr] * pert[idr]
    print('dt =', np.real(dt), 'E_rec' )

def plot_w_0():
    V_0 , pert = calc_pot()
    #w_n_0 = calc_w_n_0()
    #u_n_k = calc_u_n_k()
    #w_n_i = calc_w_n_i()
    w_n_i_prime, _ = evolve_w_n_i()
    pertpot = []
    for idr in range(len(r_vals)):
        pertpot.append( V_0[idr] + pert[idr])

    plt.figure()
    #plt.plot(r_vals, pertpot, label="Potential")
    #plt.plot(r_vals, np.real(w_n_i[0][9]), label="2")
    #plt.plot(r_vals, np.imag(w_n_i[0][9]), label="3")
    #plt.plot(r_vals, np.abs(w_n_i[0][9])**2, label="4")
    #plt.plot(r_vals, np.real(w_n_i_prime[2][0]), label = "5")
    #plt.plot(r_vals, np.real(w_n_i_prime[2][0]), label = "6")
    plt.plot(r_vals, np.abs(w_n_i_prime[2][0]), label="7")
    #plt.plot(r_vals, pert, label ="Perturbation")
    plt.legend()
    plt.ylabel("U/E$_{rec}$")
    plt.xlabel("x/a")
    plt.show()

@time_this
def wegner_flow():
    # H[n][m][i][j]
    H_0 = np.real(calc_mu_t_n())


    #Commutator
    def commutator(A,B):
        return np.einsum('npik,pmkj->nmij', A, B) - np.einsum('npik,pmkj->nmij', B, A)

    #Define G[n][m][i][j]
    G1 = np.zeros((band_cutoff, band_cutoff, sites, sites))
    id = np.eye(sites)
    for idn in range(band_cutoff):
        if(idn == 0):
            G1[idn][idn] = id
        else:
            G1[idn][idn] = 2*id
    #Calculate Band gap
    #H_exc = (H_0[1:, 1:].transpose(0, 2, 1, 3).reshape((len(band_vals)-1) * sites, (len(band_vals)-1) * sites))
    #H_ground = H_0[0, 0]
    #ground_eigs = np.linalg.eigvals(H_ground)
    #excited_eigs = np.linalg.eigvals(H_exc)
    #highest_ground = ground_eigs.max()
    #lowest_excited = excited_eigs.min()
    #d = lowest_excited - highest_ground
    #print(d)

    #Iteratation
    H_prime = H_0
    eta_I = []
    #coupling = []

    for i in range(cutoff):
        #for idn in range(len(band_vals)):
            #G2[idn][idn] = H_prime[idn][idn]
        #c_I = np.zeros((2 * len(band_vals) - 2, sites, sites)) #produces coupling[idn][idi][idj][i]
        #for idi in range(sites):
         #   for idj in range(sites):
          #      for idn in range(1, len(band_vals) ):
           #         c_I[idn-1][idi][idj] = H_prime[0][idn][idi][idj]
            #    for idm in range(1, len(band_vals)):
             #       c_I[len(band_vals)-1 + idm-1][idi][idj] = H_prime[idm][0][idi][idj]
        #coupling.append(c_I)
        eta = commutator(G1, H_prime)
        dH = commutator(eta, H_prime)
        H_prime = H_prime + stepsize*dH
        eta_I.append(eta)
    print(H_prime)
    return H_prime, eta_I

@time_this
def plot_wegner():
    H_prime, _ = wegner_flow()
    #H_0 = calc_mu_t_n()
    H_big_prime = H_prime.transpose(0, 2, 1, 3).reshape(band_cutoff * sites, band_cutoff * sites)
    #H_big_0 = H_0.transpose(0, 2, 1, 3).reshape(band_cutoff * sites, band_cutoff * sites)
    cmap = plt.cm.magma.copy()
    cmap.set_bad("black")
    plt.figure(figsize=(sites*band_cutoff, sites*band_cutoff))
    plt.imshow(np.abs(H_big_prime), origin ='upper', cmap = cmap, norm = LogNorm(vmin=1e-14, vmax=np.max(H_big_prime)))

    plt.colorbar(label=r"$|H_{ij}|$")
    plt.xlabel("State index")
    plt.ylabel("State index")
    plt.title("Hamiltonian")
    plt.tight_layout()

    plt.show()

@time_this
def evolve_w_n_i():
    _, eta_I = wegner_flow() #eta_I[n][m][i][j][I]
    eta_I = np.array(eta_I)
    w_n_i = calc_w_n_i() # w_n_i[n][i][r]
    w_flat = w_n_i.reshape(band_cutoff*sites, len(r_vals))
    eta_big = eta_I.transpose(0, 2, 1, 3, 4).reshape(cutoff, band_cutoff * sites, band_cutoff * sites)
    for I in range(cutoff):
        U = expm(-eta_big[I] * stepsize)
        w_flat = U @ w_flat

    w_n_i_prime = w_flat.reshape(band_cutoff, sites, len(r_vals))

    overlap = simpson(np.conj(w_n_i_prime[2][0])* w_n_i[2][0], r_vals)
    print(overlap)
    print(simpson(np.conj(w_n_i_prime[2][0])* w_n_i_prime[2][0], r_vals))
    return w_n_i_prime, overlap

#H_0 = calc_mu_t_n()
#H_prime, _ = wegner_flow()
plot_wegner()
evolve_w_n_i()
#plot_w_0()