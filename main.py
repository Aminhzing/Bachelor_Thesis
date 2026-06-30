import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import hbar
from scipy.integrate import simpson
import matplotlib.pyplot as plt

U = 10
a = 1
G = 2*np.pi
dim = 50
k_L = np.pi


m_vals = np.arange(-dim, dim+1)
r_vals = np.linspace(-4*a,4*a,500)
k_vals = np.linspace(-np.pi/a,np.pi/a,100)
r_valscell = np.linspace(-a/2,a/2, 100)

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

def calc_w_k():
    u_k = calc_u_k()
    w = np.zeros( len(r_vals), dtype= complex)
    for idr, r in enumerate(r_vals):
        for idk, k in enumerate(k_vals):
            w[idr] += np.exp(1j * k  * r) * u_k[idk][idr] / len(k_vals)
    return w

def calc_pot():
    A_0 = U
    A_p = -U
    Rms_p = 0.05
    V_0 = []
    Pert = []
    for idr, r in enumerate(r_vals):
        V_0.append(-A_0*np.cos(G*r))
    for idr, r in enumerate(r_vals):
        Pert.append(A_p * np.exp(-((r - a / 2) ** 2) / (2 * Rms_p)))
    return V_0, Pert

def calc_mu_t():
    def disprel(k, mu, t):
        y = -2*t*np.cos(k*a) + mu
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

def plot_w_0():
    V_0 , pert = calc_pot()
    pertpot = []
    for idr in range(len(r_vals)):
        pertpot.append( V_0[idr] + pert[idr])

    plt.figure()
    plt.plot(r_vals, pertpot, label="Potential")
    plt.plot(r_vals, calc_w_k(), label="Wannier (without units)")
    plt.plot(r_vals, pert, label ="Perturbation")
    plt.legend()
    plt.ylabel("U/E$_{rec}$")
    plt.xlabel("x/a")
    plt.show()

def calc_dmu_dt():
    dt = 0
    dr = r_vals[1] - r_vals[0]
    indlenofsite = len(r_vals) // (r_vals[len(r_vals)-1] - r_vals[0])
    w_0 = calc_w_k()
    _ , pert = calc_pot()
    w_1 = np.roll(w_0, indlenofsite)
    for idr in range(int(indlenofsite)):
        w_1[idr] = 0

    dmu = simpson(np.abs(w_0)**2*pert, r_vals)
    print( 'dmu =', dmu, 'E_rec')
    for idr in range(len(r_vals)):
        dt += dr * np.conj(w_0[idr]) * w_1[idr] * pert[idr]
    print('dt =', np.real(dt), 'E_rec' )

    #plt.plot(r_vals, pert, '-', label='pert')
    #plt.plot(r_vals, np.real(w_1), '-', label='w_1')
    #plt.plot(r_vals, np.real(w_0) , '-', label='w_0')
    #plt.show()

plot_disp()
