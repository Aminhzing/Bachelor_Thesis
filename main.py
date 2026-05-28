import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import matplotlib.pyplot as plt

U = 1
a = 15
G = 2*np.pi/a
dim = 50

m_vals = np.arange(-dim, dim+1)
r_vals = np.linspace(-2*a,2*a,1000)
k_vals = np.linspace(-np.pi/a,np.pi/a,100)


def calculating_c_k_m():
    main_diag = np.zeros((len(k_vals), len(m_vals)))
    sub_diag = np.full(2*dim, -U / 2)
    Matrix = []
    gs_c_k_m = []
    for idk, k in enumerate(k_vals):
        for idj, m in enumerate(m_vals):
            main_diag[idk][idj] = (k+m*G)**2
        Matrix.append(np.diag(main_diag[idk]) + np.diag(sub_diag, 1) + np.diag(sub_diag, -1))
        eigvals, eigvecs = np.linalg.eigh(Matrix[idk])
        gs_c_k_m.append(eigvecs[:, 0])
    return gs_c_k_m

def calculating_E_k():
    Eigenenergies = []
    main_diag = np.zeros((len(k_vals), len(m_vals)))
    sub_diag = np.full(2 * dim, -U / 2)
    Matrix = []
    for idk, k in enumerate(k_vals):
        for idj, m in enumerate(m_vals):
            main_diag[idk][idj] = (k+m*G)**2
        Matrix.append(np.diag(main_diag[idk]) + np.diag(sub_diag, 1) + np.diag(sub_diag, -1))
        Eigenenergies.append(np.linalg.eigvalsh(Matrix[idk])[0])
    return Eigenenergies

def calculating_u_k():
    c_k_m = calculating_c_k_m()
    u_k = np.zeros((len(k_vals), len(r_vals)), dtype= complex)
    for idk, k in enumerate(k_vals):
        for idr, r in enumerate(r_vals):
            x = 0j
            for idm, m in enumerate(m_vals):
                x += np.exp(1j*m*G*r) * c_k_m[idk][idm]
            u_k[idk, idr] = x / np.sqrt(r_vals[len(r_vals)-1] - r_vals[0])
        area = simpson(np.abs(u_k[idk])**2, r_vals)
        #print(area)
    return u_k

def calculating_w_k():
    u_k = calculating_u_k()
    w = np.zeros( len(r_vals), dtype= complex)
    for idr, r in enumerate(r_vals):
        for idk, k in enumerate(k_vals):
            w[idr] += np.exp(1j * k  * r) * u_k[idk][idr] / len(k_vals)
    area = simpson(np.abs(w) ** 2, r_vals)
    print(area)
    return w

def calculating_cos_pot():
    Amp = -0.5
    V = []
    for idr, r in enumerate(r_vals):
        V.append(-np.cos(G*r)+Amp*np.exp(-((r-a/2)**2)))
    return V

def disprel(k, mu, t):
    y = -2*t*np.cos(k*a) + mu
    return y

parameters, covariance = curve_fit(disprel, k_vals, calculating_E_k())
fit_mu = parameters[0]
fit_t = parameters[1]
fit_disprel = disprel(k_vals, fit_mu, fit_t)

SE = np.sqrt(np.diag(covariance))
SE_A = SE[0]
SE_B = SE[1]
print(F' mu is {fit_mu:.5f} with standard error of {SE_A:.5f}.')
print(F' t is {fit_t:.5f} with standard error of {SE_B:.5f}.')


plt.figure()
plt.plot(k_vals, calculating_E_k(), 'o', label='data')
plt.plot(k_vals, fit_disprel, '-', label='fit')
plt.legend()



plt.figure()
plt.plot(r_vals, calculating_cos_pot(), label="Potential")
plt.plot(r_vals, np.real(calculating_w_k()), label="Wannier")
plt.xlim(-2*a,2*a)
plt.legend()

plt.show()