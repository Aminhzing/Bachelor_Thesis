import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import matplotlib.pyplot as plt

U = 1
a = 5
G = 2*np.pi/a
dim = 50

m_vals = np.arange(-dim, dim+1)
r_vals = np.linspace(-a/2,a/2,100)
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
    V = []
    for idr, r in enumerate(r_vals):
        V.append(-np.cos(G*r)+np.exp(-((r-a/2)**2)))
    return V

def disprel(k, mu, t):
    y = -2*t*np.cos(k*a) + mu
    return y

parameters, covariance = curve_fit(disprel, k_vals, calculating_E_k())
fit_mu = parameters[0]
fit_t = parameters[1]
fit_disprel = disprel(k_vals, fit_mu, fit_t)
#print(fit_mu)
#print(fit_t)
SE = np.sqrt(np.diag(covariance))
SE_A = SE[0]
SE_B = SE[1]
#print(F'The value of A is {fit_mu:.5f} with standard error of {SE_A:.5f}.')
#print(F'The value of B is {fit_t:.5f} with standard error of {SE_B:.5f}.')

w_0 = calculating_w_k()
u_k = calculating_u_k()

#y_1 = np.abs(w_0)
y_2 = np.real(w_0)
#y_3 = np.imag(w_0)
#y_4 = calculating_E_k()
y_5 = calculating_cos_pot()


#plt.plot(r_vals, y_5, label="Potential")
plt.plot(r_vals, y_2, label="Wannier")
#plt.plot(k_vals, y_4, label = "E_k")
#plt.plot(k_vals, calculating_E_k(), 'o', label='data')
#plt.plot(k_vals, fit_disprel, '-', label='fit')

plt.legend()
plt.xlim(-2*a,2*a)
plt.show()