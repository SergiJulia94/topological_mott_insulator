# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_Utils.ipynb (unless otherwise specified).

__all__ = ['numbern', 'chern', 'plot_bonds', 'plot_lattice', 'Rydberg_v3v4']

# Cell
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import matplotlib as mpl
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq,newton

# Cell
def numbern(mu_, nn_, beta_, energies_):
    e_aux = np.copy(energies_)
    e_aux[beta_ * (e_aux - mu_) > 30] = 30 / beta_ + mu_
    ferm = 1 / (np.exp(beta_ * (e_aux - mu_)) + 1)
    return np.sum(ferm) - nn_

# Cell
def chern(states, mx, my, band):
        J_kx = np.zeros((mx, my, mx, my), dtype=int)
        J_ky = np.zeros((mx, my, mx, my), dtype=int)
        for ix in range(0, mx):
            for iy in range(0, my):
                    J_kx[ix, iy, (ix+1)%mx, iy] = 1
                    J_ky[ix, iy, ix, (iy+1)%my] = 1

        J_kx = np.reshape(J_kx, (mx*my,mx*my))
        J_kx = np.argwhere(J_kx!=0)

        J_ky = np.reshape(J_ky, (mx*my,mx*my))
        J_ky = np.argwhere(J_ky!=0)

        U_up = np.einsum('ijk,ilk->jlk', np.conjugate(states[:, 0:int(band), J_ky[:, 0]]), states[:, 0:int(band), J_ky[:, 1]])
        U_right = np.einsum('ijk,ilk->jlk', np.conjugate(states[:, 0:int(band), J_kx[:, 0]]), states[:, 0:int(band), J_kx[:, 1]])

        U_u = np.zeros(mx*my, dtype=complex)
        U_r = np.zeros(mx*my, dtype=complex)

        for i1 in range(0, mx*my):
            U_u[i1] = np.linalg.det(U_up[:, :, i1])
            U_r[i1] = np.linalg.det(U_right[:, :, i1])

        U_u = U_u/np.abs(U_u)
        U_r = U_r/np.abs(U_r)


        U = U_r[J_kx[:,0]] * U_u[J_kx[:,1]] * (U_r[J_ky[:,1]])**(-1) * (U_u[J_kx[:,0]])**(-1)

        F = np.log(U)
        chern_band = 1/(2*np.pi*1j)*np.sum(F)
        return chern_band, F

# Cell
def plot_bonds(vecpos, J_, mf_):
    segment = []
    color = []
    lenght = np.size(J_, 0)
    for i1 in np.arange(lenght):
        [x1, x2] = [vecpos[J_[i1, 0], 0], vecpos[J_[i1, 1], 0]]
        [y1, y2] = [vecpos[J_[i1, 0], 1], vecpos[J_[i1, 1], 1]]
        if np.abs(x1 - x2) < 4 and np.abs(y1 - y2) < 4:
            segment.append(np.array([(x1, y1), (x2, y2)]))
            color.append(mf_[i1])


    color = np.array(color)

    mini, maxi = min(np.amin(color), 1.01 * np.mean(color), 0.99 * np.mean(color)), max(np.amax(color),
                                                                                        1.01 * np.mean(color),
                                                                                        0.99 * np.mean(color))
    return segment, color, mini, maxi

# Cell
def plot_lattice(posx_, posy_, color_, title_):

    plt.figure()
    plt.scatter(posx_, posy_, c=color_, s=50)
    plt.colorbar()
    plt.title(title_)
    plt.show()
    plt.close()

# Cell
def Rydberg_v3v4(v1_,v2_):
    tau = np.copy(v1_/v2_)
    alpha = ((tau-1)/(8-tau))**(1/6)
    kappa = v1_*(1+alpha**6)
    v3 = kappa*(1/(1+(2*alpha)**6))
    v4 = kappa*(1/(1+(np.sqrt(5)*alpha)**6))
    return v3, v4