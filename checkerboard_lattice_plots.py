import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import os
os.environ["OMP_NUM_THREADS"] = "18"
import numpy as np

from tqdm import tqdm 
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import pickle
import time
import matplotlib.colors as colors
import matplotlib as mpl
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



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

def plot_lattice(posx_, posy_, color_, title_):
    
    plt.figure()
    plt.scatter(posx_, posy_, c=color_, s=50)
    plt.colorbar()
    plt.title(title_)
    plt.show()
    plt.close()
    

    
# fig  = plt.figure(figsize=(0.55*2*8.646/2.54, 0.9*2.2*8.646/(1.68*2.54)))
# gs0 = gridspec.GridSpec(1, 1, left=0.12, right=0.75, top=1., bottom=0., wspace=0)
# gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0], hspace=0., wspace=0.)
# ax = fig.add_subplot(gs00[0])
# sc = ax.scatter(un_mf.pos[:,0],un_mf.pos[:,1], c=np.real(un_mf.mfden[new_pos]), s=20, cmap='viridis')
# ax.set_aspect(aspect='equal')
# ax.minorticks_on()
# ax.set_xlabel(r'$x$', fontsize=10)
# ax.set_ylabel(r'$y$', fontsize=10)
# cbar = fig.add_axes([0.77, 0.2, 0.06, 0.6])
# cb1 = fig.colorbar(sc, cax=cbar, orientation='vertical')
# plt.title(r'$n(j)$', fontsize=14)

# plt.savefig('/home/sjulia/Desktop/GitTopoMott/Plots/density_polaron.pdf',dpi=200)
# plt.show()


# fig  = plt.figure(figsize=(0.55*2*8.646/2.54, 0.9*2.2*8.646/(1.68*2.54)))
# gs0 = gridspec.GridSpec(1, 1, left=0.12, right=0.75, top=1., bottom=0., wspace=0)
# gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0], hspace=0., wspace=0.)
# ax = fig.add_subplot(gs00[0])

# segment, color, mini, maxi = tools.plot_bonds(un_mf.pos, un_mf.J_nn, np.imag(un_mf.mfhop_nn))
# ligne = LineCollection(segment,linestyles='solid',
#                                 cmap=plt.get_cmap('RdBu'),
#                                 array=color, norm=plt.Normalize(mini, maxi),
#                                 linewidths=3)
 
# ax.set_xlim(0,2*un_mf.ny)
# ax.set_ylim(0,2*un_mf.nx)
# ax.add_collection(ligne)
# ax.set_xlabel(r'$x$', fontsize=10)
# ax.set_ylabel(r'$y$', fontsize=10)
# cbar = fig.add_axes([0.77, 0.2, 0.06, 0.6])
# fig.colorbar(ligne, cax=cbar)
# # circle1 = plt.Circle((un_mf.nx-1, un_mf.ny-1), 5, color='r',alpha=0.4)
# # ax.add_artist(circle1)
# ax.set_aspect('equal')
# plt.title(r'$\xi_{AB}^I(j)$', fontsize=14)
# plt.savefig('/home/sjulia/Desktop/GitTopoMott/Plots/loop_polaron.pdf',dpi=200)
# plt.show()


# fig  = plt.figure(figsize=(0.55*2*8.646/2.54, 0.9*2.2*8.646/(1.68*2.54)))
# gs0 = gridspec.GridSpec(1, 1, left=0.12, right=0.75, top=0.8, bottom=0., wspace=0)
# gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0], hspace=0., wspace=0.)
# ax = fig.add_subplot(gs00[0])
# ax.plot(np.sort(re_mf_2.energies.flatten())[550:610], '.', ms=10,label=r'restricted')
# ax.plot(un_mf.energies[550:610],'.', fillstyle='none', lw=10,ms=20 ,label='unrestricted')
# ax.plot(un_mf.energies_fermi[550:610],'.', label='occupied')
# ax.set_aspect('equal')
# plt.legend()
# plt.savefig('/home/sjulia/Desktop/GitTopoMott/Plots/spectrum_polaron.pdf',dpi=200)
# plt.show()