import numpy as np
from scipy.optimize import brentq,newton

def numbern(mu_, nn_, beta_, energies_):
    e_aux = np.copy(energies_)
    e_aux[beta_ * (e_aux - mu_) > 30] = 30 / beta_ + mu_
    ferm = 1 / (np.exp(beta_ * (e_aux - mu_)) + 1)
    return np.sum(ferm) - nn_

class checkerboard_lattice_4unitcell:
    def __init__(self, nx, ny, t0, jax, jay, jbx, jby, v1, v2, beta, cell_filling):
        self.nx  = nx # unit cells in x direction
        self.ny  = ny # unit cells in y direction
        self.L   = self.nx*self.ny
        self.L_sites = 4*self.L
        self.t0  = t0
        self.jax = jax
        self.jay = jay
        self.jbx = jbx
        self.jby = jby
        self.v1  = v1
        self.v2  = v2
        self.beta = beta
        self.cell_filling  = cell_filling
        self.mu = 0.
        self.energies = None
        self.states   = None
        self.energies_fermi = None
        self.total_energy = 1000000.
        self.mfchi_ab   = 0.13+0.1j
        self.mfchi_ac   = 0.15-0.1j
        self.mfchi_cd   = 0.16-0.1j
        self.mfchi_bd   = 0.12+0.1j
        self.mfchi_adx   = -0.1
        self.mfchi_ady   = +0.1
        self.mfchi_bcx   = +0.1
        self.mfchi_bcy   = -0.1
        self.mfn_a      = 0.6
        self.mfn_b      = 0.4
        self.mfn_c      = 0.6
        self.mfn_d      = 0.4
        self.tre = 1E-10
        self.e_re = np.array([])

    def update_solve_hamiltonian(self, k1, k2):
        H = np.zeros((4, 4), dtype=complex)

        H[0, 1] = -2*(self.t0 + self.v1*np.real(self.mfchi_ab))*np.cos(k2) \
                               - 1j*2*self.v1*np.imag(self.mfchi_ab)*np.cos(k2)
        H[0, 2] = -2*(self.t0 + self.v1*np.real(self.mfchi_ac))*np.cos(k1) \
                               - 1j*2*self.v1*np.imag(self.mfchi_ac)*np.cos(k1)

        H[0, 3] = 2*(self.jax-self.v2*self.mfchi_adx)*np.cos(k1+k2) \
                + 2*(self.jay-self.v2*self.mfchi_ady)*np.cos(k1-k2)

        H[1, 2] = 2*(self.jbx-self.v2*self.mfchi_bcx)*np.cos(k1+k2) \
                + 2*(self.jby-self.v2*self.mfchi_bcy)*np.cos(k1-k2)

        H[1, 3] = -2*(self.t0 + self.v1*np.real(self.mfchi_bd))*np.cos(k1) \
                               - 1j*2*self.v1*np.imag(self.mfchi_bd)*np.cos(k1)
        H[2, 3] = -2*(self.t0 + self.v1*np.real(self.mfchi_cd))*np.cos(k2) \
                               - 1j*2*self.v1*np.imag(self.mfchi_cd)*np.cos(k2)

        H += np.conjugate(np.transpose(H))

        H[0, 0] = 2*self.v1*(self.mfn_b+self.mfn_c) + 4*self.v2*self.mfn_d
        H[1, 1] = 2*self.v1*(self.mfn_a+self.mfn_d) + 4*self.v2*self.mfn_c
        H[2, 2] = 2*self.v1*(self.mfn_a+self.mfn_d) + 4*self.v2*self.mfn_b
        H[3, 3] = 2*self.v1*(self.mfn_b+self.mfn_c) + 4*self.v2*self.mfn_a


        energies, U_t = np.linalg.eigh(H)
        idx = energies.argsort()
        energies = energies[idx]
        U_t = U_t[:, idx]
        #U = np.transpose(np.conjugate(np.copy(U_t)))
        # print(np.dot(np.dot(U, H), Ut)) should be diagonal

        return energies, U_t



    def reset_mfparams(self):
        self.total_energy, self.mfn_a, self.mfn_b, self.mfn_c, self.mfn_d, self.mfchi_ab, \
        self.mfchi_ac, self.mfchi_bd, self.mfchi_cd, self.mfchi_adx, self.mfchi_ady, self.mfchi_bcx,\
        self.mfchi_bcy= 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.


    def update_mu(self):
        energies_ = self.energies_fermi.flatten
        [a, b] = [np.amin(self.energies_fermi), np.amax(self.energies_fermi)]
        #[a,b]=[EE[int(nn-nn/6)],EE[int(nn+nn/6)]]
        self.mu = brentq(numbern, a, b, args=(int(self.cell_filling*self.L), self.beta, self.energies_fermi))
        #    mu=newton(numbern_un,EE[int(nn)],args=(nn,beta,EE),tol=1E-6,maxiter=100)


    def iterate_mf(self):
        k1 = np.arange(0, np.pi, np.pi / self.nx)
        k2 = np.arange(0, np.pi, np.pi / self.ny)
        K1, K2 = np.meshgrid(k1, k2)
        K1, K2 = K1.flatten(), K2.flatten()

        self.energies = np.zeros((4, K1.size))
        self.states   = np.zeros((4, 4, K1.size), dtype=complex)
        for i1 in range(0, K1.size):
            sol_ = self.update_solve_hamiltonian(K1[i1], K2[i1])
            self.energies[:, i1] = sol_[0]
            self.states[:, :, i1]= sol_[1]

        self.energies_fermi = np.reshape(np.copy(self.energies), 4*K1.size)
        self.update_mu()

        self.energies_fermi[self.beta*(self.energies_fermi-self.mu) > 30] = 30./self.beta + self.mu
        weights = 1. / (np.exp(self.beta * (self.energies_fermi - self.mu)) + 1)
        weights = np.reshape(weights,(4,K1.size))

        total_energy_, mfn_a_, mfn_b_, mfn_c_, mfn_d_, mfchi_ab_, mfchi_ac_, mfchi_bd_, mfchi_cd_, \
        mfchi_adx_, mfchi_ady_, mfchi_bcx_, mfchi_bcy_=  0., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0., 0.


        for i1 in range(0, K1.size):
            total_energy_ += np.sum(weights[:,i1]*self.energies[:,i1]) / self.L
            U_t = np.copy(self.states[:, :, i1])
            U_t = U_t*np.sqrt(weights[:,i1])
            U = np.transpose(np.conjugate(np.copy(U_t)))
            na_, nb_, nc_, nd_, corr_ab_, corr_ac_, corr_bd_, corr_cd_, corr_ad_, corr_bc_ = \
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            for i2 in range(0,4):
                na_ += U[i2, 0]*U_t[0, i2]
                nb_ += U[i2, 1]*U_t[1, i2]
                nc_ += U[i2, 2]*U_t[2, i2]
                nd_ += U[i2, 3]*U_t[3, i2]
                corr_ab_ += U[i2, 0] * U_t[1, i2]
                corr_ac_ += U[i2, 0] * U_t[2, i2]
                corr_cd_ += U[i2, 2] * U_t[3, i2]
                corr_bd_ += U[i2, 1] * U_t[3, i2]
                corr_ad_ += U[i2, 0] * U_t[3, i2]
                corr_bc_ += U[i2, 1] * U_t[2, i2]

            mfn_a_ += na_ / self.L
            mfn_b_ += nb_ / self.L
            mfn_c_ += nc_ / self.L
            mfn_d_ += nd_ / self.L

            mfchi_ab_ += np.exp(1j*K2[i1]) * corr_ab_/ self.L
            mfchi_ac_ += np.exp(1j*K1[i1]) * corr_ac_/ self.L
            mfchi_bd_ += np.exp(1j*K1[i1]) * corr_bd_/ self.L
            mfchi_cd_ += np.exp(1j*K2[i1]) * corr_cd_/ self.L
            mfchi_adx_ += np.cos(K1[i1]+K2[i1]) * corr_ad_ / self.L
            mfchi_ady_ += np.cos(K2[i1]-K1[i1]) * corr_ad_ / self.L
            mfchi_bcx_ += np.cos(K1[i1]+K2[i1]) * corr_bc_ / self.L
            mfchi_bcy_ += np.cos(K2[i1]-K1[i1]) * corr_bc_ / self.L

        self.mfn_a = mfn_a_
        self.mfn_b = mfn_b_
        self.mfn_c = mfn_c_
        self.mfn_d = mfn_d_
        self.mfchi_ab = mfchi_ab_
        self.mfchi_ac = mfchi_ac_
        self.mfchi_bd = mfchi_bd_
        self.mfchi_cd = mfchi_cd_
        self.mfchi_adx = mfchi_adx_
        self.mfchi_ady = mfchi_ady_
        self.mfchi_bcx = mfchi_bcx_
        self.mfchi_bcy = mfchi_bcy_

        self.total_energy = total_energy_ + \
        2*self.v1 * (np.abs(self.mfchi_ab)** 2 + np.abs(self.mfchi_ac)** 2 + np.abs(self.mfchi_bd)** 2 +\
                   np.abs(self.mfchi_cd)** 2 - (self.mfn_a+self.mfn_d)*(self.mfn_b+self.mfn_c)) +\
        2*self.v2*((self.mfchi_adx)** 2+ (self.mfchi_ady)** 2 + (self.mfchi_bcx)** 2 + (self.mfchi_bcy)** 2 -\
                2*(self.mfn_a*self.mfn_d+self.mfn_b*self.mfn_c))

        self.e_re = np.append(self.e_re, self.total_energy)
