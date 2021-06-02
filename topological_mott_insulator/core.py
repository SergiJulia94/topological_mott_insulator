# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['checkerboard_lattice_un', 'checkerboard_lattice_4unitcell', 'numbern', 'chern']

# Cell
class checkerboard_lattice_un:
    def __init__(self, nx, ny, t0, jax, jay, jbx, jby, v1, v2, beta, cell_filling, phix, phiy):
        self.tre = 1E-10
        self.nx  = nx
        self.ny  = ny
        self.L   = self.nx*self.ny
        self.L_sites = 2*self.L
        self.t0  = t0
        self.jax = jax
        self.jay = jay
        self.jbx = jbx
        self.jby = jby
        self.v1  = v1
        self.v2  = v2
        self.beta = beta
        self.cell_filling  = cell_filling
        self.filling = int(self.L*self.cell_filling)
        self.mu = 0.
        self.energies = np.zeros(self.L_sites)
        self.energies_fermi = np.array([])
        self.total_energy = 1E+10
        self.states = np.array([])
        self.states_fermi = np.array([])
        self.J_nn = None
        self.J_nn_1 = None
        self.J_nn_2 = None
        self.J_ax = None
        self.J_ay = None
        self.J_bx = None
        self.J_by = None
        self.J_nn_tw = None
        self.J_ax_tw = None
        self.J_ay_tw = None
        self.J_bx_tw = None
        self.J_by_tw = None
        self.mfden  = None
        self.mfhop_nn = None
        self.mfhop_ax = None
        self.mfhop_ay = None
        self.mfhop_bx = None
        self.mfhop_by = None
        self.mfden_0  = None
        self.mfhop_nn_0 = None
        self.mfhop_ax_0 = None
        self.mfhop_ay_0 = None
        self.mfhop_bx_0 = None
        self.mfhop_by_0 = None
        self.pos = None
        self.posA = None
        self.posB = None
        self.lattice_positions()
        self.set_initcond()
        self.tre = 1E-10
        self.H  = np.zeros((self.L_sites, self.L_sites), dtype=complex)
        self.phix = phix
        self.phiy = phiy
        self.set_hoppings()
        self.set_initcond()
        self.c_mark = None
        self.iterations = int(0)
        self.etas = np.array([])
        self.e_un = np.array([])


    def set_initcond(self):
        self.mfden  = np.ones(self.L_sites, dtype=complex) + 0.1*np.random.rand(self.L_sites)
        self.mfden    = 0.5*self.L_sites*self.cell_filling*self.mfden/(np.sum(self.mfden))
        self.mfden_0  = np.copy(self.mfden)
        self.mfhop_nn = 0.16*np.ones(2*self.L_sites, dtype=complex) + 0.1*np.random.rand(2*self.L_sites)+1j*0.05*np.random.rand(2*self.L_sites)
        self.mfhop_ax = np.zeros(self.L, dtype=complex)
        self.mfhop_ay = np.zeros(self.L, dtype=complex)
        self.mfhop_bx = np.zeros(self.L, dtype=complex)
        self.mfhop_by = np.zeros(self.L, dtype=complex)
        self.mfhop_nn_0 = np.copy(self.mfhop_nn)
        self.mfhop_ax_0 = np.copy(self.mfhop_ax)
        self.mfhop_ay_0 = np.copy(self.mfhop_ay)
        self.mfhop_bx_0 = np.copy(self.mfhop_bx)
        self.mfhop_by_0 = np.copy(self.mfhop_by)



    def lattice_positions(self):
        x_, y_ = np.arange(0, self.nx, dtype=float), np.arange(0, 2*self.ny, dtype=float)
        x_, y_ = np.meshgrid(x_, y_)
        y_ = y_ + 0.5*(y_%2)
        x_ = x_ + 0.5*(y_%2)
        x_, y_ = (2*x_).astype(int), (y_).astype(int)
        vecpos = np.zeros((x_.size,2))
        vecpos[:,0] = x_.flatten()
        vecpos[:,1] = y_.flatten()
        self.pos = np.copy(vecpos)

    def traslation(self, x0, y0):
        new_pos = (0.5*((self.pos[:,0]-x0)%(2*self.nx)-((self.pos[:,1]-y0)%(2*self.ny))%2)+self.nx*     ((self.pos[:,1]-y0)%(2*self.ny))).astype(int)
        self.states = self.states[new_pos,:]
        self.states_fermi = self.states_fermi[new_pos,:]
        self.update_mfparams(eta=1.)

    def set_hoppings(self):
        # define hopping connections with flux in the borders (TBC)
        J = np.zeros((self.nx, 2*self.ny, self.nx, 2*self.ny), dtype=complex) ## nearest-neighbours
        for ix in range(0, self.nx):
            for iy in range(0, 2*self.ny):
                if iy%2 == 0:
                    J[ix, iy, (ix)%(self.nx), (iy+1)%(2*self.ny)] = 1
                    J[ix, iy, (ix)%(self.nx), (iy-1)%(2*self.ny)] = 1
                    if iy == 0:
                        J[ix, iy, (ix)%(self.nx), (iy-1)%(2*self.ny)] = np.exp(1j * (self.phix-self.phiy))
                else:
                    J[ix, iy, (ix+1)%(self.nx), (iy+1)%(2*self.ny)] = 1
                    J[ix, iy, (ix+1)%(self.nx), (iy-1)%(2*self.ny)] = 1
                    if iy == 2*self.ny-1:
                        J[ix, iy, (ix+1)%(self.nx), (iy+1)%(2*self.ny)] = np.exp(1j * (self.phix+self.phiy))
                    if ix == self.nx-1:
                        J[ix, iy, (ix+1)%(self.nx), (iy+1)%(2*self.ny)] = np.exp(1j * (self.phix+self.phiy))
                        J[ix, iy, (ix+1)%(self.nx), (iy-1)%(2*self.ny)] = np.exp(1j * (self.phix-self.phiy))

        J = np.reshape(J, (self.L_sites,self.L_sites), order='F')
        self.J_nn = np.argwhere(J!=0)
        self.J_nn_tw = J[self.J_nn[:,0],self.J_nn[:,1]]

        ## nearest-neighbours splitted in (x+y) and (x-y). This is needed for a vectorized addition of v_1*n_i*n_j terms in the Hamiltonian
        J_1 = np.zeros((self.nx, 2*self.ny, self.nx, 2*self.ny), dtype=complex) ## nn in (x+y) direction
        J_2 = np.zeros((self.nx, 2*self.ny, self.nx, 2*self.ny), dtype=complex) ## nn in (x-y) direction
        for ix in range(0, self.nx):
            for iy in range(0, 2*self.ny):
                if iy%2 == 0:
                    J_1[ix, iy, (ix)%(self.nx), (iy+1)%(2*self.ny)] = 1
                    J_2[ix, iy, (ix)%(self.nx), (iy-1)%(2*self.ny)] = 1
                else:
                    J_1[ix, iy, (ix+1)%(self.nx), (iy+1)%(2*self.ny)] = 1
                    J_2[ix, iy, (ix+1)%(self.nx), (iy-1)%(2*self.ny)] = 1

        J_1 = np.reshape(J_1, (self.L_sites,self.L_sites), order='F')
        self.J_nn_1 = np.argwhere(J_1!=0)

        J_2 = np.reshape(J_2, (self.L_sites,self.L_sites), order='F')
        self.J_nn_2 = np.argwhere(J_2!=0)


        J = np.zeros((self.nx, 2*self.ny, self.nx, 2*self.ny), dtype=complex)
        for ix in range(0, self.nx):
            for iy in range(0, 2*self.ny):
                if iy%2 == 0:
                    J[ix, iy, (ix+1)%(self.nx), iy] = 1
                    if ix == self.nx - 1:
                        J[ix, iy, (ix+1)%(self.nx), iy] = np.exp(1j * 2 *self.phix)

        J = np.reshape(J, (self.L_sites,self.L_sites), order='F')
        self.J_ax = np.argwhere(J!=0)
        self.J_ax_tw = J[self.J_ax[:,0],self.J_ax[:,1]]

        J = np.zeros((self.nx, 2*self.ny, self.nx, 2*self.ny), dtype=complex)
        for ix in range(0, self.nx):
            for iy in range(0, 2*self.ny):
                if iy%2 == 0:
                    J[ix, iy, ix, (iy+2)%(2*self.ny)] = 1
                    if iy == 2*self.ny - 2:
                        J[ix, iy, ix, (iy+2)%(2*self.ny)] = np.exp(1j * 2 *self.phiy)

        J = np.reshape(J, (self.L_sites,self.L_sites), order='F')
        self.J_ay = np.argwhere(J!=0)
        self.J_ay_tw = J[self.J_ay[:,0],self.J_ay[:,1]]

        J = np.zeros((self.nx, 2*self.ny, self.nx, 2*self.ny), dtype=complex)
        for ix in range(0, self.nx):
            for iy in range(0, 2*self.ny):
                if iy%2 == 1:
                    J[ix, iy, (ix+1)%(self.nx), iy] = 1
                    if ix == self.nx - 1:
                        J[ix, iy, (ix+1)%(self.nx), iy] = np.exp(1j * 2 *self.phix)

        J = np.reshape(J, (self.L_sites, self.L_sites), order='F')
        self.J_bx = np.argwhere(J!=0)
        self.J_bx_tw = J[self.J_bx[:, 0], self.J_bx[:, 1]]

        J = np.zeros((self.nx, 2*self.ny, self.nx, 2*self.ny), dtype=complex)
        for ix in range(0, self.nx):
            for iy in range(0, 2*self.ny):
                if iy%2 == 1:
                    J[ix, iy, ix, (iy+2)%(2*self.ny)] = 1
                    if iy == 2*self.ny - 1:
                        J[ix, iy, ix, (iy+2)%(2*self.ny)] = np.exp(1j * 2 *self.phiy)

        J = np.reshape(J, (self.L_sites, self.L_sites), order='F')
        self.J_by = np.argwhere(J!=0)
        self.J_by_tw = J[self.J_by[:, 0], self.J_by[:, 1]]

    def update_mu(self):
        [a, b] = [np.amin(self.energies_fermi), np.amax(self.energies_fermi)]
        #[a,b]=[EE[int(nn-nn/6)],EE[int(nn+nn/6)]]
        self.mu = brentq(numbern, a, b, args=(self.filling, self.beta, self.energies_fermi))
        #    mu=newton(numbern_un,EE[int(nn)],args=(nn,beta,EE),tol=1E-6,maxiter=100)

    def iterate_mf(self, eta):
        self.update_hamiltonian()
        self.diagonalize_hamiltonian()
        self.update_mfparams(eta=eta)
        self.etas = np.append(self.etas, eta)
        self.iterations += 1
        self.e_un = np.append(self.e_un, self.total_energy)

    def update_hamiltonian(self):
        self.H = np.zeros((self.L_sites, self.L_sites), dtype=complex)

        ### hopping terms
        self.H[self.J_nn[:, 0], self.J_nn[:, 1]] += self.J_nn_tw*(self.t0  - self.v1 * self.mfhop_nn)
        self.H[self.J_ax[:, 0], self.J_ax[:, 1]] += self.J_ax_tw*(self.jax - self.v2 * self.mfhop_ax)
        self.H[self.J_ay[:, 0], self.J_ay[:, 1]] += self.J_ay_tw*(self.jay - self.v2 * self.mfhop_ay)
        self.H[self.J_bx[:, 0], self.J_bx[:, 1]] += self.J_bx_tw*(self.jbx - self.v2 * self.mfhop_bx)
        self.H[self.J_by[:, 0], self.J_by[:, 1]] += self.J_by_tw*(self.jby - self.v2 * self.mfhop_by)

        self.H = self.H + np.conjugate(np.transpose(self.H))

        ### density diagonal terms
        self.H[self.J_nn_1[:, 0], self.J_nn_1[:, 0]] += self.v1 * self.mfden[self.J_nn_1[:, 1]]
        self.H[self.J_nn_1[:, 1], self.J_nn_1[:, 1]] += self.v1 * self.mfden[self.J_nn_1[:, 0]]
        self.H[self.J_nn_2[:, 0], self.J_nn_2[:, 0]] += self.v1 * self.mfden[self.J_nn_2[:, 1]]
        self.H[self.J_nn_2[:, 1], self.J_nn_2[:, 1]] += self.v1 * self.mfden[self.J_nn_2[:, 0]]
        self.H[self.J_ax[:, 0], self.J_ax[:, 0]] += self.v2 * self.mfden[self.J_ax[:, 1]]
        self.H[self.J_ax[:, 1], self.J_ax[:, 1]] += self.v2 * self.mfden[self.J_ax[:, 0]]
        self.H[self.J_ay[:, 0], self.J_ay[:, 0]] += self.v2 * self.mfden[self.J_ay[:, 1]]
        self.H[self.J_ay[:, 1], self.J_ay[:, 1]] += self.v2 * self.mfden[self.J_ay[:, 0]]
        self.H[self.J_bx[:, 0], self.J_bx[:, 0]] += self.v2 * self.mfden[self.J_bx[:, 1]]
        self.H[self.J_bx[:, 1], self.J_bx[:, 1]] += self.v2 * self.mfden[self.J_bx[:, 0]]
        self.H[self.J_by[:, 0], self.J_by[:, 0]] += self.v2 * self.mfden[self.J_by[:, 1]]
        self.H[self.J_by[:, 1], self.J_by[:, 1]] += self.v2 * self.mfden[self.J_by[:, 0]]



    def diagonalize_hamiltonian(self):
        self.energies, self.states = np.linalg.eigh(self.H)
        idx = self.energies.argsort()
        self.energies = self.energies[idx]
        self.states = self.states[:, idx]
        self.energies_fermi = np.copy(self.energies)
        self.update_mu()
        self.energies_fermi[self.beta*(self.energies_fermi-self.mu) > 30] = 30./self.beta + self.mu
        weights = 1. / (np.exp(self.beta * (self.energies_fermi - self.mu)) + 1)

        self.energies_fermi, self.states_fermi, weights = \
        self.energies_fermi[weights>self.tre], self.states[:, weights>self.tre], weights[weights>self.tre]

        self.states_fermi = np.copy(self.states_fermi*np.sqrt(weights))
        self.energies_fermi = self.energies_fermi*weights


    def update_mfparams(self, eta):
        mfhop_nn_new = np.conjugate(np.sum((np.multiply(np.conjugate(self.states_fermi[self.J_nn[:, 0], :]),
                                        self.states_fermi[self.J_nn[:, 1], :])), axis=1))
        mfhop_ax_new = np.conjugate(np.sum((np.multiply(np.conjugate(self.states_fermi[self.J_ax[:, 0], :]),
                                        self.states_fermi[self.J_ax[:, 1], :])), axis=1))
        mfhop_ay_new = np.conjugate(np.sum((np.multiply(np.conjugate(self.states_fermi[self.J_ay[:, 0], :]),
                                        self.states_fermi[self.J_ay[:, 1], :])), axis=1))
        mfhop_bx_new = np.conjugate(np.sum((np.multiply(np.conjugate(self.states_fermi[self.J_bx[:, 0], :]),
                                        self.states_fermi[self.J_bx[:, 1], :])), axis=1))
        mfhop_by_new = np.conjugate(np.sum((np.multiply(np.conjugate(self.states_fermi[self.J_by[:, 0], :]),
                                        self.states_fermi[self.J_by[:, 1], :])), axis=1))
        mfden_new    = np.sum(np.abs(self.states_fermi[:, :]) ** 2, axis=1)

        self.mfhop_nn = eta*mfhop_nn_new + (1-eta)*self.mfhop_nn
        self.mfhop_ax = eta*mfhop_ax_new + (1-eta)*self.mfhop_ax
        self.mfhop_ay = eta*mfhop_ay_new + (1-eta)*self.mfhop_ay
        self.mfhop_bx = eta*mfhop_bx_new + (1-eta)*self.mfhop_bx
        self.mfhop_by = eta*mfhop_by_new + (1-eta)*self.mfhop_by
        self.mfden    = eta*mfden_new + (1-eta)*self.mfden

        self.total_energy = np.sum(self.energies_fermi)
        self.total_energy += \
            self.v1*np.sum(np.abs(self.mfhop_nn)**2) - \
            self.v1 * np.dot(self.mfden[self.J_nn[:, 0]], self.mfden[self.J_nn[:, 1]]) \
            + self.v2 * np.sum(np.abs(self.mfhop_ax)**2+np.abs(self.mfhop_ay)**2+\
                                np.abs(self.mfhop_bx)**2+np.abs(self.mfhop_by)**2) \
            - self.v2 * np.dot(self.mfden[self.J_ax[:, 0]], self.mfden[self.J_ax[:, 1]]) \
            - self.v2 * np.dot(self.mfden[self.J_ay[:, 0]], self.mfden[self.J_ay[:, 1]]) \
            - self.v2 * np.dot(self.mfden[self.J_bx[:, 0]], self.mfden[self.J_bx[:, 1]]) \
            - self.v2 * np.dot(self.mfden[self.J_by[:, 0]], self.mfden[self.J_by[:, 1]])
        self.total_energy *= 1./(self.L) #energy per unit cells

    def Chern_loc(self, r, x0, y0):

        P = np.einsum('ij,kj->ik',(self.states_fermi),np.conjugate(self.states_fermi))
        Q = np.eye(self.L_sites)-np.copy(P)
        x, y = (np.copy(self.pos[:,0])-x0+self.nx)%(2*self.nx), (np.copy(self.pos[:,1])-y0+self.nx)%(2*self.ny)
        xq = np.dot(Q*x,P)
        yp = np.dot(P*y,Q)
        self.c_mark = -4*np.pi*np.imag(np.diagonal(np.dot(xq,yp)))
        self.c_mark =self.c_mark[np.argwhere(self.pos[:,1]%2==0)]+self.c_mark[np.argwhere(self.pos[:,1]%2!=0)]

        dis = self.disk(r,x0,y0)
        chern_av = np.sum(self.c_mark[dis.astype(int).flatten()])/(4*np.size(self.c_mark[dis]))

        return chern_av

    def disk(self, r, x0, y0):
        x, y = np.copy(self.pos[np.argwhere(self.pos[:,1]%2==0), 0]), np.copy(self.pos[np.argwhere(self.pos[:,1]%2==0), 1])
        dx_a, dy_a = np.minimum(((x-x0)%(2*self.nx))**2,((x0-x)%(2*self.nx))**2), np.minimum(((y-y0)%(2*self.ny))**2,((y0-y)%(2*self.ny))**2)
        dx_b, dy_b = np.minimum(((x+1-x0)%(2*self.nx))**2,((x0-x-1)%(2*self.nx))**2), np.minimum(((y+1-y0)%(2*self.ny))**2,((y0-y-1)%(2*self.ny))**2)
        vecd_a = (np.sqrt(dx_a+dy_a)).flatten()
        vecd_b = (np.sqrt(dx_b+dy_b)).flatten()
        vecd = np.minimum(vecd_a,vecd_b)
        return np.argwhere(vecd<=r)

# Cell
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