
# Welcome to the Topological Mott Insulator toolbox
> A Hartree-Fock library to study an interaction-induced Chern Insulator in the checkerboard lattice.


(under development)

## Install

`to be released`

## How to use

Perform a simple self-consistent algorithm of the unrestricted Hartree-Fock method to compare between site-nematic and quantum anomalous Hall phases

### Site-nematic phase

```python
nx, ny = 12, 12
t0 = -1
jax, jay, jbx, jby = 0.5, -0.5, -0.5, 0.5
v1, v2 = 4., 1.
tau = np.copy(v1/v2)
alpha = ((tau-1)/(8-tau))**(1/6)
kappa = v1*(1+alpha**6)

v3 = kappa*(1/(1+(2*alpha)**6))
v4 = kappa*(1/(1+(np.sqrt(5)*alpha)**6))

cf = (nx*ny)/(nx*ny)
phix, phiy = 0., 0.
beta = 1E+5

un_mf = checkerboard_lattice_un(nx=nx,ny=ny,t0=-1, jax=jax, jay=jay, 
		                        jbx=jbx, jby=jby, v1=v1, v2=v2, v3=v3, v4=v4,
		                        beta=beta, cell_filling=cf, phix=phix, phiy=phiy, cylinder=False, field=0.*1j, induce='nothing', border=False)

for i1 in (range(0,50)):
    un_mf.iterate_mf(eta=0.6)

for i1 in (range(0,50)):
    un_mf.iterate_mf(eta=1.)
    
```

### Quantum Anomalous Hall phase

```python
nx, ny = 12, 12
t0 = -1
jax, jay, jbx, jby = 0.5, -0.5, -0.5, 0.5
v1, v2 = 4., 2.5
tau = np.copy(v1/v2)
alpha = ((tau-1)/(8-tau))**(1/6)
kappa = v1*(1+alpha**6)

v3 = kappa*(1/(1+(2*alpha)**6))
v4 = kappa*(1/(1+(np.sqrt(5)*alpha)**6))

cf = (nx*ny)/(nx*ny)
phix, phiy = 0., 0.
beta = 1E+5

un_mf = checkerboard_lattice_un(nx=nx,ny=ny,t0=-1, jax=jax, jay=jay, 
		                        jbx=jbx, jby=jby, v1=v1, v2=v2, v3=v3, v4=v4,
		                        beta=beta, cell_filling=cf, phix=phix, phiy=phiy, cylinder=False, field=0.1*1j, induce='nothing', border=False)

for i1 in range(0,50):
    un_mf.iterate_mf(eta=0.6)

for i1 in range(0,50):
    un_mf.iterate_mf(eta=1.)
    
un_mf.field = 0.

for i1 in (range(0,50)):
    un_mf.iterate_mf(eta=0.6)

for i1 in (range(0,50)):
    un_mf.iterate_mf(eta=1.)
```

### Self-trapped polaron

For a finite hole/particle doping, the unrestricted Hartree-Fock method gives rise to localized solutions due to the appearance of states inside the gap.

```python
nx, ny = 12, 12
t0 = -1
jax, jay, jbx, jby = 0.5, -0.5, -0.5, 0.5
v1, v2 = 4., 2.5
tau = np.copy(v1/v2)
alpha = ((tau-1)/(8-tau))**(1/6)
kappa = v1*(1+alpha**6)

v3 = kappa*(1/(1+(2*alpha)**6))
v4 = kappa*(1/(1+(np.sqrt(5)*alpha)**6))

cf = (nx*ny+1)/(nx*ny)
phix, phiy = 0., 0.
beta = 1E+5

un_mf = checkerboard_lattice_un(nx=nx,ny=ny,t0=-1, jax=jax, jay=jay, 
		                        jbx=jbx, jby=jby, v1=v1, v2=v2, v3=v3, v4=v4,
		                        beta=beta, cell_filling=cf, phix=phix, phiy=phiy, cylinder=False, field=0.1*1j, induce='nothing', border=False)

for i1 in (range(0,50)):
    un_mf.iterate_mf(eta=0.6)

for i1 in (range(0,50)):
    un_mf.iterate_mf(eta=1.)
    
un_mf.field = 0.

for i1 in (range(0,50)):
    un_mf.iterate_mf(eta=0.6)

for i1 in (range(0,50)):
    un_mf.iterate_mf(eta=1.)
```

### Topological domains

When increasing the number of particles from half filling, the system eventually generates two domains with opposite spontaneous breaking of the time-reversal symmetry

```python
nx, ny = 24, 24
t0 = -1
jax, jay, jbx, jby = 0.5, -0.5, -0.5, 0.5
v1, v2 = 4., 2.5
tau = np.copy(v1/v2)
alpha = ((tau-1)/(8-tau))**(1/6)
kappa = v1*(1+alpha**6)

v3 = kappa*(1/(1+(2*alpha)**6))
v4 = kappa*(1/(1+(np.sqrt(5)*alpha)**6))

cf = (nx*ny+5)/(nx*ny)
phix, phiy = 0., 0.
beta = 1E+5

un_mf = checkerboard_lattice_un(nx=nx,ny=ny,t0=-1, jax=jax, jay=jay, 
		                        jbx=jbx, jby=jby, v1=v1, v2=v2, v3=v3, v4=v4,
		                        beta=beta, cell_filling=cf, phix=phix, phiy=phiy, cylinder=False, field=0.*1j, induce='nothing', border=False)

for i1 in (range(0,2)):
    un_mf.iterate_mf(eta=0.6)

for i1 in (range(0,2)):
    un_mf.iterate_mf(eta=1.)
```
