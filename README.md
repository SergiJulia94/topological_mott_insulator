# Welcome to the Topological Mott Insulator toolbox
> A Hartree-Fock library to study an interaction-induced Chern Insulator in the checkerboard lattice. <br>


## Install



This project allows to install the library `topological_mott_insulator`.

To install locally the library, go to the folder containing the library and run `pip install -e topological_mott_insulator`.

The, to call the library, use `from topological_mott_insulator import ...`

## Publications
This code has been used in 
- [S. Julià-Farré, M. Müller, M. Lewenstein and A. Dauphin Phys. Rev. Lett. **125**,240601 (2020)](https://doi.org/10.1103/PhysRevLett.125.240601)
- [L. Cardarelli, S. Julià-Farré, M. Lewenstein, A. Dauphin and M. Müller, arXiv:xxx]().

## How to use

As an example, we perform a simple self-consistent algorithm of the unrestricted Hartree-Fock method to compare between site-nematic and quantum anomalous Hall phases.

```python
t0, jax, jay, jbx, jby = -1, 0.5, -0.5, -0.5, 0.5 # We fix the hopping parameters of the Hamiltonian
```

### Site-nematic phase

```python
#hide_output
nx, ny = 12, 12

v1, v2 = 4., 1.
v3, v4 = Rydberg_v3v4(v1,v2)
un_mf = checkerboard_lattice_un(nx=nx,ny=ny,t0=-1, jax=jax, jay=jay, 
		                        jbx=jbx, jby=jby, v1=v1, v2=v2, v3=v3, v4=v4,
		                        beta=1E+5, cell_filling=1, phix=0., phiy=0., cylinder=False, field=0.*1j, induce='nothing', border=False)
    
l_eta = [0.6,1.]

mb = master_bar(range(len(l_eta)),hide_graph=False)

for  eta in mb:
    for i1 in progress_bar(range(0,400),parent=mb):
        un_mf.iterate_mf(eta=eta)
    
```

We then plot the density and the energy spectrum (the color is the occupation of each state).


![png](docs/images/output_9_0.png)


### Quantum Anomalous Hall phase

```python
#hide_output
nx, ny = 12, 12
v1, v2 = 4., 2.5
v3, v4 = Rydberg_v3v4(v1,v2)


un_mf = checkerboard_lattice_un(nx=nx,ny=ny,t0=-1, jax=jax, jay=jay, 
		                        jbx=jbx, jby=jby, v1=v1, v2=v2, v3=v3, v4=v4,
		                        beta=1E+5, cell_filling=1, phix=0., phiy=0., cylinder=False, field=0.1*1j, induce='nothing', border=False)

l_eta = [0.6,1.]

mb = master_bar(range(len(l_eta)),hide_graph=False)

for  eta in mb:
    for i1 in progress_bar(range(0,50),parent=mb):
        un_mf.iterate_mf(eta=eta)
    
un_mf.field = 0.

for  eta in mb:
    for i1 in progress_bar(range(0,50),parent=mb):
        un_mf.iterate_mf(eta=eta)
```

We then plot the expetation value of the imaginary hopping and the energy spectrum. 


![png](docs/images/output_13_0.png)


### Self-trapped polaron

For a finite hole/particle doping, the unrestricted Hartree-Fock method gives rise to localized solutions due to the appearance of states inside the gap. We here consider the case of one extra particle.

```python
#hide_output
nx, ny = 12, 12
v1, v2 = 4., 2.5
v3, v4 = Rydberg_v3v4(v1,v2)

cf = (nx*ny+1)/(nx*ny)

un_mf = checkerboard_lattice_un(nx=nx,ny=ny,t0=-1, jax=jax, jay=jay, 
		                        jbx=jbx, jby=jby, v1=v1, v2=v2, v3=v3, v4=v4,
		                        beta=1E+5, cell_filling=cf, phix=0., phiy=0., cylinder=False, field=0.1*1j, induce='nothing', border=False)

l_eta = [0.6,1.]

mb = master_bar(range(len(l_eta)),hide_graph=False)

for  eta in mb:
    for i1 in progress_bar(range(0,50),parent=mb):
        un_mf.iterate_mf(eta=eta)
    
un_mf.field = 0.

for  eta in mb:
    for i1 in progress_bar(range(0,50),parent=mb):
        un_mf.iterate_mf(eta=eta)

```

We then plot the expetation value of the imaginary hopping and the energy spectrum. We observe the appearance of a self-trapped polaron both in the analysis of the imaginary hopping and of the energy spetrum.


![png](docs/images/output_18_0.png)


### Topological domains

When increasing the number of particles from half filling, the system eventually generates two domains with opposite spontaneous breaking of the time-reversal symmetry

```python
nx, ny = 24, 24
v1, v2 = 4., 2.5
v3, v4 = Rydberg_v3v4(v1,v2)

cf = (nx*ny+5)/(nx*ny)

un_mf = checkerboard_lattice_un(nx=nx,ny=ny,t0=-1, jax=jax, jay=jay, 
		                        jbx=jbx, jby=jby, v1=v1, v2=v2, v3=v3, v4=v4,
		                        beta=1E+5, cell_filling=cf, phix=0., phiy=0., cylinder=False, field=0.*1j, induce='nothing', border=False)

l_eta = [0.6,1.]

mb = master_bar(range(len(l_eta)),hide_graph=False)

for  eta in mb:
    for i1 in progress_bar(range(0,50),parent=mb):
        un_mf.iterate_mf(eta=eta)
    
un_mf.field = 0.

for  eta in mb:
    for i1 in progress_bar(range(0,50),parent=mb):
        un_mf.iterate_mf(eta=eta)
```










![png](docs/images/output_22_0.png)


### Effect of finite temperature 

For increasing temperature the gap of the quantum anomalous Hall phase closes, and eventually the phase disappears. We first consider the effect of a small temperature $T/t=0.1$, for which no important changes are observed compared to the zero temperature case shown in the first example.

```python
nx, ny = 12, 12
v1, v2 = 4., 2.5
v3, v4 = Rydberg_v3v4(v1,v2)


un_mf = checkerboard_lattice_un(nx=nx,ny=ny,t0=-1, jax=jax, jay=jay, 
		                        jbx=jbx, jby=jby, v1=v1, v2=v2, v3=v3, v4=v4,
		                        beta=10, cell_filling=1, phix=0., phiy=0., cylinder=False, field=0.1*1j, induce='nothing', border=False)

l_eta = [0.6,1]

mb = master_bar(range(len(l_eta)),hide_graph=False)

for  eta in mb:
    for i1 in progress_bar(range(0,50),parent=mb):
        un_mf.iterate_mf(eta=eta)
    
un_mf.field = 0.

for  eta in mb:
    for i1 in progress_bar(range(0,50),parent=mb):
        un_mf.iterate_mf(eta=eta)
```










![png](docs/images/output_26_0.png)


The situation is different for $T/t=0.2$ where we observe that the upper band begins to have a finite population.










![png](docs/images/output_28_2.png)


For $T/t=1$ the quantum anomalous Hall phase has disappeared: the system is not gapped at all and there is no spontaneous time-reversal symmetry breaking in the nearest-neighbors hoppings.










![png](docs/images/output_30_2.png)

