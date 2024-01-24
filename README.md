# cudaRing
Ring polymer simulator for molecular dynamics and energy minimization, written in CUDA and C++

Current version includes:

2D deformable particles modeled as interpenetrable ring polymers
Implemented geometries are Lees-Edwards, fixed and periodic boundary conditions in rectangular box
Molecular dynamics in the NVE and NVT ensembles
Interaction potentials: harmonic, Lennard-Jones potential, Weeks-Chandler-Anderson
Langevin dynamics with self-propulsion
Please contact me at arceri.fra at gmail.com for more info

Example of deformable particles in Langevin dynamics color-coded by size
![poly128-A1-thermal](https://github.com/farceri/cudaRing/assets/32315176/373f9931-eb4e-451f-ae61-1cd1bf42f099)
