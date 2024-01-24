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
![comp-poly128-A1_1](https://github.com/farceri/cudaRing/assets/32315176/5b6f6ba1-abb9-4721-9e98-04a757c45c90)

Confluent monolayer showing cage-breaking rearrangements
![rearrange-poly128-A1_1-phi8-v01e-03](https://github.com/farceri/cudaRing/assets/32315176/170d0245-6a3e-4cc6-95e6-830337fce742)
