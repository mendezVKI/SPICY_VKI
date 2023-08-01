---
title: 'SPICY: a Python toolbox for meshless assimilation from image velocimetry using radial basis functions'
tags:
  - Python 
  - Radial Basis Functions
  - Super resolution in Image Velocimetry
  - Data Assimilation in Image Velocimetry
  - Poisson Equation


authors:
  - name: Pietro Sperotto
    orcid: 0000-0001-9412-0828
    #equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: M. Ratz
    #equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    orcid: 0009-0008-8491-8367
    affiliation: 1
  - name: M. A. Mendez
    orcid: 0000-0002-1115-2187
    #equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: The von Karman Institute for Fluid Dynamics (VKI), Rhode St. Genese, 1640, Belgium
   index: 1

date: 16 June 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

This work presents our `SPICY' (meshlesS Pressure from Image veloCimetrY) toolbox for meshless data assimilation in image velocimetry. The proposed approach allows for computing an analytic representation of velocity and pressure fields from noisy and sparse fields, regardless of whether these are acquired using cross-correlation or particle tracking algorithms. SPICY uses penalized and constrained Radial Basis Functions (RBFs), allowing for enforcing physical priors (e.g., divergence-free in incompressible flows) or boundary conditions (e.g., no slip). The resulting analytic expression allows for super-resolution on arbitrary points and accurate computation of derivatives. These could be used to compute derived quantities (e.g., vorticity) and to integrate the pressure Poisson equation to compute pressure fields in the case of incompressible flows.
A set of video tutorials on how to use SPICY is provided.

# Statement of need

Data assimilation methods are becoming increasingly crucial in image velocimetry, thanks to high spatial and temporal resolutions available with modern interrogation processing methods [@Sciacchitano]. Assimilation techniques optimally combine measurements and first principle models (e.g. conservation laws) to maximize noise removal, achieve measurement super-resolution, and compute related quantities such as vorticity and pressure fields. Several methods have been proposed in the literature and assessed in the framework of the European project HOMER (Holistic Optical Metrology for Aero-Elastic Research), grant agreement number 769237. The most classic approaches for the velocity assimilation involve regression of the particle trajectories (as in TrackFit by [@Gesemann2016]), while the computation of derived quantities is usually carried out by first mapping the tracks onto Eulerian grids and then solving the relevant PDEs using standard CFD approaches (as in [@Schneiders2016a, @Agarwal2021] ).  

Alternatives mesh-free methods are the second-generation Flowfit [@Gesemann2016], which combines the velocity regression and the pressure integration into a large nonlinear optimization problem, and methods based on physics-informed neural networks (PINNs) [@Rao2020,] which uses penalized artificial neural networks to solve for the velocity and the pressure fields. 
Recently, in [@Sperotto2022], we proposed a meshless approach based on constrained Radial Basis Functions (RBFs) to solve both the velocity regression problem and the pressure computation. This approach is akin to the well-known Kansa method [@Fornberg2015] for the meshless integration of PDEs. The main novelty is that this formulation yields linear regression problems that can be easily constrained (rather than penalized) and solved using standard linear system solvers. All the codes developed have now been released in the open SPICY (Super Resolution and Pressure from Image Velocimetry) toolbox linked to this contribution. Documentation, installation, and tutorials are available in the provided repository and on a [Youtube channel](https://youtu.be/VYeiip_mEtg). 


# Tutorials and ongoing works

A total of four tutorials have been published in the repository, allowing for reproducing the results in [@Sperotto2022]. The first tutorial presents the use of SPICY for solving the Laplace Equation in 2D, while tutorials two and three focus on the velocity regression and pressure computation on 2D velocity fields with or without the divergence-free constraints. Finally, tutorial four tackles a 3D case, namely the Stokes flow past a sphere. The solver currently implemented is a minor variant of the direct approach proposed in the original publication. Ongoing works are the extension to Reynolds average formulation to treat turbulent flows, as presented in [@Sperotto2022b] and the implementation of a Partition of Unity (PUM) approach to limit the memory usage, as in [@Ratz2022a]. 


# Acknowledgments
The development of SPICY has been carried out in the framework of the Research Master program of Pietro Sperotto (AY 2021/2022) and Manuel Ratz (AY 2022/2023) at the von Karman Institute. 

# References
