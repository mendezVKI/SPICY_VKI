---
title: 'SPICY: a Python toolbox for meshless assimilation from image velocimetry using Radial Basis Functions'
tags:
  - Python
  - Radial Basis Functions
  - Poisson Equation
  - Super resolution
  - Data Assimilation in Image Velocimetry

authors:
  - name: Pietro Sperotto
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: M. Ratz
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: M. A. Mendez
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: the von Karman Institute for Fluid Dynamics (VKI), Rhode St. Genese, 1640, Belgium
   index: 1
 - #name: the Von Karman Institute for Fluid Dynamics (VKI), Rhode St. Genese, 1640, Belgium
   #index: 2

date: 3 May 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

This work presents our `SPICY' (meshlesS Pressure from Image veloCimetrY) toolbox for meshless data assimilation in image velocimetry. The proposed approach allows for computing an analytic representation of velocity and pressure fields from noisy and sparse fields, regardless of whether these are acquired using cross-correlation or particle tracking algorithms. SPICY uses penalized and constrained Radial Basis Functions (RBFs), allowing for enforcing physical priors (e.g., divergence-free in incompressible flows) or boundary conditions (e.g., no slip). The resulting analytic expression allows for super-resolution on arbitrary points and accurate computation of derivatives. These could be used to compute derived quantities (e.g., vorticity) and integrate the pressure Poisson equation to compute pressure fields in the case of incompressible flows.
A set of video tutorials on how to use SPICY is made available on [This is an external link to genome.gov](https://www.genome.gov/)

# Statement of need

- 1-page to explain the ROM problem (loosely) and give working-knowledge to user
- 2 rows to explain whom this package is intended to
- 5 rows to present the available material in the repo 

When dealing with complex, nonlinear, multi-scale phenomena it is needed to unveil the patterns hidden behind the ensemble viewpoint. This is a common issue shared with several applied sciences - put some examples. In Fluid Mechanics this is a fundamental problem, as many fluidic phenomena do possess several coherent structures in both time and space. The efforts of the community could be divided broadly in two categories: _energy-based_ and _frequency-based_. 
The following is intended to give a working knowledge to the reader, posing a common definition for the modal reduction problem, and overviews the main characteristics of the different techniques implemented in `MODULO`. 

Modal analysis decomposes a given dataset into a _linear_ combination of 'rank-1' portions, called _modes_ $\mathcal{M}$. Each mode can be intended as composed by a spatial ($\phi$) and temporal structure ($\psi$), which are weighted by some amplitude coefficient ($\sigma$). Thus, 

$$ D(x_i, t_k) \rightarrow D\big[i, k\big] = \sum_{r=i}^R \mathcal{M}_k \big[i, k\big] = \sum_{r=1}^R \sigma_r\phi_r\big[ i]\psi\big[k\big] $$

Or, equivalently, in a matrix form: 

$$ D = \Phi \Sigma \Psi^T $$

Being $\Psi$ the spatial matrix: 

$$ 
\mathbf{\Phi} = 
\begin{bmatrix}
\phi_1[i] & \phi_2[i] & ... & \phi_R[i] \\ 
\vdots & \vdots & \vdots
\end{bmatrix}
$$

$\Sigma$ the energy contribution matrix, 

$$
\Sigma = 
\begin{bmatrix}
\sigma_1 & 0 & ... & 0 \\ 
0 & \sigma_2 & ... & 0 \\ 
0 & 0 & \ddots & 0 \\
0 & 0 & ... & \sigma_{nt}
\end{bmatrix}
$$

and $\Psi$ the temporal basis matrix, 

$$
\Psi = 
\begin{bmatrix}
\psi_1[k] & \psi_2[k] & ... & \psi_R[k] \\ 
\vdots & \vdots & \vdots & \vdots
\end{bmatrix}
$$

Each technique differs in term of computation and, consequently, meaning of these modes but the equation above still holds.


1 - Reduced Order Modeling in a nutshell 
2 - Specify each technique (just matrices differences and physical interpretation)



`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).


# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References