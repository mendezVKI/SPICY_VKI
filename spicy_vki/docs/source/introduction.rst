=================================
SPICY: Introduction
=================================

.. image:: spicy_logo.png
  :target: https://github.com/mendezVKI/SPICY_VKI/blob/main/spicy_vki/docs/source/spicy_logo.png
  :width: 700
  
SPICY (Super-resolution and Pressure from Image veloCimetrY) is a software developed at the von Karman Institute to perform data assimilation of image velocimetry using constrained Radial Basis Functions (RBF). 
The framework works for structured data (as produced by cross-correlation-based algorithms in PIV or Optical FlowS) and unstructured data (produced by tracking algorithms in PTV).

While the main scope is the assimilation of velocity fields, SPICY can also be used for the regression of other fields (e.g., temperature fields).
The theoretical foundation of the constrained RBF approach is described in - P. Sperotto, S. Pieraccini, M.A. Mendez, A Meshless Method to Compute Pressure Fields from Image Velocimetry, Measurement Science and Technology 33(9), May 2022. (pre-print at https://arxiv.org/abs/2112.12752).

The GitHub folder contains four exercises in Python. These include regression of synthetic velocity fields as well as the solution of Poisson problems.
For each tutorial, there is a video on our YouTube channel: https://www.youtube.com/@spicyVKI


The list of proposed exercises is following:

1 - Solution of a Laplace problem on the unit square.

2 - Regression of the velocity field of a 2D Lamb-Oseen vortex.

3 - Regression of the velocity field and integration of the Poisson equation for the 2D flow past a cylinder.

4 - Regression of the velocity field and integration of the Poisson equation for the 3D Stokes flow past a sphere.

Exercises 2 - 4 are taken from the article from Sperotto et al. (2022) https://arxiv.org/abs/2112.12752