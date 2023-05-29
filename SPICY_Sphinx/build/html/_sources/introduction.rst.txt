=================================
SPICY: Introduction
=================================

.. image:: spicy_logo.png
  :width: 700
  :align: center
  :alt: Alternative text
  
SPICY (Super-resolution and Pressure from Image veloCimetrY) is a software developed at the von Karman Institute to 
perform data assimilation by means of Radial Basis Functions (RBF). The framwork works both for structured and unstructered data. 
Currently, the main application is to perform a regression of image velocimetry data and then solve the pressure equation.
However, the framework can be readily extended to regression of other fields (e.g. temperature fields).

The theoretical foundation of the RBF framework is described in
- P. Sperotto, S. Pieraccini, M.A. Mendez, A Meshless Method to Compute Pressure Fields from
Image Velocimetry, Measurement Science and Technology 33(9), May 2022. The pre-print is 
available at https://arxiv.org/abs/2112.12752.

Currently, this folder contains four exercises. All of them are done in Python. The exercises
include regression of synthetic velocity fields as well as the solution of Poisson problems.

For each tutorial, there is a video on our YouTube channel: https://www.youtube.com/@spicyVKI 

The proposed exercises are the following:

1 - Solution of a Laplace problem on the unit square.

2 - Regression of the velocity field of a 2D Lamb-Oseen vortex.

3 - Regression of the velocity field and integration of the Poisson equation for the 2D flow past a cylinder.

4 - Regression of the velocity field and integration of the Poisson equation for the 3D Stokes flow past a sphere.

Exercises 2 - 4 are taken from the article from Sperotto et al. (2022) https://arxiv.org/abs/2112.12752