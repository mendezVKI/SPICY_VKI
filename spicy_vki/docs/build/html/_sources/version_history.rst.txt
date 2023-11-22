===============
Release History
===============
2023-11-11 - `spicy_vki == 1.0.2`
-----------------------------------
- Bugfix of the shape parameters. The shape parameters were computed such that the RBFs were half as big as intended. This is now fixed. The tutorials are updated, but old tutorials and other code still yield the same result when the minimum and maximum diameter are given twice as large.
- Fixed consistent notation in the computation of distances. All computations are now done with :math:`d = \sqrt{\boldsymbol{X} - \boldsymbol{X}_C^2}`


2023-09-20 - `spicy_vki == 1.0.1`
-----------------------------------
- Bugfix where a 3D scalar regression threw an error

2023-06-13 - `spicy_vki == 1.0.0`
-----------------------------------
- First release of the SPICY class along with four tutorials
- Supports 2D/3D scalar and vector regression problems and 2D/3D poisson problems
- RBF system assembled globally; RBFs can be Gaussians ('gauss') or compact support ('c4')