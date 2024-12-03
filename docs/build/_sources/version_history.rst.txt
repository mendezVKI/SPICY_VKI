===============
Release History
===============

2024-02-12 - `spicy_vki == 1.1.0`
-----------------------------------
- Code extensively restructured. Many functions have been outsourced to a utils folder. They can still be imported as normal from there but keep the spicy class more free. The regression assembly also has been merged substantially to make it shorter.
- Added feature: Semi-random and regular collocation
- Added feature: Re-using factorizations. It is now possible to re-use the Cholesky factorization by coupling multiple scalar regressions. It is also possible to copy it from another process.
- Changed verbatim:
    * Introduced the 'model' parameter when initializing the class. This allows to split between laminar and scalar regressions more easily.
    * 'clustering' has been renamed to 'collocation'.
    * 'Get_Sol', 'Get_first_Derivatives' have been merged into 'get_sol' which computes derivatives from zeroth to second order.
    * The 'Areas' parameter has been moved to optional keywords. This ensures that multi-region refinement is not mandatory


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