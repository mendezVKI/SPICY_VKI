from os import path, sep
from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()


NAME = 'spicy_vki'
DESCRIPTION = "SPICY (Super-resolution and Pressure from Image veloCimetrY) is a software developed at the von Karman Institute to perform data assimilation of image velocimetry using constrained Radial Basis Functions (RBF). The framework works for structured data (as produced by cross-correlation-based algorithms in PIV or Optical Flows) and unstructured data (produced by tracking algorithms in PTV)."
URL = 'https://github.com/mendezVKI/SPICY_VKI/tree/main/'
EMAIL = 'mendez@vki.ac.be'
AUTHOR = "P. Sperotto, M. Ratz, M. A. Mendez"
PYTHON_REQUIRES = '>=3.8.0'
VERSION = "1.0.5"

REQUIRED = [
    "numpy>=1.20",    
    "scikit-learn>=1.0",
    "ipython>=7.16.1",
    "scipy>=1.5",
    "shapely>=1.7.0", 
    "matplotlib>=3.3.0",
    "sphinx_rtd_theme>=1.3.0",
    "numpydoc>=1.6.0",
]


setup(
    name=NAME,
    python_requires=PYTHON_REQUIRES,
    version=VERSION,
    description=DESCRIPTION,
    long_description=readme,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    package_data={
        'spicy': []
    },
    install_requires=REQUIRED,
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)