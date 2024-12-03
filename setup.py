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
VERSION = "1.1.2"

# Read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

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
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    install_requires=parse_requirements('requirements.txt'),
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
