from os import path, sep
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, '..' + sep + 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

setup(
    name='spicy_vki',
    version='1.0.2',
    description="SPICY (Super-resolution and Pressure from Image veloCimetrY) is a software developed at the von Karman Institute to perform data assimilation of image velocimetry using constrained Radial Basis Functions (RBF). The framework works for structured data (as produced by cross-correlation-based algorithms in PIV or Optical FlowS) and unstructured data (produced by tracking algorithms in PTV).",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="P. Sperotto, M. Ratz, M. A. Mendez",
    author_email='manuel.ratz@vki.ac.be',
    url='https://github.com/mendezVKI/SPICY_VKI/tree/main/',
    packages=find_packages(exclude=['docs', 'tests']),
    entry_points={
        'console_scripts': [
            # 'command = some.module:some_function',
        ],
    },
    include_package_data=True,
    package_data={
        'spicy': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=[
        'numpy==1.21.5',
        'numpydoc==1.4.0',
        'scikit-learn==1.0.2',
        'scipy==1.10.1',
        'shapely==2.0.1',
        'matplotlib==3.5.2',
        "ipykernel==6.15.2",
        "ipython==7.31.1",
        "ipython-genutils==0.2.0",
        "ipywidgets==7.6.5",
    ],
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)