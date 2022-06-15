import io, os, sys
from setuptools import find_packages, setup

# What packages are required for this module to be executed?
REQUIRED = ['matplotlib', 'numpy', 'networkx', 'cvxpy', 'pandas', 'scipy', 'sklearn', 'colorlog']
EXTRAS = {'GPU': ['cupy']}


setup(
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "experiments"]),

    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
)
