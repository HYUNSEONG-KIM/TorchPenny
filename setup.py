from setuptools import setup, find_packages

from torchpenny import __version__, __authors__
setup(
    name = 'torchpenny',
    version= __version__,
    description='Pennylane based Quantum circuit layer library for PyTorch.',
    author = __authors__,
    author_email= "Hyunseong.Kim@rice.edu",
    install_requires = ['torch', 'pennylane>=0.4.0'],
    packages= find_packages(include=["torchpenny", "torchpenny.*"])
)