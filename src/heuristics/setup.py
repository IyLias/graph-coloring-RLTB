from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'gcp_solver',
        ['py_bindings.cpp', 'gcp_solver.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True)
        ],
        language='c++'
    ),
]

setup(
    name='gcp_solver',
    version='0.1',
    author='IyLias',
    description='Graph Coloring Problem Solver using TabuCol',
    ext_modules=ext_modules,
    install_requires=['pybind11'],
)
