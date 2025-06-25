# mccfr_ofc-main/setup.py

from setuptools import setup, Extension
import pybind11
import glob
import os
import sys

# Определяем аргументы компилятора
cpp_args = ['-std=c++17', '-O3', '-fopenmp']

# Для macOS может потребоваться другой флаг OpenMP и библиотеки
if sys.platform == 'darwin':
    cpp_args = ['-std=c++17', '-O3', '-Xpreprocessor', '-fopenmp', '-lomp']
    link_args = ['-lomp']
else: # Для Linux (Colab)
    link_args = ['-fopenmp']


# Находим все C++ исходники
ompeval_sources = glob.glob("cpp_src/ompeval/omp/*.cpp")
ofc_sources = [
    "cpp_src/game_state.cpp",
    "cpp_src/DeepMCCFR.cpp"
]

ext_modules = [
    Extension(
        'ofc_engine', # Имя модуля, которое будем импортировать в Python
        ['pybind_wrapper.cpp'] + ompeval_sources + ofc_sources,
        include_dirs=[
            pybind11.get_include(),
            "cpp_src",
            "cpp_src/ompeval"
        ],
        language='c++',
        extra_compile_args=cpp_args,
        extra_link_args=link_args
    ),
]

setup(
    name="ofc_engine",
    version="2.0.0",
    author="Azerus96 & AI Solver",
    description="Deep MCCFR solver for Pineapple OFC poker.",
    ext_modules=ext_modules,
    zip_safe=False,
)
