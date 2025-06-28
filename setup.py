# D2CFR-main/setup.py

from setuptools import setup, Extension
import pybind11
import glob
import os
import sys

# --- LIBTORCH CONFIGURATION ---
libtorch_path = os.getenv('LIBTORCH_PATH')
if not libtorch_path:
    raise EnvironmentError("LIBTORCH_PATH environment variable not set. Please set it to your LibTorch directory.")

# --- BOOST CONFIGURATION ---
# УДАЛЕНО: Boost больше не нужен для этого модуля.

# --- COMPILER ARGUMENTS ---
cpp_args = ['-std=c++17', '-O3', '-fopenmp']
link_args = ['-fopenmp']

if sys.platform == 'darwin':
    cpp_args = ['-std=c++17', '-O3', '-Xpreprocessor', '-fopenmp']
    link_args = ['-lomp']

# --- SOURCE FILES ---
ompeval_sources = glob.glob("cpp_src/ompeval/omp/*.cpp")
ofc_sources = [
    "cpp_src/game_state.cpp",
    "cpp_src/DeepMCCFR.cpp"
]

ext_modules = [
    Extension(
        'ofc_engine',
        ['pybind_wrapper.cpp'] + ompeval_sources + ofc_sources,
        include_dirs=[
            pybind11.get_include(),
            "cpp_src",
            "cpp_src/ompeval",
            os.path.join(libtorch_path, 'include'),
            os.path.join(libtorch_path, 'include', 'torch', 'csrc', 'api', 'include'),
        ],
        language='c++',
        extra_compile_args=cpp_args,
        library_dirs=[
            os.path.join(libtorch_path, 'lib'),
        ],
        # ИЗМЕНЕНО: Удалена линковка с -lboost_interprocess
        extra_link_args=link_args + [
            '-ltorch', 
            '-ltorch_cpu', 
            '-lc10',
            f'-Wl,-rpath,{os.path.join(libtorch_path, "lib")}'
        ]
    ),
]

setup(
    name="ofc_engine",
    version="5.0.0", # Thread-safe version!
    author="Azerus96 & AI Solver",
    description="SOTA Deep MCCFR solver for OFC poker with C++ inference and thread-safe in-memory buffer.",
    ext_modules=ext_modules,
    zip_safe=False,
)
