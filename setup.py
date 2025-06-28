# D2CFR-main/setup.py

from setuptools import setup, Extension
import pybind11
import glob
import os
import sys

# --- LIBTORCH CONFIGURATION ---
# Set this environment variable to the path of your unzipped libtorch directory
# For example: export LIBTORCH_PATH=/path/to/libtorch
libtorch_path = os.getenv('LIBTORCH_PATH')
if not libtorch_path:
    raise EnvironmentError("LIBTORCH_PATH environment variable not set. Please set it to your LibTorch directory.")

# --- COMPILER ARGUMENTS ---
cpp_args = ['-std=c++17', '-O3', '-fopenmp']
link_args = ['-fopenmp']

# For macOS
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
            # ADDED: LibTorch include paths
            os.path.join(libtorch_path, 'include'),
            os.path.join(libtorch_path, 'include', 'torch', 'csrc', 'api', 'include'),
        ],
        language='c++',
        extra_compile_args=cpp_args,
        # MODIFIED: Linking arguments for LibTorch
        library_dirs=[os.path.join(libtorch_path, 'lib')],
        extra_link_args=link_args + [
            '-ltorch', 
            '-ltorch_cpu', 
            '-lc10',
            # This is important for the dynamic linker to find the .so files at runtime
            f'-Wl,-rpath,{os.path.join(libtorch_path, "lib")}'
        ]
    ),
]

setup(
    name="ofc_engine",
    version="3.0.0", # Version bump!
    author="Azerus96 & AI Solver",
    description="Deep MCCFR solver for Pineapple OFC poker with C++ inference.",
    ext_modules=ext_modules,
    zip_safe=False,
)
