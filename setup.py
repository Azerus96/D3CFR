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
# Убедитесь, что Boost установлен в вашей системе.
# Например, для Ubuntu: sudo apt-get install libboost-all-dev
# Путь к библиотекам Boost может понадобиться, если они не в стандартных путях
boost_lib_path = os.getenv('BOOST_LIB_PATH', '/usr/lib/x86_64-linux-gnu') # Пример для Ubuntu

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
            # Может понадобиться, если Boost не в стандартных путях
            # os.getenv('BOOST_INCLUDE_PATH', '/usr/include') 
        ],
        language='c++',
        extra_compile_args=cpp_args,
        library_dirs=[
            os.path.join(libtorch_path, 'lib'),
            boost_lib_path 
        ],
        extra_link_args=link_args + [
            '-ltorch', 
            '-ltorch_cpu', 
            '-lc10',
            '-lboost_interprocess', # <-- ДОБАВЛЕНО: линкуем библиотеку Boost
            f'-Wl,-rpath,{os.path.join(libtorch_path, "lib")}'
        ]
    ),
]

setup(
    name="ofc_engine",
    version="4.0.0", # State-of-the-art!
    author="Azerus96 & AI Solver",
    description="SOTA Deep MCCFR solver for OFC poker with C++ inference and shared memory buffer.",
    ext_modules=ext_modules,
    zip_safe=False,
)
