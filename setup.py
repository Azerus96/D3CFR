from setuptools import setup, Extension
from torch.utils import cpp_extension
import glob
import os

# Находим все исходные файлы .cpp
sources = glob.glob('cpp_src/**/*.cpp', recursive=True)
sources.append('pybind_wrapper.cpp')

# Определяем пути для заголовочных файлов
include_dirs = [
    'cpp_src',
    'cpp_src/ompeval',
    'cpp_src/concurrentqueue'
]

# Определяем C++ расширение
ext_module = cpp_extension.CppExtension(
    name='ofc_engine',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args=['-O3', '-g', '-fopenmp'], # Флаги для компилятора
    extra_link_args=['-fopenmp'] # Флаги для линковщика
)

# Настраиваем сборку
setup(
    name='ofc_engine',
    version='1.0',
    ext_modules=[ext_module],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
