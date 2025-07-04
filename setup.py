from setuptools import setup, Extension
from torch.utils import cpp_extension
import glob
import os

print("--- Preparing sources for compilation ---")

# --- ИЗМЕНЕНИЕ: Явно указываем ТОЛЬКО нужные нам исходные файлы ---
# Мы больше не используем glob, чтобы случайно не захватить
# тесты и бенчмарки из сторонних библиотек.

sources = [
    "cpp_src/DeepMCCFR.cpp",
    "cpp_src/game_state.cpp",
    "cpp_src/ompeval/omp/CardRange.cpp",
    "cpp_src/ompeval/omp/CombinedRange.cpp",
    "cpp_src/ompeval/omp/EquityCalculator.cpp",
    "cpp_src/ompeval/omp/HandEvaluator.cpp",
    "pybind_wrapper.cpp"
]

# Проверяем, что все файлы существуют
for source in sources:
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source file not found: {source}")
    print(f"Found source: {source}")


# Определяем пути для заголовочных файлов
include_dirs = [
    'cpp_src',
    'cpp_src/ompeval',
    'cpp_src/concurrentqueue' # concurrentqueue - это header-only библиотека, ее нужно только включить
]
print(f"Include directories: {include_dirs}")

# Определяем C++ расширение
ext_module = cpp_extension.CppExtension(
    name='ofc_engine',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args=['-O3', '-g', '-fopenmp'], # Флаги для компилятора
    extra_link_args=['-fopenmp'] # Флаги для линковщика
)

# Настраиваем сборку
print("\n--- Configuring setup ---")
setup(
    name='ofc_engine',
    version='1.0',
    ext_modules=[ext_module],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
print("Setup configured successfully.")
