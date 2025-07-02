# D2CFR-main/CMakeLists.txt (ИСПРАВЛЕННАЯ ВЕРСИЯ)

cmake_minimum_required(VERSION 3.10)
project(OFCEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.1
)
FetchContent_MakeAvailable(pybind11)

# Эта команда найдет PyTorch, который был установлен через pip
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# Явно указываем исходные файлы для компиляции, чтобы избежать сборки
# тестов и бенчмарков из зависимостей (например, concurrentqueue).
set(SOURCES
    "cpp_src/DeepMCCFR.cpp"
    "cpp_src/game_state.cpp"
    "pybind_wrapper.cpp"
)

pybind11_add_module(ofc_engine SHARED ${SOURCES})

# Добавляем пути к заголовочным файлам, включая header-only библиотеку concurrentqueue
target_include_directories(ofc_engine PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp_src"
    "${PROJECT_SOURCE_DIR}/cpp_src/ompeval"
    "${PROJECT_SOURCE_DIR}/cpp_src/concurrentqueue"  # <-- ДОБАВЛЕНО
)

# Линкуемся с библиотеками, которые нашел find_package(Torch)
target_link_libraries(ofc_engine PRIVATE OpenMP::OpenMP_CXX ${TORCH_LIBRARIES})

# Опции компиляции
target_compile_options(ofc_engine PRIVATE -O3 -g)

# Выводим отладочную информацию
message(STATUS "Pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
