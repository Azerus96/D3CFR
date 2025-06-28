# D2CFR-main/CMakeLists.txt (ИСПРАВЛЕННАЯ ВЕРСИЯ 2.0)

cmake_minimum_required(VERSION 3.10)
project(OFCEngine)

# --- Настройки C++ ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Поиск зависимостей ---
# 1. Pybind11
# Мы больше не используем find_package. Вместо этого мы добавляем его как поддиректорию.
# pybind11 будет скачан автоматически, если его нет.
# Это самый надежный способ интеграции.
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.1 # Используем конкретную, стабильную версию
)
FetchContent_MakeAvailable(pybind11)

# 2. LibTorch (PyTorch C++ API)
find_package(Torch REQUIRED)

# 3. OpenMP для параллелизма
find_package(OpenMP REQUIRED)

# --- Собираем все исходники ---
file(GLOB_RECURSE SOURCES "cpp_src/*.cpp")
list(APPEND SOURCES "pybind_wrapper.cpp")

# --- Создание Python модуля ---
# ofc_engine - это имя модуля, которое будет импортироваться в Python
pybind11_add_module(ofc_engine SHARED ${SOURCES})

# --- Линковка с библиотеками ---
# 1. Подключаем OpenMP
target_link_libraries(ofc_engine PRIVATE OpenMP::OpenMP_CXX)

# 2. Подключаем LibTorch
target_link_libraries(ofc_engine PRIVATE ${TORCH_LIBRARIES})

# --- Настройки для корректной работы ---
target_compile_options(ofc_engine PRIVATE -O3 -g)

if(UNIX AND NOT APPLE)
    target_link_options(ofc_engine PRIVATE "-Wl,-rpath,${TORCH_INSTALL_PREFIX}/lib")
endif()

# Выводим пути для отладки
message(STATUS "Pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
