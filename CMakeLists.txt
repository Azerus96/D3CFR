# D2CFR-main/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(OFCEngine)

# --- Настройки C++ ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Поиск зависимостей ---
# 1. Pybind11
find_package(pybind11 REQUIRED)
# 2. LibTorch (PyTorch C++ API)
find_package(Torch REQUIRED)
# 3. OpenMP для параллелизма
find_package(OpenMP REQUIRED)

# --- Собираем все исходники ---
# Собираем все .cpp файлы из cpp_src и его подпапок
file(GLOB_RECURSE SOURCES "cpp_src/*.cpp")
# Добавляем главный файл-обертку
list(APPEND SOURCES "pybind_wrapper.cpp")

# --- Создание Python модуля ---
# ofc_engine - это имя модуля, которое будет импортироваться в Python
pybind11_add_module(ofc_engine SHARED ${SOURCES})

# --- Линковка с библиотеками ---
# 1. Подключаем OpenMP
target_link_libraries(ofc_engine PRIVATE OpenMP::OpenMP_CXX)

# 2. Подключаем LibTorch
# ${TORCH_LIBRARIES} - это переменная, которую создает find_package(Torch)
# Она содержит все необходимые библиотеки (torch, torch_cpu, c10 и т.д.)
target_link_libraries(ofc_engine PRIVATE ${TORCH_LIBRARIES})

# --- Настройки для корректной работы ---
# Устанавливаем флаги компиляции
target_compile_options(ofc_engine PRIVATE -O3 -g)

# Устанавливаем RPATH, чтобы модуль мог найти .so файлы LibTorch во время выполнения
if(UNIX AND NOT APPLE)
    target_link_options(ofc_engine PRIVATE "-Wl,-rpath,${TORCH_INSTALL_PREFIX}/lib")
endif()

# --- Установка (опционально, но полезно) ---
# Эта часть нужна, если вы захотите делать `make install`
# Она скопирует скомпилированный .so файл в нужное место
install(TARGETS ofc_engine DESTINATION ${CMAKE_INSTALL_PREFIX})

# Выводим пути для отладки
message(STATUS "Pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
