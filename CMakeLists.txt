# D2CFR-main/CMakeLists.txt (ФИНАЛЬНАЯ ВЕРСИЯ 3.0)

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

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB_RECURSE SOURCES "cpp_src/*.cpp")
list(APPEND SOURCES "pybind_wrapper.cpp")

pybind11_add_module(ofc_engine SHARED ${SOURCES})

# --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ЗДЕСЬ ---
# Добавляем пути к нашим заголовочным файлам
target_include_directories(ofc_engine PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp_src"
    "${PROJECT_SOURCE_DIR}/cpp_src/ompeval"
)
# ---------------------------------

target_link_libraries(ofc_engine PRIVATE OpenMP::OpenMP_CXX ${TORCH_LIBRARIES})

target_compile_options(ofc_engine PRIVATE -O3 -g)

if(UNIX AND NOT APPLE)
    target_link_options(ofc_engine PRIVATE "-Wl,-rpath,${TORCH_INSTALL_PREFIX}/lib")
endif()

message(STATUS "Pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
