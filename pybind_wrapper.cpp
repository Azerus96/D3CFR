// D2CFR-main/pybind_wrapper.cpp (ВЕРСИЯ 6.0 - MULTIPROCESSING)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cpp_src/DeepMCCFR.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine for Multiprocessing";

    // SharedReplayBuffer больше не нужен в биндингах

    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<const std::string&, size_t, py::object>(), 
             py::arg("model_path"), py::arg("action_limit"), py::arg("queue"))
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, py::call_guard<py::gil_scoped_release>());
}
