#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "cpp_src/DeepMCCFR.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Pineapple DeepMCCFR Engine";

    py::class_<ofc::TrainingSample>(m, "TrainingSample")
        .def(py::init<>())
        .def_readwrite("infoset_vector", &ofc::TrainingSample::infoset_vector)
        .def_readwrite("target_regrets", &ofc::TrainingSample::target_regrets)
        .def_readwrite("num_actions", &ofc::TrainingSample::num_actions);

    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<py::function, size_t>())
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, "Runs one full game traversal and returns training data.");
}
