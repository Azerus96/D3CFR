// D2CFR-main/pybind_wrapper.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "cpp_src/DeepMCCFR.hpp"
// #include "cpp_src/request_manager.hpp" // REMOVE THIS LINE

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Pineapple DeepMCCFR Engine with C++ Inference";

    py::class_<ofc::TrainingSample>(m, "TrainingSample")
        .def(py::init<>())
        .def_readwrite("infoset_vector", &ofc::TrainingSample::infoset_vector)
        .def_readwrite("target_regrets", &ofc::TrainingSample::target_regrets)
        .def_readwrite("num_actions", &ofc::TrainingSample::num_actions);

    // REMOVED: All bindings for RequestManager, PredictionRequest, and PredictionResult are gone.

    // Биндинг для солвера
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        // MODIFIED: The constructor now takes a string (the model path)
        .def(py::init<const std::string&, size_t>())
        // This function is CPU-bound, so releasing the GIL is critical for parallelism
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, "Runs one full game traversal and returns training data.", py::call_guard<py::gil_scoped_release>());
}
