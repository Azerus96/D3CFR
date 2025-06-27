// D2CFR-main/pybind_wrapper.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/request_manager.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Pineapple DeepMCCFR Engine with Batched Inference";

    py::class_<ofc::TrainingSample>(m, "TrainingSample")
        .def(py::init<>())
        .def_readwrite("infoset_vector", &ofc::TrainingSample::infoset_vector)
        .def_readwrite("target_regrets", &ofc::TrainingSample::target_regrets)
        .def_readwrite("num_actions", &ofc::TrainingSample::num_actions);

    // Биндинги для структур запросов и результатов
    py::class_<ofc::PredictionRequest>(m, "PredictionRequest")
        .def_readonly("id", &ofc::PredictionRequest::id)
        .def_readonly("infoset_vector", &ofc::PredictionRequest::infoset_vector)
        .def_readonly("num_actions", &ofc::PredictionRequest::num_actions);

    py::class_<ofc::PredictionResult>(m, "PredictionResult")
        .def(py::init<>())
        .def_readwrite("id", &ofc::PredictionResult::id)
        .def_readwrite("regrets", &ofc::PredictionResult::regrets);

    // Биндинг для менеджера запросов
    py::class_<ofc::RequestManager, std::shared_ptr<ofc::RequestManager>>(m, "RequestManager")
        .def(py::init<>())
        // ИСПРАВЛЕНО: Добавлен call_guard для освобождения GIL во время ожидания.
        // Это предотвращает deadlock.
        .def("get_requests", &ofc::RequestManager::get_requests, py::arg("max_batch_size"), py::call_guard<py::gil_scoped_release>())
        .def("post_results", &ofc::RequestManager::post_results);

    // Биндинг для солвера, который теперь принимает указатель на менеджер
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<std::shared_ptr<ofc::RequestManager>, size_t>())
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, "Runs one full game traversal and returns training data.");
}
