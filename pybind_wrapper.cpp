// D2CFR-main/pybind_wrapper.cpp (ВЕРСИЯ 7.1 - ИСПРАВЛЕННАЯ)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/InferenceQueue.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with Batch Inference";

    // Биндинг для очереди запросов
    py::class_<InferenceQueue>(m, "InferenceQueue")
        .def(py::init<>())
        .def("pop_all", &InferenceQueue::pop_all)
        .def("wait", &InferenceQueue::wait, py::call_guard<py::gil_scoped_release>()); // Освобождаем GIL во время ожидания

    // Биндинг для структуры запроса, чтобы Python мог ее читать
    py::class_<InferenceRequest>(m, "InferenceRequest")
        .def_readonly("infoset", &InferenceRequest::infoset)
        .def_readonly("num_actions", &InferenceRequest::num_actions)
        .def("set_result", [](InferenceRequest &req, std::vector<float> result) {
            req.promise.set_value(result);
        });

    // Биндинг для SharedReplayBuffer
    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        .def(py::init<uint64_t, int>(), py::arg("capacity"), py::arg("action_limit"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count)
        .def("get_max_actions", &ofc::SharedReplayBuffer::get_max_actions)
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) {
            int action_limit = buffer.get_max_actions();
            
            auto infosets_np = py::array_t<float>(batch_size * ofc::INFOSET_SIZE);
            auto regrets_np = py::array_t<float>(batch_size * action_limit);

            buffer.sample(
                batch_size, 
                static_cast<float*>(infosets_np.request().ptr), 
                static_cast<float*>(regrets_np.request().ptr)
            );
            
            infosets_np.resize({batch_size, ofc::INFOSET_SIZE});
            regrets_np.resize({batch_size, action_limit});

            return std::make_pair(infosets_np, regrets_np);
        }, py::arg("batch_size"));

    // Биндинг для DeepMCCFR
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        // ИСПРАВЛЕНО: Убрано ofc:: перед InferenceQueue*
        .def(py::init<size_t, ofc::SharedReplayBuffer*, InferenceQueue*>(), 
             py::arg("action_limit"), py::arg("buffer"), py::arg("queue"))
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, py::call_guard<py::gil_scoped_release>());
}
