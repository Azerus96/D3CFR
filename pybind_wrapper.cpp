// D2CFR-main/pybind_wrapper.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with Thread-Safe In-Memory Replay Buffer";

    // Биндинг для SharedReplayBuffer
    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        // ИЗМЕНЕНО: Конструктор теперь принимает только capacity
        .def(py::init<uint64_t>(), py::arg("capacity"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count, "Returns the current number of items in the buffer.")
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) {
            auto infosets_np = py::array_t<float>({batch_size, ofc::INFOSET_SIZE});
            auto regrets_np = py::array_t<float>({batch_size, ofc::ACTION_LIMIT});

            float* infosets_ptr = static_cast<float*>(infosets_np.request().ptr);
            float* regrets_ptr = static_cast<float*>(regrets_np.request().ptr);

            buffer.sample(batch_size, infosets_ptr, regrets_ptr);

            return std::make_pair(infosets_np, regrets_np);
        }, py::arg("batch_size"), "Samples a batch and returns (infosets, regrets) as numpy arrays.");

    // Биндинг для солвера
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init([](const std::string& model_path, size_t action_limit, ofc::SharedReplayBuffer& buffer) {
            return std::make_unique<ofc::DeepMCCFR>(model_path, action_limit, &buffer);
        }), py::arg("model_path"), py::arg("action_limit"), py::arg("buffer"))
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, "Runs one full game traversal.", py::call_guard<py::gil_scoped_release>());

    // УДАЛЕНО: Функция cleanup_shared_memory больше не нужна.
}
