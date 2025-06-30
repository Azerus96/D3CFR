// D2CFR-main/pybind_wrapper.cpp (ВЕРСИЯ ДЛЯ ТЕСТА ИНИЦИАЛИЗАЦИИ)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with ThreadPoolExecutor and SharedReplayBuffer";

    // Биндинг для SharedReplayBuffer, он нам нужен
    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        .def(py::init<uint64_t, int>(), py::arg("capacity"), py::arg("action_limit"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count)
        .def("get_max_actions", &ofc::SharedReplayBuffer::get_max_actions)
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) {
            int action_limit = buffer.get_max_actions();
            
            // Создаем одномерные numpy массивы, которые потом переформируем в Python
            auto infosets_np = py::array_t<float>(batch_size * ofc::INFOSET_SIZE);
            auto regrets_np = py::array_t<float>(batch_size * action_limit);

            buffer.sample(
                batch_size, 
                static_cast<float*>(infosets_np.request().ptr), 
                static_cast<float*>(regrets_np.request().ptr)
            );
            
            // Переформировываем в 2D массивы
            infosets_np.resize({batch_size, ofc::INFOSET_SIZE});
            regrets_np.resize({batch_size, action_limit});

            return std::make_pair(infosets_np, regrets_np);
        }, py::arg("batch_size"));

    // Биндинг для DeepMCCFR, который принимает указатель на SharedReplayBuffer
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<const std::string&, size_t, ofc::SharedReplayBuffer*>(), 
             py::arg("model_path"), py::arg("action_limit"), py::arg("buffer"))
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, py::call_guard<py::gil_scoped_release>());
}
