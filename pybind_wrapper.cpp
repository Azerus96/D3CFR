// D2CFR-main/pybind_wrapper.cpp (ФИНАЛЬНАЯ ВЕРСИЯ)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"

namespace py = pybind11;

// Обертка для конструктора, чтобы pybind мог работать с указателем
ofc::DeepMCCFR* create_deepmccfr(const std::string& model_path, size_t action_limit, ofc::SharedReplayBuffer& buffer) {
    return new ofc::DeepMCCFR(model_path, action_limit, &buffer);
}

// Обертка для sample, чтобы вернуть numpy массивы
py::object sample_wrapper(ofc::SharedReplayBuffer &buffer, int batch_size) {
    auto result_pair = buffer.sample(batch_size);
    if (result_pair.first.empty()) {
        return py::cast(std::make_pair(py::none(), py::none()));
    }
    
    // Создаем numpy массив для infosets
    py::array_t<float> infosets_np({batch_size, ofc::INFOSET_SIZE}, result_pair.first.data());

    // Создаем numpy массив для regrets
    py::array_t<float> regrets_np({batch_size, buffer.get_max_actions()}, result_pair.second.data());

    return py::cast(std::make_pair(infosets_np, regrets_np));
}


PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Pineapple Poker Engine with Deep MCCFR";

    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        .def(py::init<uint64_t, int>(), py::arg("capacity"), py::arg("action_limit"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count)
        .def("get_max_actions", &ofc::SharedReplayBuffer::get_max_actions)
        .def("sample", &sample_wrapper, py::arg("batch_size"));

    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init(&create_deepmccfr), 
             py::arg("model_path"), py::arg("action_limit"), py::arg("buffer"))
        .def("run_traversal_for_profiling", 
             &ofc::DeepMCCFR::run_traversal_for_profiling, 
             "Runs one full game traversal and returns profiling stats.");
}
