// D2CFR/pybind_wrapper.cpp (ИСПРАВЛЕННАЯ ВЕРСИЯ)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Pineapple Poker Engine with Deep MCCFR";

    // --- Биндинг для SharedReplayBuffer ---
    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        .def(py::init<size_t, size_t>(), py::arg("capacity"), py::arg("max_actions"))
        
        .def("sample", [](ofc::SharedReplayBuffer &self, size_t batch_size) {
            auto result_pair = self.sample(batch_size);
            
            if (result_pair.first.empty()) {
                std::vector<ssize_t> shape_infosets = {0, ofc::INFOSET_SIZE};
                std::vector<ssize_t> shape_targets = {0, (ssize_t)self.get_max_actions()};
                py::array_t<float> empty_infosets(shape_infosets);
                py::array_t<float> empty_targets(shape_targets);
                return std::make_pair(empty_infosets, empty_targets);
            }
            
            std::vector<ssize_t> shape_infosets = { static_cast<ssize_t>(batch_size), static_cast<ssize_t>(ofc::INFOSET_SIZE) };
            py::array_t<float> infosets_np(shape_infosets);
            float* infosets_ptr = static_cast<float*>(infosets_np.request().ptr);
            std::copy(result_pair.first.begin(), result_pair.first.end(), infosets_ptr);

            std::vector<ssize_t> shape_targets = { static_cast<ssize_t>(batch_size), static_cast<ssize_t>(self.get_max_actions()) };
            py::array_t<float> targets_np(shape_targets);
            float* targets_ptr = static_cast<float*>(targets_np.request().ptr);
            std::copy(result_pair.second.begin(), result_pair.second.end(), targets_ptr);

            return std::make_pair(infosets_np, targets_np);
        }, "Sample a batch from the buffer.") // Убрали call_guard, т.к. sample быстрый и вызывается из основного потока
        
        .def("get_count", &ofc::SharedReplayBuffer::get_count, "Get current size of the buffer.")
        .def("get_max_actions", &ofc::SharedReplayBuffer::get_max_actions, "Get max actions limit.");


    // --- Биндинг для DeepMCCFR ---
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<const std::string&, size_t, ofc::SharedReplayBuffer*>(), 
             py::arg("model_path"), py::arg("action_limit"), py::arg("buffer"))
        
        // ИЗМЕНЕНО: Убираем py::call_guard. Управление GIL теперь внутри C++ функции.
        .def("run_traversal_for_profiling", 
             &ofc::DeepMCCFR::run_traversal_for_profiling, 
             "Runs one full game traversal and returns profiling stats as a list of doubles.");
}
