// D2CFR/pybind_wrapper.cpp (ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)

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
        
        // Метод sample, который возвращает пару numpy-массивов
        .def("sample", [](ofc::SharedReplayBuffer &self, size_t batch_size) {
            // Вызываем C++ метод sample, который возвращает пару векторов
            auto result_pair = self.sample(batch_size);
            
            if (result_pair.first.empty()) {
                // Возвращаем пустые массивы, если сэмплирование не удалось
                return std::make_pair(py::array_t<float>({0, ofc::INFOSET_SIZE}), py::array_t<float>({0, (unsigned long)self.get_max_actions()}));
            }
            
            // Создаем numpy массив для инфосетов
            py::array_t<float> infosets_np(
                {static_cast<ssize_t>(batch_size), static_cast<ssize_t>(ofc::INFOSET_SIZE)},
                result_pair.first.data()
            );

            // Создаем numpy массив для таргетов
            size_t max_actions = self.get_max_actions();
            py::array_t<float> targets_np(
                {static_cast<ssize_t>(batch_size), static_cast<ssize_t>(max_actions)},
                result_pair.second.data()
            );

            // Возвращаем пару numpy массивов в Python
            return std::make_pair(infosets_np, targets_np);
        }, "Sample a batch from the buffer.", py::call_guard<py::gil_scoped_release>())
        
        .def("get_count", &ofc::SharedReplayBuffer::get_count, "Get current size of the buffer.")
        // ИСПРАВЛЕНО: Используем правильное имя функции
        .def("get_max_actions", &ofc::SharedReplayBuffer::get_max_actions, "Get max actions limit.");


    // --- Биндинг для DeepMCCFR ---
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<const std::string&, size_t, ofc::SharedReplayBuffer*>(), 
             py::arg("model_path"), py::arg("action_limit"), py::arg("buffer"))
        
        // ИСПРАВЛЕНО: Привязываем функцию для профилирования
        .def("run_traversal_for_profiling", 
             &ofc::DeepMCCFR::run_traversal_for_profiling, 
             "Runs one full game traversal and returns profiling stats as a list of doubles.", 
             py::call_guard<py::gil_scoped_release>());
}
