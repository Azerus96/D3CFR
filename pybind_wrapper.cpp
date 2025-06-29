// D2CFR/pybind_wrapper.cpp (ФИНАЛЬНАЯ ВЕРСИЯ 4.0 - ЯВНОЕ СОЗДАНИЕ)

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
                // ИСПРАВЛЕНО: Явное создание пустых массивов через std::vector
                std::vector<ssize_t> shape_infosets = {0, ofc::INFOSET_SIZE};
                std::vector<ssize_t> shape_targets = {0, (ssize_t)self.get_max_actions()};
                py::array_t<float> empty_infosets(shape_infosets);
                py::array_t<float> empty_targets(shape_targets);
                return std::make_pair(empty_infosets, empty_targets);
            }
            
            // --- НАДЕЖНЫЙ СПОСОБ: КОПИРОВАНИЕ ДАННЫХ ---

            // 1. Создаем numpy массив для инфосетов нужного размера
            std::vector<ssize_t> shape_infosets = {
                static_cast<ssize_t>(batch_size), 
                static_cast<ssize_t>(ofc::INFOSET_SIZE)
            };
            py::array_t<float> infosets_np(shape_infosets);
            // Получаем указатель на его буфер
            float* infosets_ptr = static_cast<float*>(infosets_np.request().ptr);
            // Копируем данные из C++ вектора в numpy массив
            std::copy(result_pair.first.begin(), result_pair.first.end(), infosets_ptr);

            // 2. Создаем numpy массив для таргетов
            std::vector<ssize_t> shape_targets = {
                static_cast<ssize_t>(batch_size), 
                static_cast<ssize_t>(self.get_max_actions())
            };
            py::array_t<float> targets_np(shape_targets);
            float* targets_ptr = static_cast<float*>(targets_np.request().ptr);
            std::copy(result_pair.second.begin(), result_pair.second.end(), targets_ptr);

            // Возвращаем пару numpy массивов в Python. Теперь они владеют своими данными.
            return std::make_pair(infosets_np, targets_np);
        }, "Sample a batch from the buffer.", py::call_guard<py::gil_scoped_release>())
        
        .def("get_count", &ofc::SharedReplayBuffer::get_count, "Get current size of the buffer.")
        .def("get_max_actions", &ofc::SharedReplayBuffer::get_max_actions, "Get max actions limit.");


    // --- Биндинг для DeepMCCFR ---
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<const std::string&, size_t, ofc::SharedReplayBuffer*>(), 
             py::arg("model_path"), py::arg("action_limit"), py::arg("buffer"))
        
        .def("run_traversal_for_profiling", 
             &ofc::DeepMCCFR::run_traversal_for_profiling, 
             "Runs one full game traversal and returns profiling stats as a list of doubles.", 
             py::call_guard<py::gil_scoped_release>());
}
