// D2CFR/pybind_wrapper.cpp (ПОЛНАЯ ВЕРСИЯ ДЛЯ ПРОФИЛИРОВАНИЯ)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"

namespace py = pybind11;

// Предполагаем, что размер инфосета - константа. Если нет, его нужно передавать.
const int INFOSET_SIZE = 1486; 

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Pineapple Poker Engine with Deep MCCFR";

    // --- Биндинг для SharedReplayBuffer ---
    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        .def(py::init<size_t, size_t>(), py::arg("capacity"), py::arg("max_actions"))
        
        // Метод sample, который возвращает пару numpy-массивов
        .def("sample", [](ofc::SharedReplayBuffer &self, size_t batch_size) {
            // Вызываем C++ метод sample, который возвращает пару векторов
            auto result_pair = self.sample(batch_size);
            
            // Создаем numpy массив для инфосетов
            // Важно: мы не можем просто вернуть указатель, так как векторы временные.
            // Мы должны скопировать данные.
            py::array_t<float> infosets_np(
                {static_cast<ssize_t>(batch_size), static_cast<ssize_t>(INFOSET_SIZE)}
            );
            // Получаем прямой доступ к буферу numpy массива
            float* infosets_ptr = static_cast<float*>(infosets_np.request().ptr);
            // Копируем данные из вектора в numpy массив
            std::copy(result_pair.first.begin(), result_pair.first.end(), infosets_ptr);

            // Создаем numpy массив для таргетов
            size_t max_actions = self.get_max_actions();
            py::array_t<float> targets_np(
                {static_cast<ssize_t>(batch_size), static_cast<ssize_t>(max_actions)}
            );
            float* targets_ptr = static_cast<float*>(targets_np.request().ptr);
            std::copy(result_pair.second.begin(), result_pair.second.end(), targets_ptr);

            // Возвращаем пару numpy массивов в Python
            return std::make_pair(infosets_np, targets_np);
        }, "Sample a batch from the buffer.", py::call_guard<py::gil_scoped_release>())
        
        .def("get_count", &ofc::SharedReplayBuffer::get_count, "Get current size of the buffer.")
        .def("get_max_actions", &ofc::SharedReplayBuffer::get_max_actions, "Get max actions limit.");


    // --- Биндинг для DeepMCCFR ---
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<const std::string&, size_t, ofc::SharedReplayBuffer*>(), 
             py::arg("model_path"), py::arg("action_limit"), py::arg("buffer"))
        
        // --- ГЛАВНОЕ ИСПРАВЛЕНИЕ ---
        // Мы привязываем новую функцию run_traversal_for_profiling, которую создали в C++.
        // Старой функции run_traversal больше не существует.
        .def("run_traversal_for_profiling", 
             &ofc::DeepMCCFR::run_traversal_for_profiling, 
             "Runs one full game traversal and returns profiling stats as a list of doubles.", 
             py::call_guard<py::gil_scoped_release>());
}
