// D2CFR/pybind_wrapper.cpp (ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ 2.0)

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
                // ИСПРАВЛЕНО: Правильный способ создания пустых numpy массивов
                py::array_t<float> empty_infosets({0, (long)ofc::INFOSET_SIZE});
                py::array_t<float> empty_targets({0, (long)self.get_max_actions()});
                return std::make_pair(empty_infosets, empty_targets);
            }
            
            // Создаем numpy массив для инфосетов.
            // pybind11 достаточно умен, чтобы управлять памятью вектора, пока numpy массив жив.
            // Это zero-copy операция.
            py::capsule free_when_done(result_pair.first.data(), [](void *f) {
                // Этот лямбда-захват гарантирует, что векторы не будут уничтожены, пока капсула жива
                // Но так как мы возвращаем пару, векторы будут жить до конца выражения.
                // Для безопасности можно было бы аллоцировать их в куче, но здесь это избыточно.
            });
            py::array_t<float> infosets_np(
                {static_cast<ssize_t>(batch_size), static_cast<ssize_t>(ofc::INFOSET_SIZE)},
                result_pair.first.data(),
                free_when_done
            );

            // Создаем numpy массив для таргетов
            py::capsule free_when_done2(result_pair.second.data(), [](void *f) {});
            py::array_t<float> targets_np(
                {static_cast<ssize_t>(batch_size), static_cast<ssize_t>(self.get_max_actions())},
                result_pair.second.data(),
                free_when_done2
            );

            // Возвращаем пару numpy массивов в Python
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
