// D2CFR-main/pybind_wrapper.cpp (ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)

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
        // ИЗМЕНЕНИЕ 1: Конструктор теперь принимает capacity и action_limit, чтобы C++ знал размер буфера.
        .def(py::init<uint64_t, int>(), py::arg("capacity"), py::arg("action_limit"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count, "Returns the current number of items in the buffer.")
        // ИЗМЕНЕНИЕ 2: sample теперь должен знать action_limit для создания numpy-массива правильного размера.
        // Мы получаем его из самого объекта буфера.
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) {
            // Получаем action_limit из самого объекта буфера, чтобы гарантировать консистентность.
            int action_limit = buffer.get_action_limit(); 
            
            // Создаем numpy массивы правильного размера.
            auto infosets_np = py::array_t<float>({batch_size, ofc::INFOSET_SIZE});
            auto regrets_np = py::array_t<float>({batch_size, action_limit});

            // Получаем указатели на данные в numpy массивах.
            float* infosets_ptr = static_cast<float*>(infosets_np.request().ptr);
            float* regrets_ptr = static_cast<float*>(regrets_np.request().ptr);

            // Вызываем C++ функцию, которая заполнит эти массивы.
            buffer.sample(batch_size, infosets_ptr, regrets_ptr);

            return std::make_pair(infosets_np, regrets_np);
        }, py::arg("batch_size"), "Samples a batch and returns (infosets, regrets) as numpy arrays.");

    // Биндинг для солвера
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init([](const std::string& model_path, size_t action_limit, ofc::SharedReplayBuffer& buffer) {
            // Передаем указатель на существующий объект буфера в конструктор C++ класса.
            return std::make_unique<ofc::DeepMCCFR>(model_path, action_limit, &buffer);
        }), py::arg("model_path"), py::arg("action_limit"), py::arg("buffer"))
        // Освобождаем GIL (Global Interpreter Lock) во время выполнения C++ кода,
        // чтобы Python мог выполнять другие задачи, и чтобы C++ воркеры работали по-настоящему параллельно.
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, "Runs one full game traversal.", py::call_guard<py::gil_scoped_release>());
}
