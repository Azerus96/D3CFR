// D2CFR-main/pybind_wrapper.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> // <-- ДОБАВЛЕНО: для работы с NumPy
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp" // <-- ДОБАВЛЕНО

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with Shared Memory Replay Buffer";

    // Биндинг для SharedReplayBuffer
    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        .def(py::init<const std::string&, uint64_t>(), py::arg("shm_name"), py::arg("capacity"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count, "Returns the current number of items in the buffer.")
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) {
            // Создаем NumPy массивы, в которые C++ будет писать данные
            auto infosets_np = py::array_t<float>({batch_size, ofc::INFOSET_SIZE});
            auto regrets_np = py::array_t<float>({batch_size, ofc::ACTION_LIMIT});

            // Получаем сырые указатели на данные NumPy массивов
            float* infosets_ptr = static_cast<float*>(infosets_np.request().ptr);
            float* regrets_ptr = static_cast<float*>(regrets_np.request().ptr);

            // Вызываем C++ метод, который заполнит эти массивы
            buffer.sample(batch_size, infosets_ptr, regrets_ptr);

            // Возвращаем два NumPy массива в Python
            return std::make_pair(infosets_np, regrets_np);
        }, py::arg("batch_size"), "Samples a batch and returns (infosets, regrets) as numpy arrays.");

    // Биндинг для солвера
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        // ИЗМЕНЕНО: Конструктор теперь принимает SharedReplayBuffer
        .def(py::init([](const std::string& model_path, size_t action_limit, ofc::SharedReplayBuffer& buffer) {
            // Мы передаем сырой указатель, так как pybind11 гарантирует, что объект `buffer`
            // будет жив, пока жив объект `DeepMCCFR`, который его использует.
            return std::make_unique<ofc::DeepMCCFR>(model_path, action_limit, &buffer);
        }), py::arg("model_path"), py::arg("action_limit"), py::arg("buffer"))
        // GIL отпускается здесь, позволяя Python-потокам работать параллельно
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, "Runs one full game traversal.", py::call_guard<py::gil_scoped_release>());

    // Добавляем статическую функцию для очистки общей памяти
    m.def("cleanup_shared_memory", &ofc::SharedReplayBuffer::cleanup, 
          "Removes the shared memory segment.", py::arg("shm_name"));
}
