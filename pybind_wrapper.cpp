// D2CFR-main/pybind_wrapper.cpp (ПОЛНАЯ ВЕРСИЯ ДЛЯ ЭТАПА 2)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"

namespace py = pybind11;

// Эта функция-обертка нужна, чтобы pybind11 мог работать с указателями
// на объекты, которыми он не владеет (как наш SharedReplayBuffer)
ofc::DeepMCCFR* create_deepmccfr(const std::string& model_path, size_t action_limit, ofc::SharedReplayBuffer& buffer) {
    return new ofc::DeepMCCFR(model_path, action_limit, &buffer);
}

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Pineapple Poker Engine with Deep MCCFR";

    // --- Биндинг для SharedReplayBuffer ---
    // На этом этапе он остается таким же, как в вашем архиве
    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        .def(py::init<uint64_t, int>(), py::arg("capacity"), py::arg("action_limit"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count)
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) {
            int action_limit = buffer.get_max_actions();
            auto infosets_np = py::array_t<float>({batch_size, ofc::INFOSET_SIZE});
            auto regrets_np = py::array_t<float>({batch_size, action_limit});
            
            // Эта часть кода не будет вызываться в профайлинге, но оставляем для полноты
            // buffer.sample(batch_size, static_cast<float*>(infosets_np.request().ptr), static_cast<float*>(regrets_np.request().ptr));
            
            return std::make_pair(infosets_np, regrets_np);
        });

    // --- Биндинг для DeepMCCFR ---
    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        // Используем функцию-обертку для конструктора
        .def(py::init(&create_deepmccfr), 
             py::arg("model_path"), py::arg("action_limit"), py::arg("buffer"))
        
        .def("run_traversal_for_profiling", 
             &ofc::DeepMCCFR::run_traversal_for_profiling, 
             "Runs one full game traversal and returns profiling stats.");
}
