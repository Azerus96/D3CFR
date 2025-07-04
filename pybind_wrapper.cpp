#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cpp_src/DeepMCCFR.hpp"
#include "cpp_src/SharedReplayBuffer.hpp"
#include "cpp_src/InferenceQueue.hpp"
#include "cpp_src/constants.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ofc_engine, m) {
    m.doc() = "OFC Engine with Batch Inference and High-Performance Queues";

    py::class_<InferenceQueue>(m, "InferenceQueue")
        .def(py::init<>())
        // --- ИЗМЕНЕНИЕ ---
        // Реализуем pop_all через неблокирующий try_dequeue_bulk.
        // Это позволяет Python-потоку забрать все доступные элементы из очереди
        // за один вызов, не блокируя C++ потоки.
        .def("pop_all", [](InferenceQueue &q) {
            // Оптимальный способ забрать все элементы из moodycamel::ConcurrentQueue
            std::vector<InferenceRequest> requests;
            // Выделяем память с запасом, чтобы избежать лишних реаллокаций.
            // size_approx() - быстрая, но не всегда точная оценка размера.
            requests.resize(q.size_approx()); 
            // Пытаемся забрать все элементы. Метод вернет реальное количество.
            size_t count = q.try_dequeue_bulk(requests.begin(), requests.size());
            // Обрезаем вектор до реального количества полученных элементов.
            requests.resize(count); 
            return requests;
        });
        // Метод wait() удален, так как безблокировочные очереди
        // не поддерживают блокирующего ожидания. Логика ожидания перенесена в Python.

    py::class_<InferenceRequest>(m, "InferenceRequest")
        .def_readonly("infoset", &InferenceRequest::infoset)
        .def_readonly("num_actions", &InferenceRequest::num_actions)
        .def("set_result", [](InferenceRequest &req, std::vector<float> result) {
            // Этот код остается без изменений, он работает с std::promise
            req.promise.set_value(result);
        });

    py::class_<ofc::SharedReplayBuffer>(m, "SharedReplayBuffer")
        .def(py::init<uint64_t, int>(), py::arg("capacity"), py::arg("action_limit"))
        .def("get_count", &ofc::SharedReplayBuffer::get_count)
        .def("get_head", &ofc::SharedReplayBuffer::get_head)
        .def("get_max_actions", &ofc::SharedReplayBuffer::get_max_actions)
        .def("sample", [](ofc::SharedReplayBuffer &buffer, int batch_size) {
            int action_limit = buffer.get_max_actions();
            auto infosets_np = py::array_t<float>(batch_size * ofc::INFOSET_SIZE);
            auto regrets_np = py::array_t<float>(batch_size * action_limit);
            buffer.sample(
                batch_size, 
                static_cast<float*>(infosets_np.request().ptr), 
                static_cast<float*>(regrets_np.request().ptr)
            );
            infosets_np.resize({batch_size, ofc::INFOSET_SIZE});
            regrets_np.resize({batch_size, action_limit});
            return std::make_pair(infosets_np, regrets_np);
        }, py::arg("batch_size"));

    py::class_<ofc::DeepMCCFR>(m, "DeepMCCFR")
        .def(py::init<size_t, ofc::SharedReplayBuffer*, InferenceQueue*>(), 
             py::arg("action_limit"), py::arg("buffer"), py::arg("queue"))
        .def("run_traversal", &ofc::DeepMCCFR::run_traversal, py::call_guard<py::gil_scoped_release>());
}
