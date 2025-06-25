// mccfr_ofc-main/cpp_src/DeepMCCFR.hpp

#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>

namespace py = pybind11;

namespace ofc {

// Структура для хранения одного тренировочного примера.
// Pybind11 автоматически преобразует ее в Python-объект.
struct TrainingSample {
    std::vector<float> infoset_vector;
    std::vector<float> target_regrets;
    int num_actions;
};

class DeepMCCFR {
public:
    // Конструктор принимает Python-функцию (нейросеть) как колбэк
    DeepMCCFR(py::function predict_callback);

    // Главный метод: запускает одну итерацию и возвращает данные для обучения
    std::vector<TrainingSample> run_traversal();

private:
    HandEvaluator evaluator_;
    py::function predict_callback_;

    // Рекурсивная функция траверса
    std::map<int, float> traverse(GameState state, int traversing_player, std::vector<TrainingSample>& samples);
    
    // Функция для преобразования GameState в вектор для нейросети
    std::vector<float> featurize(const GameState& state);
};

} // namespace ofc
