#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>

namespace py = pybind11;

namespace ofc {

struct TrainingSample {
    std::vector<float> infoset_vector;
    std::vector<float> target_regrets;
    int num_actions;
};

class DeepMCCFR {
public:
    DeepMCCFR(py::function predict_callback, size_t action_limit);
    std::vector<TrainingSample> run_traversal();

private:
    HandEvaluator evaluator_;
    py::function predict_callback_;
    size_t action_limit_;

    std::map<int, float> traverse(GameState state, int traversing_player, std::vector<TrainingSample>& samples);
    std::vector<float> featurize(const GameState& state);
};

} // namespace ofc
