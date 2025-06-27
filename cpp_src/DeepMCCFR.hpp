#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "request_manager.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>
#include <memory>

namespace py = pybind11;

namespace ofc {

struct TrainingSample {
    std::vector<float> infoset_vector;
    std::vector<float> target_regrets;
    int num_actions;
};

class DeepMCCFR {
public:
    DeepMCCFR(std::shared_ptr<RequestManager> manager, size_t action_limit);
    std::vector<TrainingSample> run_traversal();

private:
    HandEvaluator evaluator_;
    std::shared_ptr<RequestManager> request_manager_;
    size_t action_limit_;

    std::map<int, float> traverse(GameState state, int traversing_player, std::vector<TrainingSample>& samples);
    // ИЗМЕНЕНО: Сигнатура featurize теперь включает player_view
    std::vector<float> featurize(const GameState& state, int player_view);
};

} // namespace ofc
