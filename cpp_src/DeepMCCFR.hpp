// D2CFR-main/cpp_src/DeepMCCFR.hpp (ВЕРСИЯ 6.0 - MULTIPROCESSING)

#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include <torch/script.h>
#include <vector>
#include <map>
#include <memory>
#include <pybind11/pybind11.h>

namespace ofc {

class DeepMCCFR {
public:
    DeepMCCFR(const std::string& model_path, size_t action_limit, pybind11::object queue);
    
    void run_traversal();

private:
    void push_to_queue(const std::vector<float>& infoset, const std::vector<float>& regrets, int num_actions);

    HandEvaluator evaluator_;
    torch::jit::script::Module model_; 
    torch::Device device_;
    size_t action_limit_;
    
    pybind11::object queue_;

    std::map<int, float> traverse(GameState& state, int traversing_player);
    std::vector<float> featurize(const GameState& state, int player_view);
};

}
