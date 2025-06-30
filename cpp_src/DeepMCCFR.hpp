// D2CFR-main/cpp_src/DeepMCCFR.hpp (ВЕРСИЯ 5.0)

#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include <torch/script.h>
#include <vector>
#include <map>
#include <memory>

namespace ofc {

class DeepMCCFR {
public:
    DeepMCCFR(const std::string& model_path, size_t action_limit, SharedReplayBuffer* buffer);
    
    void run_traversal();

private:
    HandEvaluator evaluator_;
    torch::jit::script::Module model_; 
    torch::Device device_;
    SharedReplayBuffer* replay_buffer_; 
    size_t action_limit_;

    std::map<int, float> traverse(GameState& state, int traversing_player);
    std::vector<float> featurize(const GameState& state, int player_view);
};

}
