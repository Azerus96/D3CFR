// D2CFR-main/cpp_src/DeepMCCFR.hpp

#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
// #include "request_manager.hpp" // REMOVE THIS LINE
#include <torch/script.h> // <--- ADD THIS
#include <vector>
#include <map>
#include <memory>

namespace ofc {

struct TrainingSample {
    std::vector<float> infoset_vector;
    std::vector<float> target_regrets;
    int num_actions;
};

class DeepMCCFR {
public:
    // MODIFIED: Constructor now takes the model path instead of the manager
    DeepMCCFR(const std::string& model_path, size_t action_limit);
    std::vector<TrainingSample> run_traversal();

private:
    HandEvaluator evaluator_;
    // REMOVED: No longer need the request manager
    // std::shared_ptr<RequestManager> request_manager_; 
    
    // ADDED: The TorchScript model
    torch::jit::script::Module model_; 
    torch::Device device_; // ADDED: To specify CPU/GPU

    size_t action_limit_;

    std::map<int, float> traverse(GameState& state, int traversing_player, std::vector<TrainingSample>& samples);
    std::vector<float> featurize(const GameState& state, int player_view);
};

} // namespace ofc
