// D2CFR-main/cpp_src/DeepMCCFR.cpp

#include "DeepMCCFR.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <torch/torch.h> // <--- ADD THIS

namespace ofc {

// MODIFIED: Constructor implementation
DeepMCCFR::DeepMCCFR(const std::string& model_path, size_t action_limit) 
    : action_limit_(action_limit), device_(torch::kCPU) { // Initialize device to CPU
    try {
        // Load the TorchScript model from the provided path
        model_ = torch::jit::load(model_path);
        // Set the model to evaluation mode
        model_.eval();
        // Move the model to the specified device (CPU)
        model_.to(device_);
        std::cout << "C++: LibTorch model loaded successfully from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        throw std::runtime_error("Failed to load LibTorch model: " + std::string(e.what()));
    }
}

std::vector<TrainingSample> DeepMCCFR::run_traversal() {
    // This function's implementation remains the same
    std::vector<TrainingSample> samples;
    GameState state_p0;
    traverse(state_p0, 0, samples);
    GameState state_p1;
    traverse(state_p1, 1, samples);
    return samples;
}

std::vector<float> DeepMCCFR::featurize(const GameState& state, int player_view) {
    // This function's implementation remains the same
    const Board& my_board = state.get_player_board(player_view);
    const Board& opp_board = state.get_opponent_board(player_view);
    const int FEATURE_SIZE = 1486;
    std::vector<float> features(FEATURE_SIZE, 0.0f);
    int offset = 0;
    features[offset++] = static_cast<float>(state.get_street());
    features[offset++] = static_cast<float>(state.get_dealer_pos());
    features[offset++] = static_cast<float>(state.get_current_player());
    const auto& dealt_cards = state.get_dealt_cards();
    for (Card c : dealt_cards) {
        if (c != INVALID_CARD) features[offset + c] = 1.0f;
    }
    offset += 52;
    auto process_board = [&](const Board& board, int& current_offset) {
        for(int i=0; i<3; ++i) {
            Card c = board.top[i];
            features[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
        current_offset += 3 * 53;
        for(int i=0; i<5; ++i) {
            Card c = board.middle[i];
            features[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
        current_offset += 5 * 53;
        for(int i=0; i<5; ++i) {
            Card c = board.bottom[i];
            features[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
        current_offset += 5 * 53;
    };
    process_board(my_board, offset);
    process_board(opp_board, offset);
    const auto& my_discards = state.get_my_discards(player_view);
    for (Card c : my_discards) {
        if (c != INVALID_CARD) features[offset + c] = 1.0f;
    }
    offset += 52;
    features[offset++] = static_cast<float>(state.get_opponent_discard_count(player_view));
    return features;
}

std::map<int, float> DeepMCCFR::traverse(GameState& state, int traversing_player, std::vector<TrainingSample>& samples) {
    if (state.is_terminal()) {
        auto payoffs = state.get_payoffs(evaluator_);
        return {{0, payoffs.first}, {1, payoffs.second}};
    }

    int current_player = state.get_current_player();
    auto legal_actions = state.get_legal_actions(action_limit_);
    int num_actions = legal_actions.size();

    if (num_actions == 0) {
        UndoInfo undo_info = state.apply_action({{}, INVALID_CARD}, traversing_player);
        auto result = traverse(state, traversing_player, samples);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    if (current_player != traversing_player) {
        int action_idx = state.get_rng()() % num_actions;
        UndoInfo undo_info = state.apply_action(legal_actions[action_idx], traversing_player);
        auto result = traverse(state, traversing_player, samples);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    std::vector<float> infoset_vec = featurize(state, traversing_player);
    
    // --- START: REPLACEMENT FOR REQUEST MANAGER ---
    std::vector<float> regrets;
    {
        // Ensure no gradient calculations are performed
        torch::NoGradGuard no_grad;

        // Convert the feature vector to a torch tensor without copying data
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input_tensor = torch::from_blob(infoset_vec.data(), {1, (long)infoset_vec.size()}, options).to(device_);

        // Create a vector of inputs for the model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // Run inference
        at::Tensor output_tensor = model_.forward(inputs).toTensor();

        // Copy the results back to a std::vector
        regrets.assign(output_tensor.data_ptr<float>(), output_tensor.data_ptr<float>() + num_actions);
    }
    // --- END: REPLACEMENT FOR REQUEST MANAGER ---

    std::vector<float> strategy(num_actions);
    float total_positive_regret = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        strategy[i] = (regrets[i] > 0) ? regrets[i] : 0.0f;
        total_positive_regret += strategy[i];
    }

    if (total_positive_regret > 0) {
        for (int i = 0; i < num_actions; ++i) strategy[i] /= total_positive_regret;
    } else {
        std::fill(strategy.begin(), strategy.end(), 1.0f / num_actions);
    }

    std::vector<std::map<int, float>> action_utils(num_actions);
    std::map<int, float> node_util = {{0, 0.0f}, {1, 0.0f}};

    for (int i = 0; i < num_actions; ++i) {
        UndoInfo undo_info = state.apply_action(legal_actions[i], traversing_player);
        action_utils[i] = traverse(state, traversing_player, samples);
        state.undo_action(undo_info, traversing_player);

        for(auto const& [player_idx, util] : action_utils[i]) {
            node_util[player_idx] += strategy[i] * util;
        }
    }

    std::vector<float> true_regrets(num_actions);
    for(int i = 0; i < num_actions; ++i) {
        true_regrets[i] = action_utils[i][current_player] - node_util[current_player];
    }
    
    samples.push_back({infoset_vec, true_regrets, num_actions});

    return node_util;
}

} // namespace ofc
