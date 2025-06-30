// D2CFR-main/cpp_src/DeepMCCFR.cpp (ВЕРСИЯ ДЛЯ ИЗМЕРЕНИЯ ИНИЦИАЛИЗАЦИИ)

#include "DeepMCCFR.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <torch/torch.h>
#include <chrono>
#include <thread>
#include <sstream>

namespace ofc {

DeepMCCFR::DeepMCCFR(const std::string& model_path, size_t action_limit, SharedReplayBuffer* buffer) 
    : action_limit_(action_limit), device_(torch::kCPU), replay_buffer_(buffer) {
    
    // --- НАЧАЛО ЗАМЕРА ---
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        model_ = torch::jit::load(model_path);
        model_.eval();
        model_.to(device_);
    } catch (const c10::Error& e) {
        throw std::runtime_error("Failed to load LibTorch model: " + std::string(e.what()));
    }
    
    // --- КОНЕЦ ЗАМЕРА И ВЫВОД ---
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::stringstream ss;
    ss << "C++: Thread " << std::this_thread::get_id() 
       << " loaded model in " << duration << " ms." << std::endl;
    std::cout << ss.str();
}

void DeepMCCFR::run_traversal() {
    GameState state; 
    traverse(state, 0);
    state.reset(); 
    traverse(state, 1);
}

std::vector<float> DeepMCCFR::featurize(const GameState& state, int player_view) {
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

std::map<int, float> DeepMCCFR::traverse(GameState& state, int traversing_player) {
    if (state.is_terminal()) {
        auto payoffs = state.get_payoffs(evaluator_);
        return {{0, payoffs.first}, {1, payoffs.second}};
    }

    int current_player = state.get_current_player();
    
    std::vector<Action> legal_actions;
    state.get_legal_actions(action_limit_, legal_actions);
    
    int num_actions = legal_actions.size();
    UndoInfo undo_info;

    if (num_actions == 0) {
        state.apply_action({{}, INVALID_CARD}, traversing_player, undo_info);
        auto result = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    if (current_player != traversing_player) {
        int action_idx = state.get_rng()() % num_actions;
        state.apply_action(legal_actions[action_idx], traversing_player, undo_info);
        auto result = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    std::vector<float> infoset_vec = featurize(state, traversing_player);
    
    std::vector<float> regrets;
    {
        torch::NoGradGuard no_grad;
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input_tensor = torch::from_blob(infoset_vec.data(), {1, (long)infoset_vec.size()}, options).to(device_);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        at::Tensor output_tensor = model_.forward(inputs).toTensor();
        regrets.assign(output_tensor.data_ptr<float>(), output_tensor.data_ptr<float>() + num_actions);
    }

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
        state.apply_action(legal_actions[i], traversing_player, undo_info);
        action_utils[i] = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);

        for(auto const& [player_idx, util] : action_utils[i]) {
            node_util[player_idx] += strategy[i] * util;
        }
    }

    std::vector<float> true_regrets(num_actions);
    for(int i = 0; i < num_actions; ++i) {
        true_regrets[i] = action_utils[i][current_player] - node_util[current_player];
    }
    
    replay_buffer_->push(infoset_vec, true_regrets, num_actions);

    return node_util;
}

}
