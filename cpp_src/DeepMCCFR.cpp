// D2CFR-main/cpp_src/DeepMCCFR.cpp (ФИНАЛЬНАЯ ВЕРСИЯ)

#include "DeepMCCFR.hpp"
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <torch/torch.h>
#include <chrono>

namespace py = pybind11;

namespace ofc {

DeepMCCFR::DeepMCCFR(const std::string& model_path, size_t action_limit, SharedReplayBuffer* buffer) 
    : action_limit_(action_limit), device_(torch::kCPU), replay_buffer_(buffer) {
    try {
        model_ = torch::jit::load(model_path);
        model_.eval();
        model_.to(device_);
    } catch (const c10::Error& e) {
        throw std::runtime_error("Failed to load LibTorch model: " + std::string(e.what()));
    }
    // Предварительно резервируем память для буферов
    legal_actions_buffer_.reserve(action_limit);
    infoset_vec_buffer_.reserve(1486);
    regrets_buffer_.reserve(action_limit);
    strategy_buffer_.reserve(action_limit);
    true_regrets_buffer_.reserve(action_limit);
    action_utils_buffer_.reserve(action_limit);
}

std::vector<double> DeepMCCFR::run_traversal_for_profiling() {
    std::vector<double> result;
    {
        py::gil_scoped_release release;
        
        ProfilingStats stats;
        GameState state; 
        
        traverse(state, 0, stats);
        state.reset(); 
        traverse(state, 1, stats);

        if (stats.call_count > 0) {
            result = {
                stats.total_traverse_time.count() / stats.call_count,
                stats.get_legal_actions_time.count() / stats.call_count,
                stats.featurize_time.count() / stats.call_count,
                stats.model_inference_time.count() / stats.call_count,
                stats.buffer_push_time.count() / stats.call_count
            };
        }
    } 
    return result;
}

std::vector<float> DeepMCCFR::featurize(const GameState& state, int player_view) {
    // Используем буфер-член класса
    infoset_vec_buffer_.assign(1486, 0.0f);
    int offset = 0;
    infoset_vec_buffer_[offset++] = static_cast<float>(state.get_street());
    infoset_vec_buffer_[offset++] = static_cast<float>(state.get_dealer_pos());
    infoset_vec_buffer_[offset++] = static_cast<float>(state.get_current_player());
    const auto& dealt_cards = state.get_dealt_cards();
    for (Card c : dealt_cards) {
        if (c != INVALID_CARD) infoset_vec_buffer_[offset + c] = 1.0f;
    }
    offset += 52;
    auto process_board = [&](const Board& board, int& current_offset) {
        for(int i=0; i<3; ++i) {
            Card c = board.top[i];
            infoset_vec_buffer_[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
        current_offset += 3 * 53;
        for(int i=0; i<5; ++i) {
            Card c = board.middle[i];
            infoset_vec_buffer_[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
        current_offset += 5 * 53;
        for(int i=0; i<5; ++i) {
            Card c = board.bottom[i];
            infoset_vec_buffer_[current_offset + i*53 + (c == INVALID_CARD ? 52 : c)] = 1.0f;
        }
        current_offset += 5 * 53;
    };
    const Board& my_board = state.get_player_board(player_view);
    const Board& opp_board = state.get_opponent_board(player_view);
    process_board(my_board, offset);
    process_board(opp_board, offset);
    const auto& my_discards = state.get_my_discards(player_view);
    for (Card c : my_discards) {
        if (c != INVALID_CARD) infoset_vec_buffer_[offset + c] = 1.0f;
    }
    offset += 52;
    infoset_vec_buffer_[offset++] = static_cast<float>(state.get_opponent_discard_count(player_view));
    return infoset_vec_buffer_;
}

std::map<int, float> DeepMCCFR::traverse(GameState& state, int traversing_player, ProfilingStats& stats) {
    auto t_start_total = std::chrono::high_resolution_clock::now();

    if (state.is_terminal()) {
        auto payoffs = state.get_payoffs(evaluator_);
        return {{0, payoffs.first}, {1, payoffs.second}};
    }

    int current_player = state.get_current_player();
    
    auto t_start_actions = std::chrono::high_resolution_clock::now();
    state.get_legal_actions(action_limit_, legal_actions_buffer_);
    auto t_end_actions = std::chrono::high_resolution_clock::now();
    stats.get_legal_actions_time += (t_end_actions - t_start_actions);

    int num_actions = legal_actions_buffer_.size();
    UndoInfo undo_info;

    if (num_actions == 0) {
        state.apply_action({{}, INVALID_CARD}, traversing_player, undo_info);
        auto result = traverse(state, traversing_player, stats);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    if (current_player != traversing_player) {
        int action_idx = state.get_rng()() % num_actions;
        state.apply_action(legal_actions_buffer_[action_idx], traversing_player, undo_info);
        auto result = traverse(state, traversing_player, stats);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    auto t_start_featurize = std::chrono::high_resolution_clock::now();
    featurize(state, traversing_player); // Заполняет infoset_vec_buffer_
    auto t_end_featurize = std::chrono::high_resolution_clock::now();
    stats.featurize_time += (t_end_featurize - t_start_featurize);
    
    regrets_buffer_.resize(num_actions);
    {
        auto t_start_inference = std::chrono::high_resolution_clock::now();
        torch::NoGradGuard no_grad;
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input_tensor = torch::from_blob(infoset_vec_buffer_.data(), {1, (long)infoset_vec_buffer_.size()}, options).to(device_);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        at::Tensor output_tensor = model_.forward(inputs).toTensor();
        std::copy(output_tensor.data_ptr<float>(), output_tensor.data_ptr<float>() + num_actions, regrets_buffer_.begin());
        auto t_end_inference = std::chrono::high_resolution_clock::now();
        stats.model_inference_time += (t_end_inference - t_start_inference);
    }

    strategy_buffer_.resize(num_actions);
    float total_positive_regret = 0.0f;
    for (int i = 0; i < num_actions; ++i) {
        strategy_buffer_[i] = (regrets_buffer_[i] > 0) ? regrets_buffer_[i] : 0.0f;
        total_positive_regret += strategy_buffer_[i];
    }

    if (total_positive_regret > 0) {
        for (int i = 0; i < num_actions; ++i) strategy_buffer_[i] /= total_positive_regret;
    } else {
        std::fill(strategy_buffer_.begin(), strategy_buffer_.end(), 1.0f / num_actions);
    }

    action_utils_buffer_.resize(num_actions);
    std::map<int, float> node_util = {{0, 0.0f}, {1, 0.0f}};

    for (int i = 0; i < num_actions; ++i) {
        state.apply_action(legal_actions_buffer_[i], traversing_player, undo_info);
        action_utils_buffer_[i] = traverse(state, traversing_player, stats);
        state.undo_action(undo_info, traversing_player);

        for(auto const& [player_idx, util] : action_utils_buffer_[i]) {
            node_util[player_idx] += strategy_buffer_[i] * util;
        }
    }

    true_regrets_buffer_.resize(num_actions);
    for(int i = 0; i < num_actions; ++i) {
        true_regrets_buffer_[i] = action_utils_buffer_[i][current_player] - node_util[current_player];
    }
    
    auto t_start_push = std::chrono::high_resolution_clock::now();
    replay_buffer_->push(infoset_vec_buffer_, true_regrets_buffer_, num_actions);
    auto t_end_push = std::chrono::high_resolution_clock::now();
    stats.buffer_push_time += (t_end_push - t_start_push);

    auto t_end_total = std::chrono::high_resolution_clock::now();
    stats.total_traverse_time += (t_end_total - t_start_total);
    stats.call_count++;

    return node_util;
}

} // namespace ofc
