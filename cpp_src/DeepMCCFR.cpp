// D2CFR-main/cpp_src/DeepMCCFR.cpp (ВЕРСИЯ ДЛЯ ПРОФИЛИРОВАНИЯ)

#include "DeepMCCFR.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <torch/torch.h>
#include <chrono> // Для замеров времени

namespace ofc {

// Структура для сбора статистики по времени выполнения
struct ProfilingStats {
    std::chrono::duration<double, std::milli> total_traverse_time{0};
    std::chrono::duration<double, std::milli> get_legal_actions_time{0};
    std::chrono::duration<double, std::milli> featurize_time{0};
    std::chrono::duration<double, std::milli> model_inference_time{0};
    std::chrono::duration<double, std::milli> buffer_push_time{0};
    long call_count = 0;

    void print_and_reset() {
        if (call_count == 0) return;
        std::cout << "--- C++ Profiling Stats (Thread " << std::this_thread::get_id() << ") ---\n"
                  << "Avg traverse() total: " << total_traverse_time.count() / call_count << " ms\n"
                  << "  -> Avg get_legal_actions(): " << get_legal_actions_time.count() / call_count << " ms\n"
                  << "  -> Avg featurize(): " << featurize_time.count() / call_count << " ms\n"
                  << "  -> Avg model_inference(): " << model_inference_time.count() / call_count << " ms\n"
                  << "  -> Avg buffer->push(): " << buffer_push_time.count() / call_count << " ms\n"
                  << "--------------------------------------------------\n" << std::flush;
        
        // Сброс статистики
        total_traverse_time = std::chrono::duration<double, std::milli>(0);
        get_legal_actions_time = std::chrono::duration<double, std::milli>(0);
        featurize_time = std::chrono::duration<double, std::milli>(0);
        model_inference_time = std::chrono::duration<double, std::milli>(0);
        buffer_push_time = std::chrono::duration<double, std::milli>(0);
        call_count = 0;
    }
};

// Глобальная для потока статистика
thread_local ProfilingStats tls_stats;


DeepMCCFR::DeepMCCFR(const std::string& model_path, size_t action_limit, SharedReplayBuffer* buffer) 
    : action_limit_(action_limit), device_(torch::kCPU), replay_buffer_(buffer) {
    try {
        model_ = torch::jit::load(model_path);
        model_.eval();
        model_.to(device_);
        std::cout << "C++: LibTorch model loaded successfully from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        throw std::runtime_error("Failed to load LibTorch model: " + std::string(e.what()));
    }
}

void DeepMCCFR::run_traversal() {
    GameState state_p0;
    traverse(state_p0, 0);
    GameState state_p1;
    traverse(state_p1, 1);

    // Печатаем статистику после каждого полного прохода (2 траверса)
    tls_stats.print_and_reset();
}

std::vector<float> DeepMCCFR::featurize(const GameState& state, int player_view) {
    // ... (код featurize остается без изменений) ...
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
    auto t_start_total = std::chrono::high_resolution_clock::now();

    if (state.is_terminal()) {
        auto payoffs = state.get_payoffs(evaluator_);
        return {{0, payoffs.first}, {1, payoffs.second}};
    }

    int current_player = state.get_current_player();
    
    auto t_start_actions = std::chrono::high_resolution_clock::now();
    auto legal_actions = state.get_legal_actions(action_limit_);
    auto t_end_actions = std::chrono::high_resolution_clock::now();
    tls_stats.get_legal_actions_time += (t_end_actions - t_start_actions);

    int num_actions = legal_actions.size();

    if (num_actions == 0) {
        UndoInfo undo_info = state.apply_action({{}, INVALID_CARD}, traversing_player);
        auto result = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    if (current_player != traversing_player) {
        int action_idx = state.get_rng()() % num_actions;
        UndoInfo undo_info = state.apply_action(legal_actions[action_idx], traversing_player);
        auto result = traverse(state, traversing_player);
        state.undo_action(undo_info, traversing_player);
        return result;
    }

    auto t_start_featurize = std::chrono::high_resolution_clock::now();
    std::vector<float> infoset_vec = featurize(state, traversing_player);
    auto t_end_featurize = std::chrono::high_resolution_clock::now();
    tls_stats.featurize_time += (t_end_featurize - t_start_featurize);
    
    std::vector<float> regrets;
    {
        auto t_start_inference = std::chrono::high_resolution_clock::now();
        torch::NoGradGuard no_grad;
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input_tensor = torch::from_blob(infoset_vec.data(), {1, (long)infoset_vec.size()}, options).to(device_);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        at::Tensor output_tensor = model_.forward(inputs).toTensor();
        regrets.assign(output_tensor.data_ptr<float>(), output_tensor.data_ptr<float>() + num_actions);
        auto t_end_inference = std::chrono::high_resolution_clock::now();
        tls_stats.model_inference_time += (t_end_inference - t_start_inference);
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
        UndoInfo undo_info = state.apply_action(legal_actions[i], traversing_player);
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
    
    auto t_start_push = std::chrono::high_resolution_clock::now();
    replay_buffer_->push(infoset_vec, true_regrets, num_actions);
    auto t_end_push = std::chrono::high_resolution_clock::now();
    tls_stats.buffer_push_time += (t_end_push - t_start_push);

    auto t_end_total = std::chrono::high_resolution_clock::now();
    tls_stats.total_traverse_time += (t_end_total - t_start_total);
    tls_stats.call_count++;

    return node_util;
}

} // namespace ofc
