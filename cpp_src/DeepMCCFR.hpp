// D2CFR-main/cpp_src/DeepMCCFR.hpp (ПОЛНАЯ ВЕРСИЯ ДЛЯ ЭТАПА 2)

#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include <torch/script.h>
#include <vector>
#include <map>
#include <memory>
#include <thread>
#include <chrono>

namespace ofc {

// Структура для сбора статистики остается той же
struct ProfilingStats {
    std::chrono::duration<double, std::milli> total_traverse_time{0};
    std::chrono::duration<double, std::milli> get_legal_actions_time{0};
    std::chrono::duration<double, std::milli> featurize_time{0};
    std::chrono::duration<double, std::milli> model_inference_time{0};
    std::chrono::duration<double, std::milli> buffer_push_time{0};
    long call_count = 0;
};


class DeepMCCFR {
public:
    DeepMCCFR(const std::string& model_path, size_t action_limit, SharedReplayBuffer* buffer);
    
    std::vector<double> run_traversal_for_profiling();

private:
    HandEvaluator evaluator_;
    torch::jit::script::Module model_; 
    torch::Device device_;
    SharedReplayBuffer* replay_buffer_; 
    size_t action_limit_;

    // УЛУЧШЕНО: traverse теперь принимает ProfilingStats по ссылке
    std::map<int, float> traverse(GameState& state, int traversing_player, ProfilingStats& stats);
    std::vector<float> featurize(const GameState& state, int player_view);
};

} // namespace ofc
