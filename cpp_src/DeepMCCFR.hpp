// D2CFR-main/cpp_src/DeepMCCFR.hpp

#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp" // <-- ИЗМЕНЕНО: подключаем новый буфер
#include <torch/script.h>
#include <vector>
#include <map>
#include <memory>

namespace ofc {

// Эта структура больше не нужна для передачи в Python, но оставим ее для ясности
struct TrainingSample {
    std::vector<float> infoset_vector;
    std::vector<float> target_regrets;
    int num_actions;
};

class DeepMCCFR {
public:
    // ИЗМЕНЕНО: Конструктор теперь принимает указатель на общий буфер
    DeepMCCFR(const std::string& model_path, size_t action_limit, SharedReplayBuffer* buffer);
    
    // ИЗМЕНЕНО: Метод теперь ничего не возвращает, он пишет напрямую в буфер
    void run_traversal();

private:
    HandEvaluator evaluator_;
    torch::jit::script::Module model_; 
    torch::Device device_;

    // ИЗМЕНЕНО: Храним указатель на общий буфер
    SharedReplayBuffer* replay_buffer_; 

    size_t action_limit_;

    // ИЗМЕНЕНО: traverse больше не нуждается в векторе для сбора сэмплов
    std::map<int, float> traverse(GameState& state, int traversing_player);
    std::vector<float> featurize(const GameState& state, int player_view);
};

} // namespace ofc
