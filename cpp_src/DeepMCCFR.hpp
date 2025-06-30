// D2CFR-main/cpp_src/DeepMCCFR.hpp (ВЕРСИЯ 7.0)

#pragma once
#include "game_state.hpp"
#include "hand_evaluator.hpp"
#include "SharedReplayBuffer.hpp"
#include "InferenceQueue.hpp" // <-- Подключаем новый файл
#include <vector>
#include <map>
#include <memory>

namespace ofc {

class DeepMCCFR {
public:
    // Конструктор теперь принимает указатели на общие ресурсы
    DeepMCCFR(size_t action_limit, SharedReplayBuffer* buffer, InferenceQueue* queue);
    
    void run_traversal();

private:
    HandEvaluator evaluator_;
    SharedReplayBuffer* replay_buffer_; 
    InferenceQueue* inference_queue_; // <-- Указатель на очередь инференса
    size_t action_limit_;

    std::map<int, float> traverse(GameState& state, int traversing_player);
    std::vector<float> featurize(const GameState& state, int player_view);
};

}
