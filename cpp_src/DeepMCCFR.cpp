#include "DeepMCCFR.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace ofc {

DeepMCCFR::DeepMCCFR(std::shared_ptr<RequestManager> manager, size_t action_limit) 
    : request_manager_(manager), action_limit_(action_limit) {
    if (!request_manager_) {
        throw std::runtime_error("RequestManager is null.");
    }
}

std::vector<TrainingSample> DeepMCCFR::run_traversal() {
    std::vector<TrainingSample> samples;
    GameState initial_state;
    
    traverse(initial_state, 0, samples);
    traverse(initial_state, 1, samples);
    
    return samples;
}

std::vector<float> DeepMCCFR::featurize(const GameState& state) {
    // ... (код featurize остается без изменений) ...
}

std::map<int, float> DeepMCCFR::traverse(GameState state, int traversing_player, std::vector<TrainingSample>& samples) {
    if (state.is_terminal()) {
        auto payoffs = state.get_payoffs(evaluator_);
        return {{0, payoffs.first}, {1, payoffs.second}};
    }

    int current_player = state.get_current_player();
    
    auto legal_actions = state.get_legal_actions(action_limit_);
    
    int num_actions = legal_actions.size();
    if (num_actions == 0) {
        return traverse(state.apply_action({{}, INVALID_CARD}), traversing_player, samples);
    }

    if (current_player != traversing_player) {
        int action_idx = state.get_rng()() % num_actions;
        return traverse(state.apply_action(legal_actions[action_idx]), traversing_player, samples);
    }

    std::vector<float> infoset_vec = featurize(state);
    
    // --- НОВАЯ ЛОГИКА: ОТПРАВКА ЗАПРОСА ВМЕСТО ПРЯМОГО ВЫЗОВА ---
    PredictionResult result = request_manager_->make_request(infoset_vec, num_actions);
    std::vector<float> regrets = result.regrets;
    // -----------------------------------------------------------

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
        action_utils[i] = traverse(state.apply_action(legal_actions[i]), traversing_player, samples);
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
