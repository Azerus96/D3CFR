#include "DeepMCCFR.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace ofc {

DeepMCCFR::DeepMCCFR(py::function predict_callback, size_t action_limit) 
    : predict_callback_(predict_callback), action_limit_(action_limit) {
    if (!predict_callback) {
        throw std::runtime_error("Predict callback function is null.");
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
    const int player = state.get_current_player();
    const Board& my_board = state.get_player_board(player);
    const Board& opp_board = state.get_opponent_board(player);
    
    const int FEATURE_SIZE = 1486;
    std::vector<float> features(FEATURE_SIZE, 0.0f);
    int offset = 0;

    features[offset++] = static_cast<float>(state.get_street());
    features[offset++] = static_cast<float>(state.get_dealer_pos());
    features[offset++] = static_cast<float>(player);

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

    const auto& my_discards = state.get_my_discards(player);
    for (Card c : my_discards) {
        if (c != INVALID_CARD) features[offset + c] = 1.0f;
    }
    offset += 52;

    features[offset++] = static_cast<float>(state.get_opponent_discards(player).size());

    return features;
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
    py::list py_regrets = predict_callback_(infoset_vec, num_actions);
    std::vector<float> regrets = py_regrets.cast<std::vector<float>>();

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
