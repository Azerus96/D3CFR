#pragma once
#include "board.hpp"
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <functional>
#include <map>
#include <string>
#include <unordered_set>

namespace ofc {

    class GameState {
    public:
        // ИЗМЕНЕНО: Конструктор теперь может принимать начальное состояние колоды
        GameState(int num_players = 2, int dealer_pos = -1, const CardSet& initial_deck = {})
            : rng_(std::random_device{}()), num_players_(num_players), street_(1), boards_(num_players) {
            
            if (initial_deck.empty()) {
                deck_.resize(52);
                std::iota(deck_.begin(), deck_.end(), 0);
                std::shuffle(deck_.begin(), deck_.end(), rng_);
            } else {
                deck_ = initial_deck;
                // Не перемешиваем, если колода задана извне
            }

            if (dealer_pos == -1) {
                std::uniform_int_distribution<int> dist(0, num_players - 1);
                dealer_pos_ = dist(rng_);
            } else {
                this->dealer_pos_ = dealer_pos;
            }
            current_player_ = (dealer_pos_ + 1) % num_players_;
            
            // ИЗМЕНЕНО: Инициализируем новые поля для сбросов
            my_discards_.resize(num_players);
            opponent_discard_counts_.resize(num_players, 0);

            deal_cards();
        }

        GameState(const GameState& other) = default;

        inline bool is_terminal() const {
            return street_ > 5 || boards_[0].get_card_count() == 13;
        }

        inline std::pair<float, float> get_payoffs(const HandEvaluator& evaluator) const {
            // ... (логика подсчета очков остается без изменений) ...
            const int SCOOP_BONUS = 3;
            const Board& p1_board = boards_[0];
            const Board& p2_board = boards_[1];
            bool p1_foul = p1_board.is_foul(evaluator);
            bool p2_foul = p2_board.is_foul(evaluator);
            int p1_royalty = p1_foul ? 0 : p1_board.get_total_royalty(evaluator);
            int p2_royalty = p2_foul ? 0 : p2_board.get_total_royalty(evaluator);

            if (p1_foul && p2_foul) return {0.0f, 0.0f};
            if (p1_foul) return {-(float)(SCOOP_BONUS + p2_royalty), (float)(SCOOP_BONUS + p2_royalty)};
            if (p2_foul) return {(float)(SCOOP_BONUS + p1_royalty), -(float)(SCOOP_BONUS + p1_royalty)};

            int line_score = 0;
            CardSet p1_top, p1_mid, p1_bot, p2_top, p2_mid, p2_bot;
            p1_board.get_row_cards("top", p1_top);
            p1_board.get_row_cards("middle", p1_mid);
            p1_board.get_row_cards("bottom", p1_bot);
            p2_board.get_row_cards("top", p2_top);
            p2_board.get_row_cards("middle", p2_mid);
            p2_board.get_row_cards("bottom", p2_bot);

            if (evaluator.evaluate(p1_top) < evaluator.evaluate(p2_top)) line_score++; else line_score--;
            if (evaluator.evaluate(p1_mid) < evaluator.evaluate(p2_mid)) line_score++; else line_score--;
            if (evaluator.evaluate(p1_bot) < evaluator.evaluate(p2_bot)) line_score++; else line_score--;

            if (abs(line_score) == 3) line_score = (line_score > 0) ? SCOOP_BONUS : -SCOOP_BONUS;
            float p1_total = (float)(line_score + p1_royalty - p2_royalty);
            return {p1_total, -p1_total};
        }

        inline std::vector<Action> get_legal_actions(size_t action_limit) const {
            // ... (логика генерации действий остается без изменений) ...
            std::vector<Action> actions;
            if (is_terminal()) return actions;

            CardSet cards_to_place;
            if (street_ == 1) {
                cards_to_place = dealt_cards_;
                generate_random_placements(cards_to_place, INVALID_CARD, actions, action_limit);
            } else {
                size_t limit_per_discard = action_limit / 3 + 1;
                for (int i = 0; i < 3; ++i) {
                    CardSet current_placement_cards;
                    Card current_discarded = dealt_cards_[i];
                    for (int j = 0; j < 3; ++j) {
                        if (i != j) current_placement_cards.push_back(dealt_cards_[j]);
                    }
                    generate_random_placements(current_placement_cards, current_discarded, actions, limit_per_discard);
                }
            }
            
            std::shuffle(actions.begin(), actions.end(), rng_);
            if (actions.size() > action_limit) {
                actions.resize(action_limit);
            }
            
            return actions;
        }

        // ИЗМЕНЕНО: apply_action теперь принимает 'player_view', чтобы знать, чьими глазами смотреть на мир
        inline GameState apply_action(const Action& action, int player_view) const {
            GameState next_state(*this);
            const auto& placements = action.first;
            const Card& discarded_card = action.second;

            for (const auto& p : placements) {
                const Card& card = p.first;
                const std::string& row = p.second.first;
                int idx = p.second.second;
                if (row == "top") next_state.boards_[current_player_].top[idx] = card;
                else if (row == "middle") next_state.boards_[current_player_].middle[idx] = card;
                else if (row == "bottom") next_state.boards_[current_player_].bottom[idx] = card;
            }

            // ИЗМЕНЕНО: Логика обработки сброса
            if (discarded_card != INVALID_CARD) {
                if (current_player_ == player_view) {
                    // Если это наш ход, мы знаем свой сброс
                    next_state.my_discards_[current_player_].push_back(discarded_card);
                } else {
                    // Если это ход оппонента, мы знаем только, что он сбросил карту
                    next_state.opponent_discard_counts_[player_view]++;
                }
            }

            if (next_state.current_player_ == next_state.dealer_pos_) next_state.street_++;
            next_state.current_player_ = (next_state.current_player_ + 1) % num_players_;
            
            if (!next_state.is_terminal()) next_state.deal_cards();
            return next_state;
        }
        
        // --- ИЗМЕНЕНО: Обновленные геттеры ---
        int get_street() const { return street_; }
        int get_current_player() const { return current_player_; }
        const CardSet& get_dealt_cards() const { return dealt_cards_; }
        const Board& get_player_board(int player_idx) const { return boards_[player_idx]; }
        const Board& get_opponent_board(int player_idx) const { return boards_[(player_idx + 1) % num_players_]; }
        const CardSet& get_my_discards(int player_idx) const { return my_discards_[player_idx]; }
        int get_opponent_discard_count(int player_idx) const { return opponent_discard_counts_[player_idx]; }
        int get_dealer_pos() const { return dealer_pos_; }
        std::mt19937& get_rng() { return rng_; }

    private:
        inline void deal_cards() {
            int num_to_deal = (street_ == 1) ? 5 : 3;
            if (deck_.size() < (size_t)num_to_deal) {
                street_ = 6; return;
            }
            dealt_cards_.assign(deck_.end() - num_to_deal, deck_.end());
            deck_.resize(deck_.size() - num_to_deal);
        }

        inline void generate_random_placements(const CardSet& cards, Card discarded, std::vector<Action>& actions, size_t limit) const {
            // ... (логика генерации расстановок остается без изменений) ...
            const Board& board = boards_[current_player_];
            std::vector<std::pair<std::string, int>> available_slots;
            for(int i=0; i<3; ++i) if(board.top[i] == INVALID_CARD) available_slots.push_back({"top", i});
            for(int i=0; i<5; ++i) if(board.middle[i] == INVALID_CARD) available_slots.push_back({"middle", i});
            for(int i=0; i<5; ++i) if(board.bottom[i] == INVALID_CARD) available_slots.push_back({"bottom", i});

            if (available_slots.size() < cards.size()) return;

            std::unordered_set<std::string> seen_placements;
            
            for (size_t i = 0; i < limit * 3 && actions.size() < limit; ++i) {
                std::shuffle(available_slots.begin(), available_slots.end(), rng_);
                
                CardSet shuffled_cards = cards;
                std::shuffle(shuffled_cards.begin(), shuffled_cards.end(), rng_);

                std::vector<Placement> current_placement;
                for(size_t j = 0; j < cards.size(); ++j) {
                    current_placement.push_back({shuffled_cards[j], available_slots[j]});
                }
                
                std::sort(current_placement.begin(), current_placement.end(), 
                    [](const Placement& a, const Placement& b){
                        if (a.second.first != b.second.first) return a.second.first < b.second.first;
                        return a.second.second < b.second.second;
                });

                std::string key;
                for(const auto& p : current_placement) {
                    key += std::to_string(p.first) + p.second.first + std::to_string(p.second.second);
                }

                if (seen_placements.find(key) == seen_placements.end()) {
                    seen_placements.insert(key);
                    actions.push_back({current_placement, discarded});
                }
            }
        }

        int num_players_;
        int street_;
        int dealer_pos_;
        int current_player_;
        std::vector<Board> boards_;
        CardSet deck_;
        CardSet dealt_cards_;
        
        // ИЗМЕНЕНО: Новые поля для хранения сбросов
        std::vector<CardSet> my_discards_;
        std::vector<int> opponent_discard_counts_;
        
        mutable std::mt19937 rng_;
    };
}
