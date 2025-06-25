// mccfr_ofc-main/cpp_src/hand_evaluator.hpp

#pragma once
#include "card.hpp"
#include <omp/HandEvaluator.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <map>
#include <vector>
#include <algorithm>

namespace ofc {

    struct HandRank {
        int rank_value;
        int hand_class;
        std::string type_str;

        bool operator<(const HandRank& other) const {
            return rank_value < other.rank_value;
        }
    };

    class HandEvaluator {
    public:
        HandEvaluator() {
            class_to_string_map_ = {
                {1, "Straight Flush"}, {2, "Four of a Kind"}, {3, "Full House"},
                {4, "Flush"}, {5, "Straight"}, {6, "Three of a Kind"},
                {7, "Two Pair"}, {8, "Pair"}, {9, "High Card"}
            };
            init_3_card_lookup();
        }

        inline HandRank evaluate(const CardSet& cards) const {
            if (cards.size() == 5) {
                omp::Hand h = omp::Hand::empty();
                for (Card c : cards) h += omp::Hand(c);
                int rank_value = evaluator_5_card_.evaluate(h);
                int hand_class_omp = rank_value >> 12;
                int hand_class = (hand_class_omp == 0) ? 9 : 10 - hand_class_omp;
                return {rank_value, hand_class, class_to_string_map_.at(hand_class)};
            }
            if (cards.size() == 3) {
                auto it = evaluator_3_card_lookup_.find(get_3_card_key(cards));
                if (it != evaluator_3_card_lookup_.end()) return it->second;
            }
            return {9999, 9, "Invalid"};
        }

        inline int get_royalty(const CardSet& cards, const std::string& row_name) const {
            static const std::unordered_map<std::string, int> ROYALTY_BOTTOM = {{"Straight", 2}, {"Flush", 4}, {"Full House", 6}, {"Four of a Kind", 10}, {"Straight Flush", 15}, {"Royal Flush", 25}};
            static const std::unordered_map<std::string, int> ROYALTY_MIDDLE = {{"Three of a Kind", 2}, {"Straight", 4}, {"Flush", 8}, {"Full House", 12}, {"Four of a Kind", 20}, {"Straight Flush", 30}, {"Royal Flush", 50}};
            
            // --- ИЗМЕНЕНО: Финальная версия роялти согласно вашему запросу ---
            static const std::array<int, 13> ROYALTY_TOP_PAIRS = {0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 10, 11, 12}; // 66..JJ, QQ, KK, AA
            static const std::array<int, 13> ROYALTY_TOP_TRIPS = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}; // 222 -> 13, ..., AAA -> 25
            // ----------------------------------------------------------------

            if (cards.empty()) return 0;
            HandRank hr = evaluate(cards);

            if (row_name == "top") {
                if (hr.type_str == "Trips") {
                    int rank = get_rank(cards[0]);
                    if (rank >= 0 && rank < 13) return ROYALTY_TOP_TRIPS[rank];
                } else if (hr.type_str == "Pair") {
                    std::vector<int> ranks = {get_rank(cards[0]), get_rank(cards[1]), get_rank(cards[2])};
                    int pair_rank = (ranks[0] == ranks[1] || ranks[0] == ranks[2]) ? ranks[0] : ranks[1];
                    if (pair_rank >= 4 && pair_rank < 13) return ROYALTY_TOP_PAIRS[pair_rank];
                }
            } else if (row_name == "middle") {
                auto it = ROYALTY_MIDDLE.find(hr.type_str);
                return (it != ROYALTY_MIDDLE.end()) ? it->second : 0;
            } else if (row_name == "bottom") {
                auto it = ROYALTY_BOTTOM.find(hr.type_str);
                return (it != ROYALTY_BOTTOM.end()) ? it->second : 0;
            }
            return 0;
        }

    private:
        omp::HandEvaluator evaluator_5_card_;
        std::unordered_map<int, HandRank> evaluator_3_card_lookup_;
        std::unordered_map<int, std::string> class_to_string_map_;

        inline int get_3_card_key(const CardSet& cards) const {
            if (cards.size() != 3) return -1;
            std::vector<int> ranks = {get_rank(cards[0]), get_rank(cards[1]), get_rank(cards[2])};
            std::sort(ranks.rbegin(), ranks.rend());
            return ranks[0] * 169 + ranks[1] * 13 + ranks[2];
        }

        inline void init_3_card_lookup() {
            for (int r = 0; r <= 12; ++r) {
                evaluator_3_card_lookup_[r*169 + r*13 + r] = {13 - r, 6, "Trips"};
            }
            int rank_val = 14;
            for (int p = 12; p >= 0; --p) {
                for (int k = 12; k >= 0; --k) {
                    if (p == k) continue;
                    std::vector<int> ranks = {p, p, k};
                    std::sort(ranks.rbegin(), ranks.rend());
                    evaluator_3_card_lookup_[ranks[0]*169+ranks[1]*13+ranks[2]] = {rank_val++, 8, "Pair"};
                }
            }
            for (int r1 = 12; r1 >= 2; --r1) {
                for (int r2 = r1 - 1; r2 >= 1; --r2) {
                    for (int r3 = r2 - 1; r3 >= 0; --r3) {
                        evaluator_3_card_lookup_[r1*169+r2*13+r3] = {rank_val++, 9, "High Card"};
                    }
                }
            }
        }
    };
}
