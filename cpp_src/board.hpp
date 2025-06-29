// D2CFR-main/cpp_src/board.hpp (ПОЛНАЯ ВЕРСИЯ ДЛЯ ЭТАПА 1)

#pragma once
#include "card.hpp"
#include "hand_evaluator.hpp"
#include <array>
#include <string>
#include <vector>
#include <numeric>

namespace ofc {

    class Board {
    public:
        std::array<Card, 3> top;
        std::array<Card, 5> middle;
        std::array<Card, 5> bottom;

        Board() {
            top.fill(INVALID_CARD);
            middle.fill(INVALID_CARD);
            bottom.fill(INVALID_CARD);
        }

        // УЛУЧШЕНО: API изменен для предотвращения лишних аллокаций памяти.
        // Функция теперь заполняет переданный по ссылке вектор.
        inline void get_row_cards(const std::string& row_name, CardSet& out_cards) const {
            out_cards.clear();
            if (row_name == "top") {
                for(Card c : top) if (c != INVALID_CARD) out_cards.push_back(c);
            } else if (row_name == "middle") {
                for(Card c : middle) if (c != INVALID_CARD) out_cards.push_back(c);
            } else if (row_name == "bottom") {
                for(Card c : bottom) if (c != INVALID_CARD) out_cards.push_back(c);
            }
        }

        inline int get_card_count() const {
            int count = 0;
            for(Card c : top) if (c != INVALID_CARD) count++;
            for(Card c : middle) if (c != INVALID_CARD) count++;
            for(Card c : bottom) if (c != INVALID_CARD) count++;
            return count;
        }

        // УЛУЧШЕНО: Принимает буферы по ссылке, чтобы избежать внутренних аллокаций.
        inline bool is_foul(const HandEvaluator& evaluator, CardSet& top_cards_buf, CardSet& mid_cards_buf, CardSet& bot_cards_buf) const {
            if (get_card_count() != 13) return false;
            
            get_row_cards("top", top_cards_buf);
            get_row_cards("middle", mid_cards_buf);
            get_row_cards("bottom", bot_cards_buf);

            HandRank top_rank = evaluator.evaluate(top_cards_buf);
            HandRank mid_rank = evaluator.evaluate(mid_cards_buf);
            HandRank bot_rank = evaluator.evaluate(bot_cards_buf);
            return (mid_rank < bot_rank) || (top_rank < mid_rank);
        }

        // УЛУЧШЕНО: Принимает буферы по ссылке.
        inline int get_total_royalty(const HandEvaluator& evaluator, CardSet& top_cards_buf, CardSet& mid_cards_buf, CardSet& bot_cards_buf) const {
            if (is_foul(evaluator, top_cards_buf, mid_cards_buf, bot_cards_buf)) return 0;

            // Векторы уже заполнены вызовом is_foul
            return evaluator.get_royalty(top_cards_buf, "top") +
                   evaluator.get_royalty(mid_cards_buf, "middle") +
                   evaluator.get_royalty(bot_cards_buf, "bottom");
        }

        // УЛУЧШЕНО: Принимает буферы по ссылке.
        inline bool qualifies_for_fantasyland(const HandEvaluator& evaluator, CardSet& top_cards_buf, CardSet& mid_cards_buf, CardSet& bot_cards_buf) const {
            if (is_foul(evaluator, top_cards_buf, mid_cards_buf, bot_cards_buf)) return false;
            
            // top_cards_buf уже заполнен вызовом is_foul
            if (top_cards_buf.size() != 3) return false;
            HandRank hr = evaluator.evaluate(top_cards_buf);
            if (hr.type_str == "Pair") {
                int r0 = get_rank(top_cards_buf[0]), r1 = get_rank(top_cards_buf[1]), r2 = get_rank(top_cards_buf[2]);
                int pair_rank = (r0 == r1 || r0 == r2) ? r0 : r1;
                return pair_rank >= 10; // Ранг 10 - это Дама (Q)
            }
            return hr.type_str == "Trips";
        }

        // УЛУЧШЕНО: Принимает буферы по ссылке.
        inline int get_fantasyland_card_count(const HandEvaluator& evaluator, CardSet& top_cards_buf, CardSet& mid_cards_buf, CardSet& bot_cards_buf) const {
            if (!qualifies_for_fantasyland(evaluator, top_cards_buf, mid_cards_buf, bot_cards_buf)) return 0;
            
            // top_cards_buf уже заполнен
            HandRank hr = evaluator.evaluate(top_cards_buf);
            if (hr.type_str == "Trips") return 17;
            if (hr.type_str == "Pair") {
                int r0 = get_rank(top_cards_buf[0]), r1 = get_rank(top_cards_buf[1]), r2 = get_rank(top_cards_buf[2]);
                int pair_rank = (r0 == r1 || r0 == r2) ? r0 : r1;
                if (pair_rank == 10) return 14; // QQ
                if (pair_rank == 11) return 15; // KK
                if (pair_rank == 12) return 16; // AA
            }
            return 0;
        }
    };
}
