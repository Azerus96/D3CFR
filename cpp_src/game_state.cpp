#include "game_state.hpp"

namespace ofc {
    // === ВОЗВРАЩАЕМ ОБЫЧНОЕ ОПРЕДЕЛЕНИЕ ===
    std::mt19937 GameState::rng_(std::random_device{}());
}
