// D2CFR-main/cpp_src/SharedReplayBuffer.hpp (ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)

#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <atomic>
#include <utility> // для std::pair

namespace ofc {

constexpr int INFOSET_SIZE = 1486;

struct TrainingSample {
    std::vector<float> infoset_vector;
    std::vector<float> target_regrets;
    int num_actions;

    TrainingSample(int action_limit) 
        : infoset_vector(INFOSET_SIZE), target_regrets(action_limit) {}
};

class SharedReplayBuffer {
public:
    SharedReplayBuffer(uint64_t capacity, int action_limit) 
        : capacity_(capacity), action_limit_(action_limit), head_(0), count_(0)
    {
        buffer_.reserve(capacity_);
        for (uint64_t i = 0; i < capacity_; ++i) {
            buffer_.emplace_back(action_limit_);
        }
        rng_.seed(std::random_device{}());
        std::cout << "C++: In-memory Replay Buffer created with capacity " << capacity 
                  << " and action_limit " << action_limit_ << std::endl;
    }

    void push(const std::vector<float>& infoset_vec, const std::vector<float>& regrets_vec, int num_actions) {
        // Используем блокировку только для критической секции
        std::lock_guard<std::mutex> lock(mtx_);
        uint64_t index = head_ % capacity_;
        head_++;

        std::copy(infoset_vec.begin(), infoset_vec.end(), buffer_[index].infoset_vector.begin());
        
        std::fill(buffer_[index].target_regrets.begin(), buffer_[index].target_regrets.end(), 0.0f);
        if (num_actions > 0 && num_actions <= action_limit_) {
            std::copy(regrets_vec.begin(), regrets_vec.end(), buffer_[index].target_regrets.begin());
        }
        buffer_[index].num_actions = num_actions;
        
        if (count_ < capacity_) {
            count_++;
        }
    }

    // ИЗМЕНЕНО: sample теперь возвращает пару векторов, что удобнее для pybind
    std::pair<std::vector<float>, std::vector<float>> sample(size_t batch_size) {
        std::vector<float> out_infosets;
        std::vector<float> out_regrets;
        out_infosets.reserve(batch_size * INFOSET_SIZE);
        out_regrets.reserve(batch_size * action_limit_);

        std::lock_guard<std::mutex> lock(mtx_);
        
        uint64_t current_count = get_count();
        if (current_count < batch_size) return {};

        std::uniform_int_distribution<uint64_t> dist(0, current_count - 1);

        for (size_t i = 0; i < batch_size; ++i) {
            uint64_t sample_idx = dist(rng_);
            const auto& sample = buffer_[sample_idx];
            out_infosets.insert(out_infosets.end(), sample.infoset_vector.begin(), sample.infoset_vector.end());
            out_regrets.insert(out_regrets.end(), sample.target_regrets.begin(), sample.target_regrets.end());
        }
        return {out_infosets, out_regrets};
    }
    
    uint64_t get_count() {
        return count_;
    }
    
    // ИЗМЕНЕНО: Название функции для консистентности
    int get_max_actions() const {
        return action_limit_;
    }

private:
    std::vector<TrainingSample> buffer_;
    uint64_t capacity_;
    int action_limit_;
    uint64_t head_; // Не нужен atomic, так как head_ изменяется под мьютексом
    uint64_t count_; // Не нужен atomic, так как count_ изменяется под мьютексом
    std::mutex mtx_;
    std::mt19937 rng_;
};

} // namespace ofc
