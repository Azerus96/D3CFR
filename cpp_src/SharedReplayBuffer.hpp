// D2CFR-main/cpp_src/SharedReplayBuffer.hpp (ИСПРАВЛЕННАЯ ВЕРСИЯ)

#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <atomic>

namespace ofc {

// УДАЛЯЕМ КОНСТАНТУ ACTION_LIMIT ОТСЮДА
constexpr int INFOSET_SIZE = 1486;
// constexpr int ACTION_LIMIT = 24; // <-- УДАЛИТЬ ЭТУ СТРОКУ

struct TrainingSample {
    std::vector<float> infoset_vector;
    std::vector<float> target_regrets;
    int num_actions;

    // ИЗМЕНЕНИЕ: Конструктор теперь принимает размер вектора регретов
    TrainingSample(int action_limit) 
        : infoset_vector(INFOSET_SIZE), target_regrets(action_limit) {}
};

class SharedReplayBuffer {
public:
    // ИЗМЕНЕНИЕ: Конструктор теперь принимает и capacity, и action_limit
    SharedReplayBuffer(uint64_t capacity, int action_limit) 
        : capacity_(capacity), action_limit_(action_limit), head_(0), count_(0)
    {
        // Инициализируем вектор сэмплов с правильным action_limit
        buffer_.reserve(capacity_);
        for (uint64_t i = 0; i < capacity_; ++i) {
            buffer_.emplace_back(action_limit_);
        }
        rng_.seed(std::random_device{}());
        std::cout << "C++: In-memory Replay Buffer created with capacity " << capacity 
                  << " and action_limit " << action_limit_ << std::endl;
    }

    void push(const std::vector<float>& infoset_vec, const std::vector<float>& regrets_vec, int num_actions) {
        uint64_t index = head_.fetch_add(1) % capacity_;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            std::copy(infoset_vec.begin(), infoset_vec.end(), buffer_[index].infoset_vector.begin());
            
            std::fill(buffer_[index].target_regrets.begin(), buffer_[index].target_regrets.end(), 0.0f);
            // Проверяем, что не выйдем за пределы
            if (num_actions > 0 && num_actions <= action_limit_) {
                std::copy(regrets_vec.begin(), regrets_vec.end(), buffer_[index].target_regrets.begin());
            }
            buffer_[index].num_actions = num_actions;
        }
        if (count_ < capacity_) {
            count_.fetch_add(1);
        }
    }

    // ИЗМЕНЕНИЕ: sample теперь использует action_limit_
    void sample(int batch_size, float* out_infosets, float* out_regrets) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        uint64_t current_count = get_count();
        if (current_count < batch_size) return;

        std::uniform_int_distribution<uint64_t> dist(0, current_count - 1);

        for (int i = 0; i < batch_size; ++i) {
            uint64_t sample_idx = dist(rng_);
            const auto& sample = buffer_[sample_idx];
            std::copy(sample.infoset_vector.begin(), sample.infoset_vector.end(), out_infosets + i * INFOSET_SIZE);
            std::copy(sample.target_regrets.begin(), sample.target_regrets.end(), out_regrets + i * action_limit_);
        }
    }
    
    uint64_t get_count() {
        return count_.load();
    }
    
    // ИЗМЕНЕНИЕ: Добавляем геттер для action_limit
    int get_action_limit() const {
        return action_limit_;
    }

private:
    std::vector<TrainingSample> buffer_;
    uint64_t capacity_;
    int action_limit_; // <-- ДОБАВЛЕНО
    std::atomic<uint64_t> head_;
    std::atomic<uint64_t> count_;
    std::mutex mtx_;
    std::mt19937 rng_;
};

} // namespace ofc
