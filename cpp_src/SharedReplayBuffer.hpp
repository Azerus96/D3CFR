#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <mutex>
#include "constants.hpp" // <-- ИЗМЕНЕНО

namespace ofc {

// УДАЛЕНО: constexpr int INFOSET_SIZE = 1486;

struct TrainingSample {
    std::vector<float> infoset_vector;
    std::vector<float> target_regrets;
    int num_actions;

    TrainingSample(int action_limit) 
        : infoset_vector(INFOSET_SIZE), target_regrets(action_limit, 0.0f) {}
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
        std::lock_guard<std::mutex> lock(mtx_);
        uint64_t index = head_ % capacity_;
        head_++;

        auto& sample = buffer_[index];
        std::copy(infoset_vec.begin(), infoset_vec.end(), sample.infoset_vector.begin());
        
        std::fill(sample.target_regrets.begin(), sample.target_regrets.end(), 0.0f);
        
        if (num_actions > 0) {
            std::copy(regrets_vec.begin(), regrets_vec.end(), sample.target_regrets.begin());
        }
        sample.num_actions = num_actions;
        
        if (count_ < capacity_) {
            count_++;
        }
    }

    void sample(int batch_size, float* out_infosets, float* out_regrets) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        if (count_ < static_cast<uint64_t>(batch_size)) {
            std::fill(out_infosets, out_infosets + batch_size * INFOSET_SIZE, 0.0f);
            std::fill(out_regrets, out_regrets + batch_size * action_limit_, 0.0f);
            return;
        }

        std::uniform_int_distribution<uint64_t> dist(0, count_ - 1);

        for (int i = 0; i < batch_size; ++i) {
            uint64_t sample_idx = dist(rng_);
            const auto& sample = buffer_[sample_idx];
            std::copy(sample.infoset_vector.begin(), sample.infoset_vector.end(), out_infosets + i * INFOSET_SIZE);
            std::copy(sample.target_regrets.begin(), sample.target_regrets.end(), out_regrets + i * action_limit_);
        }
    }
    
    uint64_t get_count() {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }
    
    int get_max_actions() const {
        return action_limit_;
    }

private:
    std::vector<TrainingSample> buffer_;
    uint64_t capacity_;
    int action_limit_;
    uint64_t head_;
    uint64_t count_;
    std::mutex mtx_;
    std::mt19937 rng_;
};

}
