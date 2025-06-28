// D2CFR-main/cpp_src/SharedReplayBuffer.hpp

#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <mutex> // <-- ИЗМЕНЕНО: Используем стандартный мьютекс
#include <atomic> // <-- ИЗМЕНЕНО: Используем атомарные переменные для счетчиков

namespace ofc {

// Определяем константы для размеров векторов.
constexpr int INFOSET_SIZE = 1486;
constexpr int ACTION_LIMIT = 24;

// Структура для хранения одного обучающего примера.
// Теперь это обычная структура, а не для общей памяти.
struct TrainingSample {
    std::vector<float> infoset_vector;
    std::vector<float> target_regrets;
    int num_actions;

    TrainingSample() 
        : infoset_vector(INFOSET_SIZE), target_regrets(ACTION_LIMIT) {}
};

// Класс буфера теперь управляет обычным вектором и защищает его мьютексом.
class SharedReplayBuffer {
public:
    // Конструктор теперь не принимает имя общей памяти.
    SharedReplayBuffer(uint64_t capacity) 
        : capacity_(capacity), head_(0), count_(0)
    {
        // Просто резервируем память в векторе.
        buffer_.resize(capacity_);
        rng_.seed(std::random_device{}());
        std::cout << "C++: In-memory Replay Buffer created with capacity " << capacity << std::endl;
    }

    // Метод для записи данных из C++ воркеров. Потокобезопасный.
    void push(const std::vector<float>& infoset_vec, const std::vector<float>& regrets_vec, int num_actions) {
        // Атомарно получаем индекс для вставки. Это быстрая, lock-free операция.
        uint64_t index = head_.fetch_add(1) % capacity_;

        // Блокируем мьютекс только на время копирования в конкретную ячейку.
        // Это минимизирует время блокировки.
        {
            std::lock_guard<std::mutex> lock(mtx_);
            // Копируем данные в ячейку в векторе.
            std::copy(infoset_vec.begin(), infoset_vec.end(), buffer_[index].infoset_vector.begin());
            
            std::fill(buffer_[index].target_regrets.begin(), buffer_[index].target_regrets.end(), 0.0f);
            if (num_actions > 0) {
                std::copy(regrets_vec.begin(), regrets_vec.end(), buffer_[index].target_regrets.begin());
            }
            buffer_[index].num_actions = num_actions;
        }

        // Атомарно увеличиваем счетчик, пока буфер не заполнится.
        if (count_ < capacity_) {
            count_.fetch_add(1);
        }
    }

    // Метод для получения батча. Вызывается из Python через pybind11.
    void sample(int batch_size, float* out_infosets, float* out_regrets) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        uint64_t current_count = get_count();
        if (current_count < batch_size) return;

        std::uniform_int_distribution<uint64_t> dist(0, current_count - 1);

        for (int i = 0; i < batch_size; ++i) {
            uint64_t sample_idx = dist(rng_);
            // Копируем данные из нашего вектора напрямую в память NumPy-массива
            const auto& sample = buffer_[sample_idx];
            std::copy(sample.infoset_vector.begin(), sample.infoset_vector.end(), out_infosets + i * INFOSET_SIZE);
            std::copy(sample.target_regrets.begin(), sample.target_regrets.end(), out_regrets + i * ACTION_LIMIT);
        }
    }
    
    // Получить текущее количество элементов в буфере.
    uint64_t get_count() {
        return count_.load();
    }

    // Статический метод для очистки больше не нужен.

private:
    std::vector<TrainingSample> buffer_;
    uint64_t capacity_;
    std::atomic<uint64_t> head_;
    std::atomic<uint64_t> count_;
    std::mutex mtx_; // Мьютекс для защиты операций
    std::mt19937 rng_;
};

} // namespace ofc
