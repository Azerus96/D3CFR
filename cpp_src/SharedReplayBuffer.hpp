// cpp_src/SharedReplayBuffer.hpp

#pragma once

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <atomic>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>

namespace ofc {

// Определяем константы для размеров векторов.
// Это важно для выделения памяти фиксированного размера.
constexpr int INFOSET_SIZE = 1486;
constexpr int ACTION_LIMIT = 24;

// Структура для хранения одного обучающего примера в общей памяти.
// Используем C-style массивы, так как std::vector не может быть размещен в SHM напрямую.
// alignas(64) помогает с производительностью, выравнивая данные по линии кэша.
struct alignas(64) SharedTrainingSample {
    float infoset_vector[INFOSET_SIZE];
    float target_regrets[ACTION_LIMIT];
    int num_actions;
};

// Метаданные для управления буфером. Также будут находиться в общей памяти.
struct SharedBufferHeader {
    std::atomic<uint64_t> head;      // Атомарный указатель на следующую ячейку для записи
    std::atomic<uint64_t> count;     // Атомарный счетчик текущего количества элементов
    uint64_t capacity;               // Максимальная вместимость буфера
    boost::interprocess::interprocess_mutex mtx; // Мьютекс для защиты операции сэмплирования
};

class SharedReplayBuffer {
public:
    // Конструктор. Открывает существующий или создает новый сегмент общей памяти.
    SharedReplayBuffer(const std::string& shm_name, uint64_t capacity) 
        : segment_(boost::interprocess::open_or_create, shm_name.c_str(), 
                   sizeof(SharedBufferHeader) + capacity * sizeof(SharedTrainingSample) + 1024), // +1KB запаса
          capacity_(capacity)
    {
        // Находим или создаем заголовок и массив сэмплов в общей памяти.
        // `find_or_construct` гарантирует, что объект будет создан только один раз.
        header_ = segment_.find_or_construct<SharedBufferHeader>("Header")();
        samples_ = segment_.find_or_construct<SharedTrainingSample>("Samples")[capacity_]();

        // Если буфер создается впервые, инициализируем его capacity.
        if (header_->capacity == 0) {
            header_->capacity = capacity_;
            header_->head = 0;
            header_->count = 0;
        }
        
        // Инициализируем генератор случайных чисел для сэмплирования
        rng_.seed(std::random_device{}());
    }

    // Метод для записи данных из C++ воркеров. Потокобезопасный.
    void push(const std::vector<float>& infoset_vec, const std::vector<float>& regrets_vec, int num_actions) {
        // Атомарно получаем индекс для вставки и увеличиваем head.
        // Это быстрая, lock-free операция.
        uint64_t index = header_->head.fetch_add(1) % capacity_;

        // Копируем данные в ячейку в общей памяти.
        // Это единственное копирование, и оно происходит внутри C++ на максимальной скорости.
        std::copy(infoset_vec.begin(), infoset_vec.end(), samples_[index].infoset_vector);
        
        // Заполняем "хвост" нулями, если вектор сожалений меньше лимита
        std::fill(samples_[index].target_regrets, samples_[index].target_regrets + ACTION_LIMIT, 0.0f);
        if (num_actions > 0) {
            std::copy(regrets_vec.begin(), regrets_vec.end(), samples_[index].target_regrets);
        }
        samples_[index].num_actions = num_actions;

        // Атомарно увеличиваем счетчик, пока буфер не заполнится.
        if (header_->count < capacity_) {
            header_->count.fetch_add(1);
        }
    }

    // Метод для получения батча. Вызывается из Python через pybind11.
    // Заполняет предоставленные Python'ом (через NumPy) массивы.
    void sample(int batch_size, float* out_infosets, float* out_regrets) {
        // Блокируем мьютекс на время сэмплирования, чтобы избежать гонок
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(header_->mtx);
        
        uint64_t current_count = get_count();
        if (current_count < batch_size) return;

        std::uniform_int_distribution<uint64_t> dist(0, current_count - 1);

        for (int i = 0; i < batch_size; ++i) {
            uint64_t sample_idx = dist(rng_);
            // Копируем данные из общей памяти напрямую в память NumPy-массива
            std::copy(samples_[sample_idx].infoset_vector, samples_[sample_idx].infoset_vector + INFOSET_SIZE, out_infosets + i * INFOSET_SIZE);
            std::copy(samples_[sample_idx].target_regrets, samples_[sample_idx].target_regrets + ACTION_LIMIT, out_regrets + i * ACTION_LIMIT);
        }
    }
    
    // Получить текущее количество элементов в буфере.
    uint64_t get_count() {
        return header_->count.load();
    }

    // Статический метод для очистки общей памяти.
    static void cleanup(const std::string& shm_name) {
        boost::interprocess::shared_memory_object::remove(shm_name.c_str());
    }

private:
    boost::interprocess::managed_shared_memory segment_;
    SharedBufferHeader* header_;
    SharedTrainingSample* samples_;
    uint64_t capacity_;
    std::mt19937 rng_;
};

} // namespace ofc
