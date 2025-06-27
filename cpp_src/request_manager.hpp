// D2CFR-main/cpp_src/request_manager.hpp

#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <atomic>
#include <memory> // для std::shared_ptr

namespace ofc {

// Структуры для запросов и базовых результатов остаются без изменений
struct PredictionRequest {
    uint64_t id;
    std::vector<float> infoset_vector;
    int num_actions;
};

struct PredictionResult {
    uint64_t id;
    std::vector<float> regrets;
};

// НОВЫЙ КЛАСС: "Обещание" для конкретного результата
class PredictionPromise {
public:
    PredictionPromise() : ready_(false) {}

    // Метод для получения результата. Блокирует поток до готовности.
    PredictionResult get() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]{ return ready_; });
        return result_;
    }

    // Метод для установки результата. Вызывается менеджером.
    void set(const PredictionResult& result) {
        std::unique_lock<std::mutex> lock(mtx_);
        result_ = result;
        ready_ = true;
        cv_.notify_one(); // Будим только один ожидающий поток
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    PredictionResult result_;
    bool ready_;
};


// ОБНОВЛЕННЫЙ КЛАСС: RequestManager
class RequestManager {
public:
    RequestManager() : next_request_id_(0) {}

    // Вызывается из C++ потока. Теперь возвращает "обещание".
    std::shared_ptr<PredictionPromise> make_request(const std::vector<float>& infoset_vector, int num_actions) {
        auto promise = std::make_shared<PredictionPromise>();
        
        std::unique_lock<std::mutex> lock(queue_mtx_);
        
        uint64_t id = next_request_id_++;
        
        // Сохраняем обещание в мапе, чтобы потом его найти и исполнить
        pending_promises_[id] = promise;
        
        request_queue_.push({id, infoset_vector, num_actions});
        
        // Уведомляем Python, что есть новый запрос
        cv_python_.notify_one();
        
        return promise;
    }

    // Вызывается из Python для получения пачки запросов (без изменений)
    std::vector<PredictionRequest> get_requests(size_t max_batch_size) {
        std::unique_lock<std::mutex> lock(queue_mtx_);
        cv_python_.wait(lock, [&]{ return !request_queue_.empty(); });
        
        std::vector<PredictionRequest> requests;
        while (!request_queue_.empty() && requests.size() < max_batch_size) {
            requests.push_back(request_queue_.front());
            request_queue_.pop();
        }
        return requests;
    }

    // Вызывается из Python для отправки пачки результатов
    void post_results(const std::vector<PredictionResult>& results) {
        std::unique_lock<std::mutex> lock(queue_mtx_);
        for (const auto& result : results) {
            auto it = pending_promises_.find(result.id);
            if (it != pending_promises_.end()) {
                // Находим обещание и исполняем его
                it->second->set(result);
                // Удаляем из мапы, так как оно больше не нужно
                pending_promises_.erase(it);
            }
        }
    }

private:
    // Мьютекс для защиты очереди запросов и мапы обещаний
    std::mutex queue_mtx_;
    std::condition_variable cv_python_;
    
    std::queue<PredictionRequest> request_queue_;
    // Мапа для хранения "обещаний", ожидающих исполнения
    std::map<uint64_t, std::shared_ptr<PredictionPromise>> pending_promises_;
    
    std::atomic<uint64_t> next_request_id_;
};

} // namespace ofc
