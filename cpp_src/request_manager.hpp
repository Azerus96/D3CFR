#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <atomic>

namespace ofc {

struct PredictionRequest {
    uint64_t id;
    std::vector<float> infoset_vector;
    int num_actions;
};

struct PredictionResult {
    uint64_t id;
    std::vector<float> regrets;
};

class RequestManager {
public:
    RequestManager() : next_request_id_(0) {}

    // Вызывается из C++ потока для отправки запроса
    PredictionResult make_request(const std::vector<float>& infoset_vector, int num_actions) {
        std::unique_lock<std::mutex> lock(mtx_);
        
        uint64_t id = next_request_id_++;
        request_queue_.push({id, infoset_vector, num_actions});
        
        // Уведомляем Python, что есть новый запрос
        cv_python_.notify_one();

        // Ждем, пока Python обработает наш запрос
        cv_cpp_.wait(lock, [&]{ return response_map_.count(id); });
        
        PredictionResult result = response_map_[id];
        response_map_.erase(id);
        
        return result;
    }

    // Вызывается из Python для получения пачки запросов
    std::vector<PredictionRequest> get_requests(size_t max_batch_size) {
        std::unique_lock<std::mutex> lock(mtx_);
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
        std::unique_lock<std::mutex> lock(mtx_);
        for (const auto& result : results) {
            response_map_[result.id] = result;
        }
        // Будим все ожидающие C++ потоки
        cv_cpp_.notify_all();
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_cpp_;
    std::condition_variable cv_python_;
    
    std::queue<PredictionRequest> request_queue_;
    std::map<uint64_t, PredictionResult> response_map_;
    
    std::atomic<uint64_t> next_request_id_;
};

} // namespace ofc
