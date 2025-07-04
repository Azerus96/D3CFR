#pragma once

#include <vector>
#include <future>
#include "concurrentqueue.h" // <-- Подключаем заголовок быстрой lock-free очереди

// Запрос на инференс, который C++ отправляет в Python.
// Структура остается без изменений.
struct InferenceRequest {
    std::vector<float> infoset;
    std::promise<std::vector<float>> promise;
    int num_actions;
};

// --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
// Вместо самописного класса с мьютексом, который является "бутылочным горлышком"
// при большом количестве потоков, мы используем псевдоним (alias) для
// moodycamel::ConcurrentQueue.
//
// Это одна из самых быстрых в мире реализаций безблокировочной (lock-free)
// MPMC (Multi-Producer, Multi-Consumer) очереди. Она идеально подходит для
// нашего сценария MPSC (96 продюсеров на C++, 1 консьюмер на Python).
//
// Это изменение должно кардинально повысить производительность, так как
// 96 C++ потоков перестанут простаивать в ожидании блокировок.
using InferenceQueue = moodycamel::ConcurrentQueue<InferenceRequest>;
