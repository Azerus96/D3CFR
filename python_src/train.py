# D2CFR-main/python_src/train.py (ВЕРСИЯ ДЛЯ ИЗМЕРЕНИЯ ИНИЦИАЛИЗАЦИИ - ПОЛНАЯ)

import os
import time
import torch
import torch.nn as nn
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# --- НАСТРОЙКИ, ИДЕНТИЧНЫЕ ИСХОДНОМУ ТЕСТУ ---
NUM_WORKERS = 24
NUM_COMPUTATION_THREADS = "4"
os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))

from .model import DuelingNetwork
from ofc_engine import DeepMCCFR, SharedReplayBuffer

# --- ГИПЕРПАРАМЕТРЫ, ИДЕНТИЧНЫЕ ИСХОДНОМУ ТЕСТУ ---
INPUT_SIZE = 1486 
# ВАЖНО: Используем ACTION_LIMIT=4 для прямого сравнения с самым первым тестом
ACTION_LIMIT = 4
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 2000000
BATCH_SIZE = 256
TRAINING_BLOCK_SIZE = 48
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"

def main():
    device = torch.device("cpu")
    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
    print("--- PROFILING INITIALIZATION TIME ---")
    print("Saving model to disk...", flush=True)
    model.eval()
    traced_script_module = torch.jit.trace(model, torch.randn(1, INPUT_SIZE))
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
    print("Model saved.", flush=True)

    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)

    # --- ЭКСПЕРИМЕНТ 1: ПОСЛЕДОВАТЕЛЬНАЯ ИНИЦИАЛИЗАЦИЯ ---
    print("\n--- Test 1: Sequential Initialization ---", flush=True)
    start_seq = time.time()
    solvers_seq = [DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, replay_buffer) for _ in range(NUM_WORKERS)]
    end_seq = time.time()
    print(f"\nSequential initialization of {NUM_WORKERS} workers took: {end_seq - start_seq:.2f} seconds.\n", flush=True)


    # --- ЭКСПЕРИМЕНТ 2: ПАРАЛЛЕЛЬНАЯ ИНИЦИАЛИЗАЦИЯ ---
    print("\n--- Test 2: Parallel Initialization ---", flush=True)
    
    def create_solver():
        # Эта функция будет выполняться в каждом потоке
        return DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, replay_buffer)

    start_par = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Запускаем создание солверов параллельно
        futures = [executor.submit(create_solver) for _ in range(NUM_WORKERS)]
        # Собираем результаты, чтобы дождаться завершения всех
        solvers_par = [future.result() for future in concurrent.futures.as_completed(futures)]
    end_par = time.time()
    print(f"\nParallel initialization of {NUM_WORKERS} workers took: {end_par - start_par:.2f} seconds.", flush=True)
    
    print("\n--- ANALYSIS ---")
    if (end_seq - start_seq) > 0.01: # Добавляем проверку, чтобы избежать деления на ноль
        slowdown_factor = (end_par - start_par) / (end_seq - start_seq)
        print(f"The difference is significant. Parallel loading is {slowdown_factor:.2f}x slower due to resource contention.")
    else:
        print("Sequential loading was too fast to measure, cannot calculate slowdown factor.")
    print("Check the C++ output above to see individual thread loading times.")

    # --- Запуск короткого цикла сбора данных для проверки работоспособности ---
    print("\n--- Running a short data collection cycle ---", flush=True)
    start_run = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        tasks_per_worker = TRAINING_BLOCK_SIZE // NUM_WORKERS
        run_futures = [executor.submit(solver.run_traversal) for solver in solvers_par for _ in range(tasks_per_worker)]
        concurrent.futures.wait(run_futures)
    end_run = time.time()
    
    duration = end_run - start_run
    throughput = replay_buffer.get_count() / duration if duration > 0 else 0
    
    print(f"Data collection finished in {duration:.2f}s.")
    print(f"Buffer size: {replay_buffer.get_count()}")
    print(f"Throughput: {throughput:.2f} samples/sec")


if __name__ == "__main__":
    main()
