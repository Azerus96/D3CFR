# D2CFR-main/python_src/train.py (ВЕРСИЯ ДЛЯ КОРРЕКТНОГО СРАВНЕНИЯ)

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import traceback
from multiprocessing import Process, Queue, cpu_count
from collections import deque
import sys
import concurrent.futures

# --- НАСТРОЙКИ, ИДЕНТИЧНЫЕ ИСХОДНОМУ ТЕСТУ ---
NUM_WORKERS = 24
NUM_COMPUTATION_THREADS = "4"
os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))

# --- ГИПЕРПАРАМЕТРЫ, ИДЕНТИЧНЫЕ ИСХОДНОМУ ТЕСТУ ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 4
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 2000000
BATCH_SIZE = 256
TRAINING_BLOCK_SIZE = 48
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"

# Поздний импорт
from .model import DuelingNetwork

def worker_process(model_path, action_limit, queue, build_path, num_traversals):
    """
    Функция воркера. Выполняет заданное число траверсов и завершается.
    """
    if build_path not in sys.path:
        sys.path.insert(0, build_path)
    
    try:
        from ofc_engine import DeepMCCFR
        solver = DeepMCCFR(model_path, action_limit, queue)
        for _ in range(num_traversals):
            solver.run_traversal()
    except Exception as e:
        print(f"Error in worker process (PID: {os.getpid()}): {e}")
        traceback.print_exc()

def main():
    device = torch.device("cpu")
    print(f"Main process (PID: {os.getpid()}) using device: {device}", flush=True)
    print(f"--- CORRECTED FINAL TEST: Comparing Architectures ---", flush=True)
    print(f"Settings: NUM_WORKERS={NUM_WORKERS}, ACTION_LIMIT={ACTION_LIMIT}, TRAINING_BLOCK_SIZE={TRAINING_BLOCK_SIZE}")

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- Квантование и сохранение модели ---
    print("\nQuantizing and saving model for workers...", flush=True)
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    traced_script_module = torch.jit.trace(quantized_model, torch.randn(1, INPUT_SIZE))
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
    print(f"Quantized model saved to {TORCHSCRIPT_MODEL_PATH}", flush=True)

    # --- Основной цикл, имитирующий старую структуру ---
    replay_buffer = deque(maxlen=REPLAY_BUFFER_CAPACITY)
    total_traversals = 0
    
    try:
        for block_counter in range(1, 3): # Запустим 2 блока для надежности
            print(f"\n--- Block {block_counter} ---", flush=True)
            
            data_queue = Queue()
            start_time = time.time()
            initial_buffer_size = len(replay_buffer)

            # --- Запуск процессов-воркеров для одного блока ---
            processes = []
            tasks_per_worker = TRAINING_BLOCK_SIZE // NUM_WORKERS
            build_path = os.path.abspath("./build")
            
            print(f"Submitting {TRAINING_BLOCK_SIZE} traversal tasks ({tasks_per_worker} per worker)...", flush=True)
            for _ in range(NUM_WORKERS):
                p = Process(target=worker_process, args=(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, data_queue, build_path, tasks_per_worker))
                p.start()
                processes.append(p)

            # Ждем завершения всех процессов
            for p in processes:
                p.join()

            # --- Сбор данных из очереди ---
            samples_generated = 0
            while not data_queue.empty():
                sample = data_queue.get_nowait()
                replay_buffer.append(sample)
                samples_generated += 1

            total_traversals += TRAINING_BLOCK_SIZE
            collection_time = time.time() - start_time
            
            print(f"Data collection finished. Buffer size: {len(replay_buffer)}. Samples generated: {samples_generated}", flush=True)

            if len(replay_buffer) < BATCH_SIZE:
                print("Buffer too small, skipping training.", flush=True)
                continue

            # --- Обучение ---
            indices = np.random.choice(len(replay_buffer), BATCH_SIZE, replace=False)
            minibatch = [replay_buffer[i] for i in indices]
            infosets, regrets, _ = zip(*minibatch)
            
            infosets_tensor = torch.tensor(infosets, dtype=torch.float32).to(device)
            targets_tensor = torch.tensor(regrets, dtype=torch.float32).to(device)

            model.train()
            optimizer.zero_grad()
            predictions = model(infosets_tensor)
            loss = criterion(predictions, targets_tensor)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            samples_per_sec = samples_generated / collection_time if collection_time > 0 else 0
            
            print(f"Block {block_counter} | Total: {total_traversals} | Loss: {loss.item():.6f} | Throughput: {samples_per_sec:.2f} samples/s", flush=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("\nTest finished.")

if __name__ == "__main__":
    main()
