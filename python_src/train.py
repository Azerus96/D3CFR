# D2CFR-main/python_src/train.py (ФИНАЛЬНАЯ ВЕРСИЯ ДЛЯ ТЕСТА)

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import concurrent.futures
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- НАСТРОЙКИ ---
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

# --- ГИПЕРПАРАМЕТРЫ ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 12 # Устанавливаем разумный лимит для теста
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 2000000
BATCH_SIZE = 4096
TRAINING_BLOCK_SIZE = 240 # 10 задач на воркера
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"

def run_final_test():
    """Запускает один блок сбора данных и измеряет производительность."""
    print("--- FINAL PERFORMANCE TEST ---")
    print(f"ACTION_LIMIT = {ACTION_LIMIT}")
    print(f"NUM_WORKERS = {NUM_WORKERS}, THREADS_PER_WORKER = {NUM_COMPUTATION_THREADS}")

    # --- Инициализация ---
    device = torch.device("cpu")
    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
    print("\nConverting model to TorchScript for C++...", flush=True)
    model.eval()
    example_input = torch.randn(1, INPUT_SIZE).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
    print(f"TorchScript model saved to {TORCHSCRIPT_MODEL_PATH}", flush=True)

    print(f"\nCreating replay buffer with capacity {REPLAY_BUFFER_CAPACITY}", flush=True)
    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)
    
    solvers = [DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, replay_buffer) for _ in range(NUM_WORKERS)]
    
    print(f"\nStarting data collection block ({TRAINING_BLOCK_SIZE} traversals)...", flush=True)
    start_time = time.time()
    initial_buffer_size = replay_buffer.get_count()
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        tasks_per_worker = TRAINING_BLOCK_SIZE // NUM_WORKERS
        futures = [executor.submit(solver.run_traversal) for solver in solvers for _ in range(tasks_per_worker)]
        concurrent.futures.wait(futures)
        
        # Проверка на ошибки
        for future in futures:
            if future.exception():
                print(f"Worker thread failed: {future.exception()}")
                traceback.print_exc()


    end_time = time.time()
    final_buffer_size = replay_buffer.get_count()

    # --- Результаты ---
    duration = end_time - start_time
    total_traversals = len(futures)
    total_samples = final_buffer_size - initial_buffer_size
    
    trav_per_sec = total_traversals / duration if duration > 0 else 0
    samples_per_sec = total_samples / duration if duration > 0 else 0
    time_per_sample_ms = (duration / total_samples) * 1000 if total_samples > 0 else 0

    print("\n" + "="*40)
    print("FINAL PERFORMANCE METRICS")
    print("="*40)
    print(f"Total time: {duration:.2f} s")
    print(f"Total traversals (run_traversal calls): {total_traversals}")
    print(f"Total samples generated: {total_samples}")
    print("-" * 40)
    print(f"Speed: {trav_per_sec:.2f} traversals/sec")
    print(f"Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"Time per sample: {time_per_sample_ms:.4f} ms")
    print("="*40)

if __name__ == "__main__":
    run_final_test()
