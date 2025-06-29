# D2CFR-main/python_src/train.py (ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ РЕШАЮЩЕГО ТЕСТА)

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import concurrent.futures
import traceback
import cProfile, pstats

# --- НАСТРОЙКИ ДЛЯ ТЕСТА "МАКСИМАЛЬНОЙ ИЗОЛЯЦИИ" ---
# Мы хотим понять, как работает сам алгоритм, исключив влияние многопоточности.
# Поэтому каждый из 48 воркеров будет строго однопоточным.
NUM_COMPUTATION_THREADS = "1"
os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))

from .model import DuelingNetwork
from ofc_engine import DeepMCCFR, SharedReplayBuffer

# --- ГИПЕРПАРАМЕТРЫ ДЛЯ БЫСТРОГО ТЕСТА ---
# Цель - заставить C++ код выполниться и выдать логи профилирования.
INPUT_SIZE = 1486 
ACTION_LIMIT = 4  # <-- КРИТИЧЕСКИ ВАЖНО: Минимальное значение для ускорения C++
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 500000
BATCH_SIZE = 256
TRAINING_BLOCK_SIZE = 48 # <-- Даем каждому воркеру ровно 1 задачу (48/48=1)
SAVE_INTERVAL_BLOCKS = 5 
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"
NUM_WORKERS = 48 # <-- Используем вашу идею с большим количеством воркеров

def run_training_loop():
    device = torch.device("cpu")
    print(f"Using device for PyTorch: {device}", flush=True)
    print(f"Using {NUM_WORKERS} worker threads for data collection.", flush=True)
    print(f"Each library (Torch, OpenMP, etc.) is limited to {NUM_COMPUTATION_THREADS} threads.", flush=True)

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    print("Starting training from scratch for this profiling run.", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("\nConverting model to TorchScript for C++...", flush=True)
    model.eval()
    example_input = torch.randn(1, INPUT_SIZE).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
    print(f"TorchScript model saved to {TORCHSCRIPT_MODEL_PATH}", flush=True)

    print(f"\nCreating in-memory replay buffer with capacity {REPLAY_BUFFER_CAPACITY} and action_limit {ACTION_LIMIT}", flush=True)
    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)
    print("Buffer created.", flush=True)

    solvers = [DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, replay_buffer) for _ in range(NUM_WORKERS)]
    print(f"{len(solvers)} solver instances created and linked to the buffer.", flush=True)

    total_traversals = 0
    # Запустим всего 2 блока для сбора статистики
    for block_counter in range(1, 3):
        start_time = time.time()
        
        print(f"\n--- Block {block_counter} ---", flush=True)
        
        tasks_per_worker = TRAINING_BLOCK_SIZE // NUM_WORKERS
        print(f"Submitting {tasks_per_worker * NUM_WORKERS} traversal tasks ({tasks_per_worker} each)...", flush=True)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(solver.run_traversal) for solver in solvers for _ in range(tasks_per_worker)]
            concurrent.futures.wait(futures)

        total_traversals += len(futures)
        buffer_size = replay_buffer.get_count()
        
        print(f"Data collection finished. Buffer size: {buffer_size}", flush=True)

        if buffer_size < BATCH_SIZE:
            print("Buffer too small, skipping training.", flush=True)
            continue

        model.train()
        infosets_np, targets_np = replay_buffer.sample(BATCH_SIZE)
        infosets = torch.from_numpy(infosets_np).to(device)
        targets = torch.from_numpy(targets_np).to(device)

        optimizer.zero_grad()
        predictions = model(infosets)
        loss = criterion(predictions, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        duration = time.time() - start_time
        traversals_per_sec = len(futures) / duration if duration > 0 else float('inf')
        
        print(f"Block {block_counter} | Loss: {loss.item():.6f} | Speed: {traversals_per_sec:.2f} trav/s", flush=True)

def main():
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        run_training_loop()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        profiler.disable()
        print("\n" + "="*40)
        print("PYTHON PROFILING RESULTS (CUMULATIVE TIME)")
        print("="*40)
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(20)
        print("="*40)

if __name__ == "__main__":
    main()
