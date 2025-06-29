# D2CFR-main/python_src/train.py

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
# Давайте используем конфигурацию, которая показала проблему
# 24 воркера, каждый использует 4 потока для вычислений
NUM_WORKERS = 24
NUM_COMPUTATION_THREADS = "4" 
os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))

# Импортируем ваши модули
from .model import DuelingNetwork
from ofc_engine import DeepMCCFR, SharedReplayBuffer

# --- ГИПЕРПАРАМЕТРЫ ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 4 # Используем значение из вашего первого теста
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 2000000
BATCH_SIZE = 256
TRAINING_BLOCK_SIZE = 48 # Количество полных обходов (p0+p1) за блок
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"

def run_training_loop():
    device = torch.device("cpu")
    print(f"Using device for PyTorch: {device}", flush=True)
    print(f"Using {NUM_WORKERS} worker threads for data collection.", flush=True)
    print(f"Each Torch/OMP instance is limited to {NUM_COMPUTATION_THREADS} threads.", flush=True)

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("\nConverting model to TorchScript for C++...", flush=True)
    model.eval()
    example_input = torch.randn(1, INPUT_SIZE).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
    print(f"TorchScript model saved to {TORCHSCRIPT_MODEL_PATH}", flush=True)

    print(f"\nCreating replay buffer with capacity {REPLAY_BUFFER_CAPACITY}", flush=True)
    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)
    
    solvers = [DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, replay_buffer) for _ in range(NUM_WORKERS)]
    print(f"{len(solvers)} C++ solver instances created.", flush=True)

    # Запустим всего 1 блок для чистоты эксперимента
    for block_counter in range(1, 2):
        start_time = time.time()
        
        print(f"\n--- Block {block_counter}: Data Collection ---", flush=True)
        
        # Каждый воркер выполнит это количество заданий
        tasks_per_worker = TRAINING_BLOCK_SIZE // NUM_WORKERS
        if tasks_per_worker == 0:
            print("Warning: TRAINING_BLOCK_SIZE is smaller than NUM_WORKERS. No tasks will be run.")
            continue

        print(f"Submitting {tasks_per_worker * NUM_WORKERS} profiling tasks ({tasks_per_worker} per worker)...", flush=True)
        
        all_cpp_stats = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Вызываем новую функцию run_traversal_for_profiling
            futures = [executor.submit(solver.run_traversal_for_profiling) for solver in solvers for _ in range(tasks_per_worker)]
            
            # Собираем результаты по мере их готовности
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result: # Убедимся, что результат не пустой
                        all_cpp_stats.append(result)
                except Exception as e:
                    print(f"A task failed: {e}")

        collection_duration = time.time() - start_time
        buffer_size = replay_buffer.get_count()
        
        print(f"Data collection finished in {collection_duration:.2f}s. Buffer size: {buffer_size}", flush=True)

        # --- Анализ результатов профилирования C++ ---
        if all_cpp_stats:
            avg_stats = np.mean(all_cpp_stats, axis=0)
            print("\n" + "="*40)
            print("C++ PROFILING RESULTS (AVERAGE PER TRAVERSE CALL, ms)")
            print("="*40)
            print(f"Avg traverse() total:      {avg_stats[0]:.4f} ms")
            print(f"  -> get_legal_actions():   {avg_stats[1]:.4f} ms")
            print(f"  -> featurize():           {avg_stats[2]:.4f} ms")
            print(f"  -> model_inference():     {avg_stats[3]:.4f} ms")
            print(f"  -> buffer->push():        {avg_stats[4]:.4f} ms")
            total_measured = sum(avg_stats[1:])
            unaccounted = avg_stats[0] - total_measured
            print(f"  -> Other (recursion etc): {unaccounted:.4f} ms")
            print("="*40 + "\n", flush=True)
        else:
            print("No C++ profiling stats were collected.", flush=True)

        if buffer_size < BATCH_SIZE:
            print("Buffer too small, skipping training.", flush=True)
            continue

        # --- Фаза обучения (остается без изменений) ---
        print(f"--- Block {block_counter}: Training ---", flush=True)
        train_start_time = time.time()
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
        train_duration = time.time() - train_start_time
        
        total_duration = time.time() - start_time
        traversals_per_sec = len(futures) / total_duration if total_duration > 0 else float('inf')
        
        print(f"Training finished in {train_duration:.2f}s.")
        print(f"Block {block_counter} | Loss: {loss.item():.6f} | Total Speed: {traversals_per_sec:.2f} trav/s", flush=True)

def main():
    try:
        run_training_loop()
    except Exception as e:
        print(f"\nAn error occurred in the main loop: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
