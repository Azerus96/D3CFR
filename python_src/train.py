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

# --- НАСТРОЙКИ (ЭКСПЕРИМЕНТ №1) ---
# Проверяем гипотезу о конкуренции потоков
# 1 воркер на 1 ядро, каждый воркер использует 1 поток для вычислений
NUM_WORKERS = 96 
NUM_COMPUTATION_THREADS = "1" 
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
ACTION_LIMIT = 4 
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 2000000
BATCH_SIZE = 4096 # Увеличим, чтобы обучение не было узким местом
TRAINING_BLOCK_SIZE = 96 # Адаптируем под количество воркеров
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"

def run_training_loop():
    device = torch.device("cpu")
    print(f"Using device for PyTorch: {device}", flush=True)
    print(f"--- EXPERIMENT 1: TESTING THREAD CONTENTION ---", flush=True)
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
        
        tasks_per_worker = TRAINING_BLOCK_SIZE // NUM_WORKERS
        if tasks_per_worker == 0:
            tasks_per_worker = 1 # Гарантируем, что хотя бы 1 задача на воркер
            print(f"Warning: TRAINING_BLOCK_SIZE < NUM_WORKERS. Setting tasks_per_worker to 1.")

        print(f"Submitting {tasks_per_worker * NUM_WORKERS} profiling tasks ({tasks_per_worker} per worker)...", flush=True)
        
        all_cpp_stats = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(solver.run_traversal_for_profiling) for solver in solvers for _ in range(tasks_per_worker)]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        all_cpp_stats.append(result)
                except Exception as e:
                    print(f"A task failed: {e}")

        collection_duration = time.time() - start_time
        buffer_size = replay_buffer.get_count()
        
        print(f"Data collection finished in {collection_duration:.2f}s. Buffer size: {buffer_size}", flush=True)

        if all_cpp_stats:
            avg_stats = np.mean(all_cpp_stats, axis=0)
            total_time = avg_stats[0]
            other_time = total_time - sum(avg_stats[1:])
            print("\n" + "="*40)
            print("C++ PROFILING RESULTS (AVERAGE PER TRAVERSE CALL, ms)")
            print("="*40)
            print(f"Avg traverse() total:      {total_time:.4f} ms")
            print(f"  -> get_legal_actions():   {avg_stats[1]:.4f} ms ({avg_stats[1]/total_time:.1%})")
            print(f"  -> featurize():           {avg_stats[2]:.4f} ms ({avg_stats[2]/total_time:.1%})")
            print(f"  -> model_inference():     {avg_stats[3]:.4f} ms ({avg_stats[3]/total_time:.1%})")
            print(f"  -> buffer->push():        {avg_stats[4]:.4f} ms ({avg_stats[4]/total_time:.1%})")
            print(f"  -> Other (recursion etc): {other_time:.4f} ms ({other_time/total_time:.1%})")
            print("="*40 + "\n", flush=True)
        else:
            print("No C++ profiling stats were collected.", flush=True)

        if buffer_size < BATCH_SIZE:
            print("Buffer too small, skipping training.", flush=True)
            continue

        print(f"--- Block {block_counter}: Training ---", flush=True)
        train_start_time = time.time()
        model.train()
        infosets_np, targets_np = replay_buffer.sample(BATCH_SIZE)
        if infosets_np.shape[0] > 0:
            infosets = torch.from_numpy(infosets_np).clone().to(device)
            targets = torch.from_numpy(targets_np).clone().to(device)

            optimizer.zero_grad()
            predictions = model(infosets)
            loss = criterion(predictions, targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_item = loss.item()
        else:
            loss_item = -1.0
            print("Sampled empty batch, skipping training step.")

        train_duration = time.time() - train_start_time
        
        total_duration = time.time() - start_time
        traversals_per_sec = len(futures) / total_duration if total_duration > 0 else float('inf')
        
        print(f"Training finished in {train_duration:.2f}s.")
        print(f"Block {block_counter} | Loss: {loss_item:.6f} | Total Speed: {traversals_per_sec:.2f} trav/s", flush=True)

def main():
    try:
        run_training_loop()
    except Exception as e:
        print(f"\nAn error occurred in the main loop: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
