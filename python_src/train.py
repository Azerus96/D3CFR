# D2CFR-main/python_src/train.py (ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ)

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

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
ACTION_LIMIT = 4
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 2000000
BATCH_SIZE = 256
# Параметры цикла
DATA_COLLECTION_SECONDS = 15 # Собираем данные 15 секунд
TRAINING_STEPS_PER_BLOCK = 5 # Делаем 5 шагов обучения после сбора
SAVE_INTERVAL_BLOCKS = 1000 

MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"

def worker_run(solver, stop_event):
    """Функция, которую выполняет каждый C++ воркер."""
    while not stop_event.is_set():
        try:
            solver.run_traversal()
        except Exception as e:
            print(f"Error in worker thread: {e}")
            # Если воркер упал, он просто перестанет работать
            break

def main():
    device = torch.device("cpu")
    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
    # ... (логика загрузки весов, как раньше) ...
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at {MODEL_PATH}. Attempting to transfer weights...")
        try:
            # ... (код загрузки весов) ...
        except Exception as e:
            print(f"Could not perform weight transfer: {e}. Starting from scratch.", flush=True)
    else:
        print(f"No model found at {MODEL_PATH}. Starting training from scratch.", flush=True)


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)
    
    block_counter = 0
    try:
        while True:
            block_counter += 1
            print(f"\n--- Block {block_counter} ---", flush=True)

            # --- ЭТАП 1: Подготовка и сбор данных ---
            print("Updating and saving model for workers...", flush=True)
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            traced_script_module = torch.jit.trace(quantized_model, torch.randn(1, INPUT_SIZE))
            traced_script_module.save(TORCHSCRIPT_MODEL_PATH)

            # Создаем новые солверы, которые загрузят обновленную модель
            solvers = [DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, replay_buffer) for _ in range(NUM_WORKERS)]
            
            print(f"Starting data collection for {DATA_COLLECTION_SECONDS} seconds...", flush=True)
            stop_event = threading.Event()
            initial_buffer_size = replay_buffer.get_count()
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Запускаем воркеры
                futures = [executor.submit(worker_run, solver, stop_event) for solver in solvers]
                # Ждем нужное время
                time.sleep(DATA_COLLECTION_SECONDS)
                # Посылаем сигнал остановки
                stop_event.set()
            
            duration = time.time() - start_time
            final_buffer_size = replay_buffer.get_count()
            samples_generated = final_buffer_size - initial_buffer_size
            samples_per_sec = samples_generated / duration if duration > 0 else 0
            
            print(f"Collection finished. Generated {samples_generated} samples. Throughput: {samples_per_sec:.2f} samples/s. Buffer size: {final_buffer_size}", flush=True)

            # --- ЭТАП 2: Обучение ---
            if final_buffer_size < BATCH_SIZE:
                print("Buffer too small, skipping training.", flush=True)
                continue
            
            print(f"Performing {TRAINING_STEPS_PER_BLOCK} training steps...", flush=True)
            model.train()
            total_loss = 0
            for _ in range(TRAINING_STEPS_PER_BLOCK):
                infosets_np, targets_np = replay_buffer.sample(BATCH_SIZE)
                
                infosets = torch.from_numpy(infosets_np).to(device)
                targets = torch.from_numpy(targets_np).to(device)

                optimizer.zero_grad()
                predictions = model(infosets)
                loss = criterion(predictions, targets)
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / TRAINING_STEPS_PER_BLOCK
            print(f"Training finished. Average Loss: {avg_loss:.6f}", flush=True)

            # --- ЭТАП 3: Сохранение ---
            if block_counter % SAVE_INTERVAL_BLOCKS == 0:
                print("\n--- Saving model checkpoint ---", flush=True)
                torch.save(model.state_dict(), MODEL_PATH)
                print("Checkpoint saved.", flush=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("\n--- Final Save ---", flush=True)
        torch.save(model.state_dict(), "d2cfr_model_final.pth")
        print("Final model saved. Exiting.")

if __name__ == "__main__":
    main()
