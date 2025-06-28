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

from .model import DuelingNetwork
# ИЗМЕНЕНО: ReplayBuffer больше не нужен, импортируем новые классы из C++
from ofc_engine import DeepMCCFR, SharedReplayBuffer, cleanup_shared_memory

# --- HYPERPARAMETERS ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 24 
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 2000000 # Вместимость буфера
BATCH_SIZE = 2048
TRAINING_BLOCK_SIZE = 24
SAVE_INTERVAL_BLOCKS = 5 
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"
NUM_WORKERS = 24
# ИМЯ ДЛЯ СЕГМЕНТА ОБЩЕЙ ПАМЯТИ
SHARED_MEMORY_NAME = "d2cfr_replay_buffer"

def main():
    # Устанавливаем device (GPU, если доступен)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for PyTorch: {device}", flush=True)
    print(f"Using {NUM_WORKERS} worker threads for data collection on CPU.", flush=True)

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # ИЗМЕНЕНО: Удаляем старый ReplayBuffer

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}", flush=True)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.", flush=True)
        except Exception as e:
            print(f"Could not load model: {e}. Starting from scratch.", flush=True)

    # --- Конвертация модели в TorchScript ---
    print("Converting model to TorchScript for C++...", flush=True)
    model.eval()
    example_input = torch.randn(1, INPUT_SIZE).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
    print(f"TorchScript model saved to {TORCHSCRIPT_MODEL_PATH}", flush=True)

    # --- Создание общего буфера и C++ солверов ---
    # Очищаем старый сегмент памяти, если он остался от предыдущего запуска
    try:
        cleanup_shared_memory(SHARED_MEMORY_NAME)
        print("Cleaned up previous shared memory segment.", flush=True)
    except Exception:
        pass # Ничего страшного, если его не было

    print(f"Creating shared replay buffer '{SHARED_MEMORY_NAME}' with capacity {REPLAY_BUFFER_CAPACITY}", flush=True)
    replay_buffer = SharedReplayBuffer(SHARED_MEMORY_NAME, REPLAY_BUFFER_CAPACITY)
    print("Buffer created.", flush=True)

    solvers = [DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, replay_buffer) for _ in range(NUM_WORKERS)]
    print("Solver instances created and linked to the shared buffer.", flush=True)

    total_traversals = 0
    block_counter = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            while True:
                block_counter += 1
                start_time = time.time()
                
                print(f"Submitting {TRAINING_BLOCK_SIZE} traversal tasks to {NUM_WORKERS} workers...", flush=True)
                
                # ИЗМЕНЕНО: Задачи теперь ничего не возвращают
                futures = [executor.submit(solvers[i % NUM_WORKERS].run_traversal) for i in range(TRAINING_BLOCK_SIZE)]
                
                # Просто ждем, пока все воркеры закончат писать в буфер
                concurrent.futures.wait(futures)

                total_traversals += TRAINING_BLOCK_SIZE
                buffer_size = replay_buffer.get_count()
                
                print(f"Data collection finished. Buffer size: {buffer_size}", flush=True)

                if buffer_size < BATCH_SIZE:
                    print(f"Block {block_counter} | Total Traversals: {total_traversals} | Buffer size {buffer_size} is too small, skipping training.", flush=True)
                    continue

                # --- Training Phase ---
                model.train()
                
                # ИЗМЕНЕНО: Сэмплируем напрямую из C++ буфера в NumPy массивы
                infosets_np, targets_np = replay_buffer.sample(BATCH_SIZE)
                
                # Создаем тензоры из NumPy БЕЗ КОПИРОВАНИЯ
                infosets = torch.from_numpy(infosets_np).to(device)
                targets = torch.from_numpy(targets_np).to(device)

                optimizer.zero_grad()
                
                predictions = model(infosets)
                
                # Маска больше не нужна, так как C++ уже заполняет нулями
                loss = criterion(predictions, targets)
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                duration = time.time() - start_time
                traversals_per_sec = TRAINING_BLOCK_SIZE / duration if duration > 0 else float('inf')
                
                print(f"Block {block_counter} | Total: {total_traversals:,} | Loss: {loss.item():.6f} | Speed: {traversals_per_sec:.2f} trav/s", flush=True)

                if block_counter % SAVE_INTERVAL_BLOCKS == 0:
                    # ... (логика сохранения и пуша в Git остается без изменений)
                    print("-" * 100, flush=True)
                    print(f"Saving models at traversal {total_traversals:,}...", flush=True)
                    torch.save(model.state_dict(), MODEL_PATH)
                    model.eval()
                    traced_script_module = torch.jit.trace(model, example_input)
                    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
                    print("Models saved successfully.", flush=True)
                    print("Pushing progress to GitHub...", flush=True)
                    os.system(f'git add {MODEL_PATH} {TORCHSCRIPT_MODEL_PATH}')
                    os.system(f'git commit -m "Training checkpoint after {total_traversals} traversals"')
                    os.system('git push')
                    print("Progress pushed successfully.", flush=True)
                    print("-" * 100, flush=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final models...", flush=True)
        torch.save(model.state_dict(), MODEL_PATH)
        model.eval()
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
        print("Models saved. Exiting.", flush=True)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", flush=True)
        traceback.print_exc()
        torch.save(model.state_dict(), "d2cfr_model_error.pth")
        print("Saved an emergency copy of the model.", flush=True)
    finally:
        # ОБЯЗАТЕЛЬНО: Очищаем общую память при выходе
        print("Cleaning up shared memory segment...", flush=True)
        cleanup_shared_memory(SHARED_MEMORY_NAME)
        print("Cleanup complete.", flush=True)

if __name__ == "__main__":
    main()
