# D2CFR-main/python_src/train.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import concurrent.futures
import traceback

# --- ГЛОБАЛЬНЫЙ КОНТРОЛЬ НАД ПОТОКАМИ ---
# Это САМАЯ ВАЖНАЯ ЧАСТЬ для стабильной работы на многоядерных CPU.
# Мы ограничиваем количество потоков, которые могут использовать
# PyTorch, NumPy и OpenMP (в вашем C++ коде).
# 8 - это безопасное и эффективное значение для начала.
# Оно оставляет много ядер для параллельной работы воркеров.
NUM_COMPUTATION_THREADS = "8"
os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))
# --- КОНЕЦ БЛОКА КОНТРОЛЯ ПОТОКОВ ---


from .model import DuelingNetwork
from ofc_engine import DeepMCCFR, SharedReplayBuffer

# --- HYPERPARAMETERS ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 24 
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 2000000
BATCH_SIZE = 2048
TRAINING_BLOCK_SIZE = 96 # Можно увеличить, чтобы лучше утилизировать ядра
SAVE_INTERVAL_BLOCKS = 5 
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"
# Оптимальное количество воркеров - это не os.cpu_count()!
# Начнем с безопасного значения. Можно будет увеличить до 16, 24, 32...
NUM_WORKERS = 16 

def main():
    # torch.set_num_threads(1) -> УДАЛЕНО, так как мы управляем этим глобально.
    device = torch.device("cpu") # Принудительно используем CPU
    print(f"Using device for PyTorch: {device}", flush=True)
    print(f"Using {NUM_WORKERS} worker threads for data collection.", flush=True)
    print(f"Each library (Torch, OpenMP, etc.) is limited to {NUM_COMPUTATION_THREADS} threads.", flush=True)

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}", flush=True)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.", flush=True)
        except Exception as e:
            print(f"Could not load model: {e}. Starting from scratch.", flush=True)

    print("Converting model to TorchScript for C++...", flush=True)
    model.eval()
    example_input = torch.randn(1, INPUT_SIZE).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
    print(f"TorchScript model saved to {TORCHSCRIPT_MODEL_PATH}", flush=True)

    print(f"Creating in-memory replay buffer with capacity {REPLAY_BUFFER_CAPACITY}", flush=True)
    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY)
    print("Buffer created.", flush=True)

    # Создаем пул солверов. Они все будут писать в один и тот же буфер.
    solvers = [DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, replay_buffer) for _ in range(NUM_WORKERS)]
    print(f"{len(solvers)} solver instances created and linked to the buffer.", flush=True)

    total_traversals = 0
    block_counter = 0
    try:
        # Используем ThreadPoolExecutor для параллельного запуска C++ воркеров
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            while True:
                block_counter += 1
                start_time = time.time()
                
                print(f"\n--- Block {block_counter} ---", flush=True)
                print(f"Submitting {TRAINING_BLOCK_SIZE} traversal tasks to {NUM_WORKERS} workers...", flush=True)
                
                # Запускаем run_traversal в параллельных потоках
                futures = [executor.submit(solver.run_traversal) for solver in solvers for _ in range(TRAINING_BLOCK_SIZE // NUM_WORKERS)]
                
                # Ждем завершения всех задач по генерации данных
                for future in concurrent.futures.as_completed(futures):
                    # Проверяем, не было ли ошибок в воркерах
                    if future.exception() is not None:
                        print(f"!!! An error occurred in a worker thread: {future.exception()}", flush=True)
                        traceback.print_exc()

                total_traversals += len(futures)
                buffer_size = replay_buffer.get_count()
                
                print(f"Data collection finished. Buffer size: {buffer_size}", flush=True)

                if buffer_size < BATCH_SIZE:
                    print(f"Buffer size {buffer_size} is too small, skipping training. Need {BATCH_SIZE}.", flush=True)
                    time.sleep(5) # Ждем немного, чтобы не спамить логами
                    continue

                # --- Тренировочный шаг ---
                model.train()
                
                # Получаем батч из C++ буфера
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
                
                print(f"Block {block_counter} | Total: {total_traversals:,} | Loss: {loss.item():.6f} | Speed: {traversals_per_sec:.2f} trav/s", flush=True)

                if block_counter % SAVE_INTERVAL_BLOCKS == 0:
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
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", flush=True)
        traceback.print_exc()
    finally:
        print("Saving final models...", flush=True)
        torch.save(model.state_dict(), "d2cfr_model_final.pth")
        model.eval()
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save("d2cfr_model_script_final.pt")
        print("Models saved. Training finished.", flush=True)


if __name__ == "__main__":
    main()
