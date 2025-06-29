# D2CFR-main/python_src/train.py (ВЕРСИЯ 6.0 - MULTIPROCESSING)

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

# --- ГЛОБАЛЬНЫЙ КОНТРОЛЬ НАД ПОТОКАМИ ---
# Эти переменные будут использоваться каждым дочерним процессом
NUM_COMPUTATION_THREADS = "1" 
os.environ['OMP_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = NUM_COMPUTATION_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_COMPUTATION_THREADS
torch.set_num_threads(int(NUM_COMPUTATION_THREADS))

# --- ГИПЕРПАРАМЕТРЫ ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 16
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 1000000
BATCH_SIZE = 4096
SAVE_INTERVAL_SECONDS = 600 
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"
# Используем на 2 меньше, чтобы оставить ресурсы для основного процесса и ОС
NUM_WORKERS = max(1, cpu_count() - 2)

# Поздний импорт, чтобы переменные окружения успели установиться
from .model import DuelingNetwork

def worker_process(model_path, action_limit, queue, build_path):
    """
    Эта функция выполняется в каждом отдельном процессе.
    """
    # Добавляем путь к скомпилированному модулю
    if build_path not in sys.path:
        sys.path.insert(0, build_path)
        
    try:
        from ofc_engine import DeepMCCFR
        # Каждый процесс создает свой собственный экземпляр движка
        solver = DeepMCCFR(model_path, action_limit, queue)
        while True:
            solver.run_traversal()
    except KeyboardInterrupt:
        # Просто выходим, если основной процесс прерван
        pass
    except Exception as e:
        print(f"Error in worker process (PID: {os.getpid()}): {e}")
        traceback.print_exc()

def main():
    device = torch.device("cpu")
    print(f"Main process (PID: {os.getpid()}) using device: {device}", flush=True)
    print(f"Starting {NUM_WORKERS} worker processes...", flush=True)

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at {MODEL_PATH}. Attempting to transfer weights...")
        try:
            old_state_dict = torch.load(MODEL_PATH, map_location=device)
            new_state_dict = model.state_dict()
            loaded_count = 0
            mismatched_layers = []
            for name, param in old_state_dict.items():
                if name in new_state_dict and new_state_dict[name].shape == param.shape:
                    new_state_dict[name].copy_(param)
                    loaded_count += 1
                else:
                    mismatched_layers.append(name)
            model.load_state_dict(new_state_dict)
            print(f"Weight transfer complete. Loaded {loaded_count} layers. Skipped: {mismatched_layers}")
        except Exception as e:
            print(f"Could not perform weight transfer: {e}. Starting from scratch.", flush=True)
    else:
        print(f"No model found at {MODEL_PATH}. Starting training from scratch.", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- Квантование и сохранение модели для воркеров ---
    def save_quantized_model(model_to_quantize, path):
        print("\nQuantizing and saving model for workers...", flush=True)
        model_to_quantize.to('cpu')
        model_to_quantize.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model_to_quantize, {torch.nn.Linear}, dtype=torch.qint8
        )
        traced_script_module = torch.jit.trace(quantized_model, torch.randn(1, INPUT_SIZE))
        traced_script_module.save(path)
        print(f"Quantized model saved to {path}", flush=True)

    save_quantized_model(model, TORCHSCRIPT_MODEL_PATH)

    data_queue = Queue()
    replay_buffer = deque(maxlen=REPLAY_BUFFER_CAPACITY)

    # --- Запуск процессов-воркеров ---
    build_path = os.path.abspath("./build")
    processes = []
    for _ in range(NUM_WORKERS):
        p = Process(target=worker_process, args=(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, data_queue, build_path))
        p.daemon = True # Процессы завершатся, если основной умрет
        p.start()
        processes.append(p)

    last_save_time = time.time()
    total_samples = 0
    block_counter = 0
    try:
        while True:
            block_counter += 1
            print(f"\n--- Block {block_counter} ---", flush=True)
            
            # --- Сбор данных ---
            start_collection_time = time.time()
            
            # Собираем данные, пока не наберем достаточно для батча
            while data_queue.qsize() < BATCH_SIZE and len(replay_buffer) < BATCH_SIZE:
                time.sleep(0.5)
                print(f"Waiting for data... Queue size: {data_queue.qsize()}", flush=True)

            samples_in_block = 0
            while not data_queue.empty():
                sample = data_queue.get()
                replay_buffer.append(sample)
                samples_in_block += 1
            
            total_samples += samples_in_block
            collection_time = time.time() - start_collection_time
            samples_per_sec = samples_in_block / collection_time if collection_time > 0 else float('inf')
            
            print(f"Collected {samples_in_block} new samples. Throughput: {samples_per_sec:.2f} samples/s. Buffer size: {len(replay_buffer)}", flush=True)

            # --- Обучение ---
            if len(replay_buffer) < BATCH_SIZE:
                continue

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
            
            print(f"Training step complete. Loss: {loss.item():.6f}", flush=True)

            # --- Сохранение ---
            if time.time() - last_save_time > SAVE_INTERVAL_SECONDS:
                print("\n--- Saving models ---", flush=True)
                torch.save(model.state_dict(), MODEL_PATH)
                save_quantized_model(model, TORCHSCRIPT_MODEL_PATH)
                last_save_time = time.time()
                print("--- Models saved ---", flush=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Terminating worker processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
        print("Saving final models...")
        torch.save(model.state_dict(), "d2cfr_model_final.pth")
        save_quantized_model(model, "d2cfr_model_script_final.pt")
        print("Training finished.", flush=True)

if __name__ == "__main__":
    main()
