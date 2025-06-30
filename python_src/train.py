# D2CFR-main/python_src/train.py (ВЕРСИЯ 7.0 - BATCH INFERENCE)

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
from collections import deque

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
from ofc_engine import DeepMCCFR, SharedReplayBuffer, InferenceQueue

# --- ГИПЕРПАРАМЕТРЫ ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 4
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 2000000
BATCH_SIZE = 256
SAVE_INTERVAL_SECONDS = 30 
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"

# Параметры для пакетного инференса
INFERENCE_BATCH_SIZE = 256
INFERENCE_MAX_DELAY_MS = 5

class InferenceWorker(threading.Thread):
    def __init__(self, model, queue, device):
        super().__init__()
        self.model = model
        self.queue = queue
        self.device = device
        self.stop_event = threading.Event()

    def run(self):
        print("InferenceWorker started.", flush=True)
        while not self.stop_event.is_set():
            self.queue.wait() # Ждем, пока в очереди не появится хотя бы один запрос
            requests = self.queue.pop_all()
            
            if not requests:
                continue

            # Собираем инфосеты в один батч
            infosets = [req.infoset for req in requests]
            tensor = torch.tensor(infosets, dtype=torch.float32).to(self.device)

            # Делаем один вызов модели
            with torch.no_grad():
                results_tensor = self.model(tensor)
            
            results_list = results_tensor.cpu().numpy()

            # Распределяем результаты по "обещаниям"
            for i, req in enumerate(requests):
                # Обрезаем результат до нужного количества действий
                result = results_list[i][:req.num_actions].tolist()
                req.set_result(result)
        print("InferenceWorker stopped.", flush=True)

    def stop(self):
        self.stop_event.set()

def main():
    device = torch.device("cpu")
    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    # ... (логика загрузки весов) ...

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- Квантование модели ---
    print("Quantizing model...", flush=True)
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # --- Создание общих ресурсов ---
    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)
    inference_queue = InferenceQueue()

    # --- Запуск потока для инференса ---
    inference_worker = InferenceWorker(quantized_model, inference_queue, device)
    inference_worker.start()

    # --- Запуск C++ воркеров ---
    # Они не загружают модель, а получают указатель на очередь
    solvers = [DeepMCCFR(ACTION_LIMIT, replay_buffer, inference_queue) for _ in range(NUM_WORKERS)]
    
    total_samples = 0
    block_counter = 0
    
    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Запускаем воркеры в бесконечном цикле
            run_futures = {executor.submit(s.run_traversal) for s in solvers}
            
            while True:
                block_counter += 1
                print(f"\n--- Block {block_counter} ---", flush=True)
                
                # Просто ждем некоторое время, пока данные копятся
                time.sleep(10) 
                
                current_samples = replay_buffer.get_count()
                samples_generated = current_samples - total_samples
                total_samples = current_samples
                
                samples_per_sec = samples_generated / 10.0
                print(f"Throughput: {samples_per_sec:.2f} samples/s. Buffer size: {total_samples}", flush=True)

                if total_samples < BATCH_SIZE:
                    continue

                # --- Шаг обучения ---
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
                
                print(f"Training step complete. Loss: {loss.item():.6f}", flush=True)
                
                # Обновляем модель для инференс-воркера
                model.eval()
                inference_worker.model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping workers...")
        inference_worker.stop()
        # Здесь executor сам завершит потоки
        inference_worker.join()
        print("Workers stopped.")
        # ... (финальное сохранение) ...

if __name__ == "__main__":
    main()
