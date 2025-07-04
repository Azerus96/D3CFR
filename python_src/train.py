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
import subprocess

# --- ИЗМЕНЕНИЕ: Импорты и проверка доступности XLA (TPU) ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    print("Warning: torch_xla not found. Running on CPU. For TPU support, install torch_xla.")

# --- НАСТРОЙКИ ---
# Для Colab TPU оптимально использовать все доступные ядра
NUM_WORKERS = int(os.cpu_count() or 96) 
# Настройки потоков для библиотек, чтобы избежать конфликтов
NUM_COMPUTATION_THREADS = "8" 
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
ACTION_LIMIT = 1000
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 8192
SAVE_INTERVAL_SECONDS = 300
MODEL_PATH = "d2cfr_model.pth"

# Параметры для пакетного инференса
INFERENCE_BATCH_SIZE = 2048
INFERENCE_MAX_DELAY_MS = 5 # Немного увеличим задержку для TPU

class InferenceWorker(threading.Thread):
    def __init__(self, model, queue, device):
        super().__init__(daemon=True)
        self.queue = queue
        self.device = device
        self.stop_event = threading.Event()
        self.model_lock = threading.Lock()
        self.set_model(model)

    def set_model(self, model):
        with self.model_lock:
            # Квантование может не поддерживаться XLA, поэтому используем .eval()
            self.model = model.eval()

    def run(self):
        print(f"InferenceWorker (ThreadID: {threading.get_ident()}) started on device {self.device}.", flush=True)
        
        while not self.stop_event.is_set():
            try:
                # --- ИЗМЕНЕНИЕ: Неблокирующая логика опроса очереди ---
                # 1. Забираем все, что есть в очереди, без ожидания
                requests = self.queue.pop_all()
                
                # 2. Если очередь пуста, немного ждем, чтобы не сжигать CPU, и переходим к следующей итерации
                if not requests:
                    time.sleep(0.001)  # 1ms пауза
                    continue
                
                # 3. Если запросы есть, обрабатываем их
                self.process_batch(requests)

            except Exception as e:
                print(f"Error in InferenceWorker: {e}", flush=True)
                traceback.print_exc()
        
        # Обработка оставшихся запросов после сигнала остановки
        final_requests = self.queue.pop_all()
        if final_requests:
            self.process_batch(final_requests)

        print(f"InferenceWorker (ThreadID: {threading.get_ident()}) stopped.", flush=True)

    def process_batch(self, requests):
        if not requests:
            return
            
        infosets = [req.infoset for req in requests]
        tensor = torch.tensor(infosets, dtype=torch.float32).to(self.device)

        with self.model_lock, torch.no_grad():
            results_tensor = self.model(tensor)
        
        results_list = results_tensor.cpu().numpy()

        for i, req in enumerate(requests):
            result = results_list[i][:req.num_actions].tolist()
            req.set_result(result)

    def stop(self):
        self.stop_event.set()

def push_to_github(model_path, commit_message):
    try:
        print("Pushing progress to GitHub...", flush=True)
        subprocess.run(['git', 'config', '--global', 'user.email', 'bot@example.com'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'Training Bot'], check=True)
        subprocess.run(['git', 'add', model_path], check=True)
        subprocess.run(['git', 'commit', '--allow-empty', '-m', commit_message], check=True)
        subprocess.run(['git', 'push'], check=True)
        print("Progress pushed successfully.", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to GitHub: {e}", flush=True)
    except Exception as e:
        print(f"An unexpected error occurred during git push: {e}", flush=True)

def main():
    # --- ИЗМЕНЕНИЕ: Автоматический выбор устройства TPU или CPU ---
    if XLA_AVAILABLE:
        device = xm.xla_device()
        print(f"TPU device found: {device}", flush=True)
    else:
        device = torch.device("cpu")
        print("TPU not found, using CPU.", flush=True)

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at {MODEL_PATH}. Loading weights...", flush=True)
        try:
            # При загрузке на XLA устройство, сначала грузим на CPU, а потом перемещаем
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.to(device)
        except Exception as e:
            print(f"Could not load state_dict. Error: {e}. Starting from scratch.", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)
    inference_queue = InferenceQueue()

    inference_worker = InferenceWorker(model, inference_queue, device)
    inference_worker.start()

    solvers = [DeepMCCFR(ACTION_LIMIT, replay_buffer, inference_queue) for _ in range(NUM_WORKERS)]
    
    stop_event = threading.Event()
    git_thread = None

    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            def worker_loop(solver):
                while not stop_event.is_set():
                    solver.run_traversal()

            print(f"Submitting {NUM_WORKERS} long-running C++ worker tasks...", flush=True)
            futures = {executor.submit(worker_loop, s) for s in solvers}
            
            last_save_time = time.time()
            last_report_time = time.time()
            loss = None
            last_report_head = 0
            last_train_head = 0

            while True:
                time.sleep(0.01) 
                
                current_head = replay_buffer.get_head()
                current_buffer_size = replay_buffer.get_count()
                
                if current_head >= last_train_head + BATCH_SIZE:
                    if current_buffer_size >= BATCH_SIZE:
                        model.train()
                        infosets_np, targets_np = replay_buffer.sample(BATCH_SIZE)
                        
                        infosets = torch.from_numpy(infosets_np).to(device)
                        targets = torch.from_numpy(targets_np).to(device)

                        optimizer.zero_grad()
                        predictions = model(infosets)
                        loss = criterion(predictions, targets)
                        loss.backward()
                        
                        # --- ИЗМЕНЕНИЕ: Для XLA нужен специальный шаг оптимизатора ---
                        if XLA_AVAILABLE:
                            # xm.optimizer_step выполняет шаг градиентного спуска и синхронизацию с TPU
                            xm.optimizer_step(optimizer)
                        else:
                            # Для CPU/GPU используем стандартный шаг
                            clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                        
                        inference_worker.set_model(model)
                        last_train_head = current_head
                
                now = time.time()
                if now - last_report_time > 10.0:
                    duration = now - last_report_time
                    samples_generated_interval = current_head - last_report_head
                    last_report_head = current_head
                    samples_per_sec = samples_generated_interval / duration if duration > 0 else 0
                    
                    print(f"\n--- Stats Update ---", flush=True)
                    print(f"Throughput: {samples_per_sec:.2f} samples/s. Buffer: {current_buffer_size}/{REPLAY_BUFFER_CAPACITY}. Total generated: {current_head:,}", flush=True)
                    
                    if XLA_AVAILABLE:
                        print(f"XLA Metrics: {xm.get_metrics_as_str()}", flush=True)
                    
                    if loss is not None:
                        print(f"Last training loss: {loss.item():.6f}", flush=True)

                    last_report_time = now

                    if now - last_save_time > SAVE_INTERVAL_SECONDS:
                        if git_thread and git_thread.is_alive():
                            print("Previous Git push is still running. Skipping this save.", flush=True)
                        else:
                            if loss is not None:
                                print("\n--- Saving model and pushing to GitHub ---", flush=True)
                                # Для сохранения модели с TPU, ее нужно сначала переместить на CPU
                                model.cpu()
                                torch.save(model.state_dict(), MODEL_PATH)
                                model.to(device) # Возвращаем модель на TPU
                                
                                commit_message = f"Training checkpoint. Total samples: {current_head:,}. Loss: {loss.item():.6f}"
                                
                                git_thread = threading.Thread(target=push_to_github, args=(MODEL_PATH, commit_message))
                                git_thread.start()
                                
                                last_save_time = now

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
    finally:
        print("Stopping workers...")
        stop_event.set()
        inference_worker.stop()
        
        if git_thread and git_thread.is_alive():
            print("Waiting for the final Git push to complete...")
            git_thread.join()

        print("\n--- Final Save ---", flush=True)
        model.cpu()
        torch.save(model.state_dict(), "d2cfr_model_final.pth")
        print("Final model saved. Exiting.")

if __name__ == "__main__":
    main()
