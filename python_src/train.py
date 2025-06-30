# D2CFR-main/python_src/train.py (ВЕРСИЯ ДЛЯ ТЕСТА ПАКЕТНОГО ИНФЕРЕНСА)

import os
import time
import torch
import torch.nn as nn
import numpy as np
import concurrent.futures
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
REPLAY_BUFFER_CAPACITY = 2000000
# Запускаем сбор данных на фиксированное время
DATA_COLLECTION_SECONDS = 10 
# Размер батча для теста инференса
INFERENCE_BATCH_SIZE = 256

TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"

def main():
    device = torch.device("cpu")
    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
    print("--- BATCH INFERENCE PERFORMANCE TEST ---")
    
    # --- Квантование модели для максимальной скорости ---
    print("\nQuantizing model for faster CPU inference...", flush=True)
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("Model quantization complete.", flush=True)
    
    # Сохраняем неквантованную модель, так как C++ код ее все равно не использует
    traced_script_module = torch.jit.trace(model, torch.randn(1, INPUT_SIZE))
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)

    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)
    solvers = [DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, replay_buffer) for _ in range(NUM_WORKERS)]

    # --- ЭТАП 1: Измерение скорости генерации данных ---
    print(f"\n--- Step 1: Measuring data generation speed for {DATA_COLLECTION_SECONDS} seconds ---", flush=True)
    
    stop_event = threading.Event()
    
    def run_worker(solver):
        while not stop_event.is_set():
            solver.run_traversal()

    initial_buffer_size = replay_buffer.get_count()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_worker, solver) for solver in solvers]
        time.sleep(DATA_COLLECTION_SECONDS)
        stop_event.set() # Посылаем сигнал остановки всем воркерам
    
    final_buffer_size = replay_buffer.get_count()
    samples_generated = final_buffer_size - initial_buffer_size
    
    gen_throughput = samples_generated / DATA_COLLECTION_SECONDS
    time_per_gen_sample_ms = (1 / gen_throughput) * 1000 if gen_throughput > 0 else float('inf')

    print(f"Generated {samples_generated} samples in {DATA_COLLECTION_SECONDS}s.")
    print(f"Pure C++ Generation Throughput: {gen_throughput:.2f} samples/sec")
    print(f"Time per generated sample (T_generate): {time_per_gen_sample_ms:.6f} ms")

    # --- ЭТАП 2: Измерение скорости пакетного инференса ---
    print(f"\n--- Step 2: Measuring batch inference speed (batch size = {INFERENCE_BATCH_SIZE}) ---", flush=True)
    
    if samples_generated < INFERENCE_BATCH_SIZE:
        print("Not enough samples generated to run inference test. Please increase DATA_COLLECTION_SECONDS.")
        return

    # Берем сэмплы из буфера
    infosets_np, _ = replay_buffer.sample(INFERENCE_BATCH_SIZE)
    infosets_tensor = torch.from_numpy(infosets_np).to(device)

    # Прогрев
    for _ in range(10):
        _ = quantized_model(infosets_tensor)

    # Замер
    inference_start_time = time.time()
    num_inference_runs = 50
    for _ in range(num_inference_runs):
        _ = quantized_model(infosets_tensor)
    inference_end_time = time.time()

    total_inference_time = inference_end_time - inference_start_time
    time_per_batch_ms = (total_inference_time / num_inference_runs) * 1000
    time_per_inf_sample_ms = time_per_batch_ms / INFERENCE_BATCH_SIZE

    print(f"Executed {num_inference_runs} batches of size {INFERENCE_BATCH_SIZE} in {total_inference_time:.2f}s.")
    print(f"Time per batch: {time_per_batch_ms:.4f} ms")
    print(f"Time per inference sample (T_inference_batch): {time_per_inf_sample_ms:.6f} ms")

    # --- ЭТАП 3: Прогноз ---
    print("\n" + "="*40)
    print("PERFORMANCE PREDICTION WITH BATCH INFERENCE")
    print("="*40)
    
    predicted_total_time_ms = time_per_gen_sample_ms + time_per_inf_sample_ms
    predicted_throughput = 1000 / predicted_total_time_ms if predicted_total_time_ms > 0 else float('inf')
    
    print(f"T_generate: {time_per_gen_sample_ms:.6f} ms")
    print(f"T_inference_batch: {time_per_inf_sample_ms:.6f} ms")
    print(f"Predicted total time per sample: {predicted_total_time_ms:.6f} ms")
    print("-" * 40)
    print(f"PREDICTED THROUGHPUT: {predicted_throughput:.2f} samples/sec")
    print(f"Expected speedup over current ~1300 samples/sec: {predicted_throughput / 1300:.2f}x")
    print("="*40)


if __name__ == "__main__":
    # Нужно импортировать threading для stop_event
    import threading
    main()
