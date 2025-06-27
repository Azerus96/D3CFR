import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import concurrent.futures
import traceback
import threading

from .model import DuelingNetwork
from .replay_buffer import ReplayBuffer
from ofc_engine import DeepMCCFR, RequestManager, PredictionResult

# --- ГИПЕРПАРАМЕТРЫ ---
INPUT_SIZE = 1486 
# ИСПРАВЛЕНО: Увеличено значение для осмысленного обучения
ACTION_LIMIT = 24 
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 2000000
BATCH_SIZE = 2048
TRAINING_BLOCK_SIZE = 24
SAVE_INTERVAL_BLOCKS = 5 
MODEL_PATH = "d2cfr_model.pth"
NUM_WORKERS = 24
# Размер батча для обработки запросов от C++. Должен быть достаточно большим.
INFERENCE_BATCH_SIZE = 1024 

def inference_thread_func(request_manager, model, device):
    """
    Этот поток постоянно слушает запросы от C++, батчит их и обрабатывает.
    """
    while True:
        # Получаем пачку запросов. Этот вызов блокирующий.
        requests = request_manager.get_requests(INFERENCE_BATCH_SIZE)
        if not requests:
            # Может произойти при завершении, если очередь пуста
            continue

        infosets = [req.infoset_vector for req in requests]
        
        with torch.no_grad():
            tensor = torch.FloatTensor(np.array(infosets)).to(device)
            pred_regrets_batch = model(tensor)
        
        results = []
        for i, req in enumerate(requests):
            # Отправляем только нужное количество "сожалений"
            regrets = pred_regrets_batch[i].cpu().tolist()[:req.num_actions]
            # ИСПРАВЛЕНО: Создаем объект и заполняем поля по одному
            res = PredictionResult()
            res.id = req.id
            res.regrets = regrets
            results.append(res)
            
        # Отправляем результаты обратно в C++
        request_manager.post_results(results)

def main():
    torch.set_num_threads(1)
    device = torch.device("cpu")
    print(f"Using device for PyTorch: {device}", flush=True)
    print(f"Using {NUM_WORKERS} worker threads for data collection on CPU.", flush=True)

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    model.eval() # Модель всегда в режиме eval, т.к. обучение и инференс разделены
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE, action_limit=ACTION_LIMIT)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}", flush=True)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.", flush=True)
        except Exception as e:
            print(f"Could not load model: {e}. Starting from scratch.", flush=True)

    # Создаем ОДИН менеджер запросов для всех потоков
    request_manager = RequestManager()

    # Запускаем отдельный поток для обработки запросов к нейросети
    inf_thread = threading.Thread(target=inference_thread_func, args=(request_manager, model, device), daemon=True)
    inf_thread.start()
    print("Inference thread started.", flush=True)

    # Создаем один экземпляр солвера для каждого потока, но все они используют ОДИН менеджер
    solvers = [DeepMCCFR(request_manager, ACTION_LIMIT) for _ in range(NUM_WORKERS)]
    print("Solver instances created. Starting training loop...", flush=True)

    total_traversals = 0
    block_counter = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            while True:
                block_counter += 1
                start_time = time.time()
                
                print(f"Submitting {TRAINING_BLOCK_SIZE} traversal tasks to a pool of {NUM_WORKERS} workers...", flush=True)
                
                # Отправляем задачи в пул, каждый воркер использует свой экземпляр солвера
                futures = [executor.submit(solvers[i % NUM_WORKERS].run_traversal) for i in range(TRAINING_BLOCK_SIZE)]
                
                collected_samples_count = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        training_samples = future.result()
                        for sample in training_samples:
                            replay_buffer.push(sample.infoset_vector, sample.target_regrets, sample.num_actions)
                        collected_samples_count += len(training_samples)
                    except Exception as exc:
                        print(f'!!! A WORKER THREAD FAILED: {exc}', flush=True)
                        traceback.print_exc()

                total_traversals += TRAINING_BLOCK_SIZE
                
                print(f"Data collection finished. Collected {collected_samples_count} samples. Buffer size: {len(replay_buffer)}", flush=True)

                if len(replay_buffer) < BATCH_SIZE:
                    print(f"Block {block_counter} | Total Traversals: {total_traversals} | Buffer size {len(replay_buffer)} is too small, skipping training.", flush=True)
                    continue

                # --- Фаза обучения ---
                model.train() # Переключаем модель в режим обучения
                
                infosets, targets, num_actions_list = replay_buffer.sample(BATCH_SIZE)
                if infosets is None: continue

                infosets = infosets.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                
                predictions = model(infosets)
                
                num_actions_tensor = torch.tensor(num_actions_list, device=device, dtype=torch.int64).unsqueeze(1)
                indices = torch.arange(ACTION_LIMIT, device=device).unsqueeze(0)
                mask = (indices < num_actions_tensor).float()
                
                predictions_masked = predictions * mask
                targets_masked = targets * mask

                loss = criterion(predictions_masked, targets_masked)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                model.eval() # Возвращаем в режим инференса для потока-обработчика
                
                duration = time.time() - start_time
                traversals_per_sec = TRAINING_BLOCK_SIZE / duration if duration > 0 else float('inf')
                
                print(f"Block {block_counter} | Total: {total_traversals:,} | Loss: {loss.item():.6f} | Speed: {traversals_per_sec:.2f} trav/s", flush=True)

                # Логика сохранения модели
                if block_counter % SAVE_INTERVAL_BLOCKS == 0:
                    print("-" * 100, flush=True)
                    print(f"Saving model at traversal {total_traversals:,}...", flush=True)
                    torch.save(model.state_dict(), MODEL_PATH)
                
                    # --- НАЧАЛО РАСКОММЕНТИРОВАННОГО БЛОКА ---
                    # ВАЖНО: Этот блок может вызвать зависание, если Git запросит аутентификацию.
                    # Убедитесь, что вы настроили Git для использования токена без интерактивного ввода.
                    print("Pushing progress to GitHub...", flush=True)
                    os.system(f'git add {MODEL_PATH}')
                    os.system(f'git commit -m "Training checkpoint after {total_traversals} traversals"')
                    os.system('git push')
                    print("Progress pushed successfully.", flush=True)
                    # --- КОНЕЦ РАСКОММЕНТИРОВАННОГО БЛОКА ---
                
                    print("-" * 100, flush=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...", flush=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved. Exiting.", flush=True)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", flush=True)
        traceback.print_exc()
        torch.save(model.state_dict(), "d2cfr_model_error.pth")
        print("Saved an emergency copy of the model.", flush=True)

if __name__ == "__main__":
    main()
