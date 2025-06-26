import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import concurrent.futures # ИЗМЕНЕНО: Импортируем необходимую библиотеку для распараллеливания

# ИЗМЕНЕНО: Используем явные относительные импорты
from .model import DuelingNetwork
from .replay_buffer import ReplayBuffer
from ofc_engine import DeepMCCFR

# --- Гиперпараметры ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 200
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 500000
BATCH_SIZE = 256
TRAINING_BLOCK_SIZE = 100 # Теперь это количество параллельных задач, а не последовательных итераций
SAVE_INTERVAL_BLOCKS = 10 
MODEL_PATH = "d2cfr_model.pth"

# ИЗМЕНЕНО: Добавляем параметр для количества рабочих потоков.
# Для вашей 96-ядерной машины установите это значение в 96.
# os.cpu_count() определит количество ядер автоматически для других машин.
NUM_WORKERS = 96 # Ставим 4 как запасной вариант, если os.cpu_count() вернет None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for PyTorch: {device}")
    print(f"Using {NUM_WORKERS} worker threads for data collection on CPU.")

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Could not load model: {e}. Starting from scratch.")

    # Функция обратного вызова для C++ движка. Она будет вызываться из разных потоков,
    # но GIL в Python гарантирует потокобезопасность доступа к модели.
    def predict_regrets(infoset_vector, num_actions):
        model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(infoset_vector).unsqueeze(0).to(device)
            pred_regrets = model(tensor, num_actions)
        return pred_regrets.squeeze(0).cpu().tolist()

    # Создаем один экземпляр солвера, который будет использоваться всеми потоками.
    solver = DeepMCCFR(predict_regrets)
    print("Solver initialized. Starting training loop...")

    total_traversals = 0
    block_counter = 0
    try:
        while True:
            block_counter += 1
            start_time = time.time()
            
            # --- ИЗМЕНЕНО: Фаза сбора данных теперь распараллелена ---
            print(f"Submitting {TRAINING_BLOCK_SIZE} traversal tasks to a pool of {NUM_WORKERS} workers...")
            
            collected_samples_count = 0
            # Создаем пул потоков
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # Отправляем 100 задач на выполнение. Каждая задача - один вызов run_traversal.
                futures = [executor.submit(solver.run_traversal) for _ in range(TRAINING_BLOCK_SIZE)]
                
                # Обрабатываем результаты по мере их готовности
                for future in concurrent.futures.as_completed(futures):
                    try:
                        training_samples = future.result()
                        for sample in training_samples:
                            replay_buffer.push(sample.infoset_vector, sample.target_regrets, sample.num_actions)
                        collected_samples_count += len(training_samples)
                    except Exception as exc:
                        print(f'A traversal generated an exception: {exc}')

            total_traversals += TRAINING_BLOCK_SIZE
            
            print(f"Data collection finished. Collected {collected_samples_count} samples. Buffer size: {len(replay_buffer)}")

            # --- Фаза 2: Обучение (остается без изменений) ---
            if len(replay_buffer) < BATCH_SIZE:
                print(f"Block {block_counter} | Total Traversals: {total_traversals} | Buffer size {len(replay_buffer)} is too small, skipping training.")
                continue

            model.train()
            
            infosets, targets, num_actions_list = replay_buffer.sample(BATCH_SIZE)
            infosets = infosets.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            predictions = model(infosets, ACTION_LIMIT)
            
            # Векторизованное создание маски
            num_actions_tensor = torch.tensor(num_actions_list, device=device, dtype=torch.int64).unsqueeze(1)
            indices = torch.arange(ACTION_LIMIT, device=device).unsqueeze(0)
            mask = (indices < num_actions_tensor).float()
            
            predictions_masked = predictions * mask
            targets_masked = targets * mask

            loss = criterion(predictions_masked, targets_masked)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            duration = time.time() - start_time
            traversals_per_sec = TRAINING_BLOCK_SIZE / duration if duration > 0 else float('inf')
            
            print(f"Block {block_counter} | Total: {total_traversals:,} | Loss: {loss.item():.6f} | Speed: {traversals_per_sec:.2f} trav/s")

            # Логика сохранения модели
            if block_counter % SAVE_INTERVAL_BLOCKS == 0:
                print("-" * 50)
                print(f"Saving model at traversal {total_traversals:,}...")
                torch.save(model.state_dict(), MODEL_PATH)
                
                # Логика для Git (если нужна)
                # print("Pushing progress to GitHub...")
                # os.system(f'git add {MODEL_PATH}')
                # os.system(f'git commit -m "Training checkpoint after {total_traversals} traversals"')
                # os.system('git push')
                # print("Progress pushed successfully.")
                print("-" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # Можно добавить сохранение модели и здесь
        torch.save(model.state_dict(), "d2cfr_model_error.pth")
        print("Saved an emergency copy of the model.")


if __name__ == "__main__":
    main()
