import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
# Убираем параллелизм для теста
# import concurrent.futures
# import traceback
# import threading

from .model import DuelingNetwork
from .replay_buffer import ReplayBuffer
# Убираем RequestManager, так как вызовы будут прямыми
from ofc_engine import DeepMCCFR 

# --- ГИПЕРПАРАМЕТРЫ ДЛЯ ТЕСТА ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 10 # Уменьшаем для скорости теста
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 256
TRAINING_BLOCK_SIZE = 1 # Одна задача за раз
SAVE_INTERVAL_BLOCKS = 50 
MODEL_PATH = "d2cfr_model_debug.pth"
# NUM_WORKERS = 1 # Больше не нужен

def main():
    # torch.set_num_threads(1) # Не так критично для одного потока
    device = torch.device("cpu")
    print(f"Using device for PyTorch: {device}")
    print(f"--- RUNNING IN SINGLE-THREADED DEBUG MODE ---")

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE, action_limit=ACTION_LIMIT)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Could not load model: {e}. Starting from scratch.")

    # --- ПРЯМОЙ ВЫЗОВ МОДЕЛИ (КАК БЫЛО РАНЬШЕ) ---
    def predict_regrets(infoset_vector, num_actions):
        model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(infoset_vector).unsqueeze(0).to(device)
            pred_regrets = model(tensor)
        return pred_regrets.squeeze(0).cpu().tolist()[:num_actions]

    # Создаем ОДИН солвер со старым конструктором
    solver = DeepMCCFR(predict_regrets, ACTION_LIMIT)
    print("Solver initialized. Starting training loop...")

    total_traversals = 0
    block_counter = 0
    try:
        while True:
            block_counter += 1
            start_time = time.time()
            
            print(f"Running 1 traversal task in the main thread...")
            
            # --- ВЫПОЛНЯЕМ ЗАДАЧУ ПРЯМО ЗДЕСЬ ---
            training_samples = solver.run_traversal()
            for sample in training_samples:
                replay_buffer.push(sample.infoset_vector, sample.target_regrets, sample.num_actions)
            
            collected_samples_count = len(training_samples)
            total_traversals += 1
            
            print(f"Data collection finished. Collected {collected_samples_count} samples. Buffer size: {len(replay_buffer)}")

            if len(replay_buffer) < BATCH_SIZE:
                print(f"Block {block_counter} | Total Traversals: {total_traversals} | Buffer size {len(replay_buffer)} is too small, skipping training.")
                continue

            # --- Фаза обучения ---
            model.train()
            
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
            
            duration = time.time() - start_time
            traversals_per_sec = 1 / duration if duration > 0 else float('inf')
            
            print(f"Block {block_counter} | Total: {total_traversals:,} | Loss: {loss.item():.6f} | Speed: {traversals_per_sec:.2f} trav/s")

            if block_counter % SAVE_INTERVAL_BLOCKS == 0:
                print("-" * 50)
                print(f"Saving model at traversal {total_traversals:,}...")
                torch.save(model.state_dict(), MODEL_PATH)
                print("-" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved. Exiting.")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
        torch.save(model.state_dict(), "d2cfr_model_error.pth")
        print("Saved an emergency copy of the model.")

if __name__ == "__main__":
    main()
