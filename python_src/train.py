# D2CFR-main/python_src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
from model import DuelingNetwork
from replay_buffer import ReplayBuffer
from ofc_engine import DeepMCCFR

# --- Гиперпараметры ---
INPUT_SIZE = 1486 # ИСПРАВЛЕНО: Точный размер вектора признаков
ACTION_LIMIT = 200
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 500000
BATCH_SIZE = 256
TRAINING_BLOCK_SIZE = 100 
SAVE_INTERVAL_BLOCKS = 10 
MODEL_PATH = "d2cfr_model.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    def predict_regrets(infoset_vector, num_actions):
        model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(infoset_vector).unsqueeze(0).to(device)
            pred_regrets = model(tensor, num_actions)
        return pred_regrets.squeeze(0).cpu().tolist()

    solver = DeepMCCFR(predict_regrets)
    print("Solver initialized. Starting training loop...")

    total_traversals = 0
    block_counter = 0
    try:
        while True:
            block_counter += 1
            start_time = time.time()
            
            # --- Сбор данных ---
            for _ in range(TRAINING_BLOCK_SIZE):
                training_samples = solver.run_traversal()
                for sample in training_samples:
                    replay_buffer.push(sample.infoset_vector, sample.target_regrets, sample.num_actions)
            
            total_traversals += TRAINING_BLOCK_SIZE
            
            # --- Обучение ---
            if len(replay_buffer) < BATCH_SIZE:
                print(f"Block {block_counter} | Total Traversals: {total_traversals} | Buffer size {len(replay_buffer)} is too small, skipping training.")
                continue

            model.train()
            
            infosets, targets, num_actions_list = replay_buffer.sample(BATCH_SIZE)
            infosets = infosets.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            predictions = model(infosets, ACTION_LIMIT)
            
            mask = torch.zeros_like(predictions)
            for i, n_actions in enumerate(num_actions_list):
                mask[i, :n_actions] = 1.0
            
            predictions_masked = predictions * mask
            targets_masked = targets * mask

            loss = criterion(predictions_masked, targets_masked)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Добавим клиппинг градиента для стабильности
            optimizer.step()
            
            duration = time.time() - start_time
            traversals_per_sec = TRAINING_BLOCK_SIZE / duration if duration > 0 else float('inf')
            
            print(f"Block {block_counter} | Total: {total_traversals:,} | Loss: {loss.item():.6f} | Speed: {traversals_per_sec:.2f} trav/s")

            # --- Сохранение и коммит в Git ---
            if block_counter % SAVE_INTERVAL_BLOCKS == 0:
                print("-" * 50)
                print(f"Saving model at traversal {total_traversals:,}...")
                torch.save(model.state_dict(), MODEL_PATH)
                
                print("Pushing progress to GitHub...")
                os.system(f'git add {MODEL_PATH}')
                os.system(f'git commit -m "Training checkpoint after {total_traversals} traversals"')
                os.system('git push')
                print("Progress pushed successfully.")
                print("-" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
        torch.save(model.state_dict(), MODEL_PATH)
        print("Pushing final model to GitHub...")
        os.system(f'git add {MODEL_PATH}')
        os.system(f'git commit -m "Final training checkpoint after {total_traversals} traversals (manual stop)"')
        os.system('git push')
        print("Model saved and pushed. Exiting.")

if __name__ == "__main__":
    main()
