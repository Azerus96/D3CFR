# mccfr_ofc-main/python_src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
from model import DuelingNetwork
from replay_buffer import ReplayBuffer
from ofc_engine import DeepMCCFR # Наш C++ модуль

# --- Гиперпараметры ---
INPUT_SIZE = 1540 # 1+1+1+52 + 13*53 + 13*53 + 52 + 1 = 1540
ACTION_LIMIT = 200 # Максимальное количество действий (должно совпадать с C++)
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 500000
BATCH_SIZE = 256
TRAINING_INTERVAL = 100 # Обучаться каждые 100 сгенерированных партий
SAVE_INTERVAL = 1000 # Сохранять модель каждые 1000 партий
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
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    def predict_regrets(infoset_vector, num_actions):
        model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(infoset_vector).unsqueeze(0).to(device)
            pred_regrets = model(tensor, num_actions)
        return pred_regrets.squeeze(0).cpu().tolist()

    solver = DeepMCCFR(predict_regrets)
    print("Solver initialized. Starting training loop...")

    total_iterations = 0
    try:
        while True:
            start_time = time.time()
            
            # Запускаем N траверсов в C++ для сбора данных
            # Можно распараллелить, если C++ код потокобезопасен
            num_traversals_per_block = 10
            for _ in range(num_traversals_per_block):
                training_samples = solver.run_traversal()
                for sample in training_samples:
                    replay_buffer.push(sample.infoset_vector, sample.target_regrets, sample.num_actions)
            
            total_iterations += num_traversals_per_block
            
            # Обучение
            if len(replay_buffer) >= BATCH_SIZE:
                model.train()
                
                infosets, targets, num_actions_list = replay_buffer.sample(BATCH_SIZE)
                infosets = infosets.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                
                # Получаем предсказания для всех элементов батча
                predictions = model(infosets, ACTION_LIMIT) # Предсказываем для максимального размера
                
                # Создаем маску, чтобы считать лосс только для релевантных действий
                mask = torch.zeros_like(predictions)
                for i, n_actions in enumerate(num_actions_list):
                    mask[i, :n_actions] = 1.0
                
                # Применяем маску к предсказаниям и целям
                predictions_masked = predictions * mask
                targets_masked = targets * mask

                loss = criterion(predictions_masked, targets_masked)
                loss.backward()
                optimizer.step()
                
                duration = time.time() - start_time
                print(f"Iter: {total_iterations}, Loss: {loss.item():.6f}, Buffer: {len(replay_buffer)}, Time: {duration:.2f}s")

            # Сохранение
            if total_iterations % SAVE_INTERVAL < num_traversals_per_block:
                print(f"Saving model at iteration {total_iterations}...")
                torch.save(model.state_dict(), MODEL_PATH)
                # Здесь можно добавить логику коммита в Git

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved. Exiting.")

if __name__ == "__main__":
    main()
