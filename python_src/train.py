# D2CFR-main/python_src/train.py (ВЕРСИЯ 5.0)

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import concurrent.futures
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- ГЛОБАЛЬНЫЙ КОНТРОЛЬ НАД ПОТОКАМИ ---
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
ACTION_LIMIT = 8 # Разумный компромисс для скорости и качества
LEARNING_RATE = 0.001
REPLAY_BUFFER_CAPACITY = 2000000
BATCH_SIZE = 256
TRAINING_BLOCK_SIZE = 48 # 10 задач на воркера
SAVE_INTERVAL_BLOCKS = 5 
MODEL_PATH = "d2cfr_model.pth"
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"
NUM_WORKERS = 24

def main():
    device = torch.device("cpu")
    print(f"Using device for PyTorch: {device}", flush=True)
    print(f"Using {NUM_WORKERS} worker threads for data collection.", flush=True)
    print(f"Each library (Torch, OpenMP, etc.) is limited to {NUM_COMPUTATION_THREADS} threads.", flush=True)

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

    print("\nConverting model to TorchScript for C++...", flush=True)
    model.eval()
    example_input = torch.randn(1, INPUT_SIZE).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
    print(f"TorchScript model saved to {TORCHSCRIPT_MODEL_PATH}", flush=True)

    replay_buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, ACTION_LIMIT)
    solvers = [DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT, replay_buffer) for _ in range(NUM_WORKERS)]
    print(f"{len(solvers)} solver instances created.", flush=True)

    total_traversals = 0
    block_counter = 0
    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            while True:
                block_counter += 1
                start_time = time.time()
                
                print(f"\n--- Block {block_counter} ---", flush=True)
                
                tasks_per_worker = TRAINING_BLOCK_SIZE // NUM_WORKERS
                print(f"Submitting {tasks_per_worker * NUM_WORKERS} traversal tasks...", flush=True)
                
                initial_buffer_size = replay_buffer.get_count()
                futures = [executor.submit(solver.run_traversal) for solver in solvers for _ in range(tasks_per_worker)]
                concurrent.futures.wait(futures)
                
                for future in futures:
                    if future.exception() is not None:
                        print(f"!!! An error occurred in a worker thread: {future.exception()}", flush=True)
                        traceback.print_exc()

                total_traversals += len(futures)
                final_buffer_size = replay_buffer.get_count()
                
                data_collection_time = time.time() - start_time
                samples_generated = final_buffer_size - initial_buffer_size
                
                print(f"Data collection finished in {data_collection_time:.2f}s. Buffer size: {final_buffer_size}. Samples generated: {samples_generated}", flush=True)

                if final_buffer_size < BATCH_SIZE:
                    print(f"Buffer size is too small, skipping training. Need {BATCH_SIZE}.", flush=True)
                    continue

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
                
                duration = time.time() - start_time
                samples_per_sec = samples_generated / duration if duration > 0 else 0
                
                print(f"Block {block_counter} | Total: {total_traversals:,} | Loss: {loss.item():.6f} | Throughput: {samples_per_sec:.2f} samples/s", flush=True)

                if block_counter % SAVE_INTERVAL_BLOCKS == 0:
                    print("-" * 50, flush=True)
                    print(f"Saving models at traversal {total_traversals:,}...", flush=True)
                    torch.save(model.state_dict(), MODEL_PATH)
                    model.eval()
                    traced_script_module = torch.jit.trace(model, example_input)
                    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
                    print("Models saved successfully.", flush=True)
                    print("-" * 50, flush=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", flush=True)
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
