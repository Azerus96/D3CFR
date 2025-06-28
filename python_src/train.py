# D2CFR-main/python_src/train.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import concurrent.futures
import traceback

from .model import DuelingNetwork
from .replay_buffer import ReplayBuffer
# MODIFIED: No longer need RequestManager or PredictionResult from the engine
from ofc_engine import DeepMCCFR

# --- HYPERPARAMETERS ---
INPUT_SIZE = 1486 
ACTION_LIMIT = 24 
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 2000000
BATCH_SIZE = 2048
TRAINING_BLOCK_SIZE = 24
SAVE_INTERVAL_BLOCKS = 5 
MODEL_PATH = "d2cfr_model.pth"
# This is the path to the *TorchScript* model, which we'll create.
TORCHSCRIPT_MODEL_PATH = "d2cfr_model_script.pt"
NUM_WORKERS = 24

# REMOVED: The inference_thread_func is no longer needed.

def main():
    torch.set_num_threads(1)
    device = torch.device("cpu")
    print(f"Using device for PyTorch: {device}", flush=True)
    print(f"Using {NUM_WORKERS} worker threads for data collection on CPU.", flush=True)

    model = DuelingNetwork(input_size=INPUT_SIZE, max_actions=ACTION_LIMIT).to(device)
    
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

    # --- ADDED: Convert and save the model in TorchScript format ---
    print("Converting model to TorchScript for C++...", flush=True)
    model.eval() # Important to be in eval mode for tracing
    # Use an example tensor to trace the model
    example_input = torch.randn(1, INPUT_SIZE).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
    print(f"TorchScript model saved to {TORCHSCRIPT_MODEL_PATH}", flush=True)
    # --- END ADDED SECTION ---

    # MODIFIED: Create solvers, passing the path to the TorchScript model
    # The C++ engine will load this file directly.
    solvers = [DeepMCCFR(TORCHSCRIPT_MODEL_PATH, ACTION_LIMIT) for _ in range(NUM_WORKERS)]
    print("Solver instances created. Starting training loop...", flush=True)

    total_traversals = 0
    block_counter = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            while True:
                block_counter += 1
                start_time = time.time()
                
                print(f"Submitting {TRAINING_BLOCK_SIZE} traversal tasks to {NUM_WORKERS} workers...", flush=True)
                
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

                # --- Training Phase (Model is only used here in Python) ---
                model.train()
                
                infosets, targets, _ = replay_buffer.sample(BATCH_SIZE)
                if infosets is None: continue

                infosets = infosets.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                predictions = model(infosets)
                
                # The mask is still a good idea to avoid calculating loss on padding
                mask = (targets != 0).float()
                loss = criterion(predictions * mask, targets)
                
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                duration = time.time() - start_time
                traversals_per_sec = TRAINING_BLOCK_SIZE / duration if duration > 0 else float('inf')
                
                print(f"Block {block_counter} | Total: {total_traversals:,} | Loss: {loss.item():.6f} | Speed: {traversals_per_sec:.2f} trav/s", flush=True)

                if block_counter % SAVE_INTERVAL_BLOCKS == 0:
                    print("-" * 100, flush=True)
                    print(f"Saving models at traversal {total_traversals:,}...", flush=True)
                    # Save the standard PyTorch model for continued training
                    torch.save(model.state_dict(), MODEL_PATH)
                    # Re-save the TorchScript version for the C++ engine
                    model.eval()
                    traced_script_module = torch.jit.trace(model, example_input)
                    traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
                    print("Models saved successfully.", flush=True)
                    print("-" * 100, flush=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final models...", flush=True)
        torch.save(model.state_dict(), MODEL_PATH)
        model.eval()
        traced_script_module = torch.jit.trace(model, example_input)
        traced_script_module.save(TORCHSCRIPT_MODEL_PATH)
        print("Models saved. Exiting.", flush=True)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()
