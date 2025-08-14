# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import re

import json
import websocket
from ws_sender import WSClient

from mcts_selfplay import MCTS, play_self_play_game, play_vs_human, push_state_ws
from checkers_env import CheckersEnv

# ===== Simple neural network for Checkers =====
class CheckersNet(nn.Module):
    def __init__(self):
        super(CheckersNet, self).__init__()
        self.fc1 = nn.Linear(8 * 8 * 5, 128)
        self.fc2 = nn.Linear(128, 128)

        # Policy head
        self.policy = nn.Linear(128, 64 * 64)  # from-square to-square representation

        # Value head
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten all except batch dim
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        policy_logits = self.policy(x)
        value = torch.tanh(self.value(x))
        return policy_logits, value
    
    def predict(self, state_np):
        self.eval()
        with torch.no_grad():
            # convert numpy to tensor, add batch dim
            state_t = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
            device = next(self.parameters()).device
            state_t = state_t.to(device)
            logits, value = self(state_t)  # forward pass
            logits = logits.squeeze(0).cpu().numpy()
            value = value.item()
        return logits, value

def find_latest_checkpoint(model_dir="models", prefix="checkers_iter_"):
    if not os.path.exists(model_dir):
        return None, 0
    files = os.listdir(model_dir)
    max_iter = 0
    latest_file = None
    pattern = re.compile(rf"{prefix}(\d+)\.pth")
    for f in files:
        match = pattern.match(f)
        if match:
            iter_num = int(match.group(1))
            if iter_num > max_iter:
                max_iter = iter_num
                latest_file = os.path.join(model_dir, f)
    return latest_file, max_iter

def train(training=True):
    # Hyperparameters
    num_iterations = 1000
    games_per_iteration = 20
    epochs = 64
    batch_size = 256
    learning_rate = 3e-3

    # Create model, optimizer, loss
    device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CheckersNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Find latest checkpoint
    checkpoint_path, start_iter = find_latest_checkpoint()
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found, training from scratch.")
        start_iter = 0

    if training:
        for iteration in range(num_iterations):
            print(f"======= Iteration {iteration+1}/{num_iterations} =======")

            # Collect self-play data
            examples = []
            for game in range(games_per_iteration):
                print(f"  ===== GAME {game+1}/{games_per_iteration} =====")
                states, mcts_policies, rewards = play_self_play_game(model)
                print(f"    === Winner {np.sum(rewards)} ===")
                # Now zip them into tuples and extend examples
                game_examples = list(zip(states, mcts_policies, rewards))
                examples.extend(game_examples)

            # Convert to tensors
            states, pis, zs = zip(*examples)
            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)

            pis = torch.tensor(np.array(pis), dtype=torch.float32).to(device)
            zs = torch.tensor(np.array(zs), dtype=torch.float32).to(device)

            # Training loop
            for epoch in range(epochs):
                permutation = torch.randperm(states.size(0))
                for i in range(0, states.size(0), batch_size):
                    idx = permutation[i:i + batch_size]
                    batch_states = states[idx]
                    batch_pis = pis[idx]
                    batch_zs = zs[idx]

                    optimizer.zero_grad()
                    out_pis, out_vals = model(batch_states)
                    loss_p = nn.CrossEntropyLoss()(out_pis, torch.argmax(batch_pis, dim=1))
                    loss_v = nn.MSELoss()(out_vals.squeeze(), batch_zs)
                    loss = loss_p + loss_v
                    loss.backward()
                    optimizer.step()
                
                # Calculate policy accuracy for the entire dataset (optional: could also do per batch)
                with torch.no_grad():
                    predicted_moves = torch.argmax(model(states)[0], dim=1)
                    target_moves = torch.argmax(pis, dim=1)
                    policy_acc = (predicted_moves == target_moves).float().mean().item()

                print(f"    Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Policy Acc: {policy_acc:.2%}")

            # Save checkpoint
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/checkers_iter_{start_iter+iteration+1}.pth")
    else:
        # Play vs human
        play_vs_human(model)

def chooseGame():
    ws = websocket.WebSocket()
    ws.connect("ws://0.0.0.0:8080/python")

    while True:
        msg = json.loads(ws.recv())  # Wait for player button choise
        if msg["type"] == "start_training":
            print("Starting AI training...")
            train(training=True)
            break
        elif msg["type"] == "start_game":
            print("Starting human vs AI...")
            train(training=False)
            break

if __name__ == "__main__":
    chooseGame()