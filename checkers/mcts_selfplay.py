# mcts_selfplay.py
import math
import numpy as np
import random
from collections import defaultdict
from checkers_env import CheckersEnv

import json
import websocket
from ws_sender import WSClient


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0

    def reset(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, game, model, n_simulations=50, c_puct=2.0):
        self.game = game
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def run(self, root):
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]

            # 1. Selection
            while node.expanded():
                action, node = self.select_child(node) #select best child
                search_path.append(node)

            # 2. Expansion
            value, policy = self.evaluate(node.state)
            self.expand(node, policy)

            # 3. Backpropagation
            self.backpropagate(search_path, value)

    def select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            ucb_score = self.ucb_score(node, child)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, parent, child):
        prior_score = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        return child.value() + prior_score

    def expand(self, node, policy):
        board, current_player = node.state
        legal_moves = self.game.get_legal_moves(board, current_player)

        for move in legal_moves:
            board_copy = board.copy()

            self.game._make_move_no_switch(board_copy, move)

            # Switch player after the move
            next_player = -current_player  # or 1 - current_player if using 0/1

            next_state = (board_copy, next_player)

            child_node = Node(next_state, parent=node)
            child_node.prior = policy.get(self.make_move_hashable(move), 1 / len(legal_moves))

            node.children[self.make_move_hashable(move)] = child_node

    def make_move_hashable(self, move):
        start_pos, path = move
        return (start_pos, tuple(path))

    def evaluate(self, state):
        board, current_player = state
        board_tensor = self.game.state_to_tensor(state)
        policy_logits, value = self.model.predict(board_tensor)
        policy = self.softmax(policy_logits)

        legal_moves = self.game.get_legal_moves(board, current_player)
        policy_dict = {}
        for i, move in enumerate(legal_moves):
            policy_dict[self.make_move_hashable(move)] = policy[i]

        return float(value), policy_dict

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Switch perspective for the other player

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


def move_to_index(move):
    # Convert move tuple ((from_r, from_c), [(to_r, to_c), ...]) to single int index
    start_pos, path = move
    from_r, from_c = start_pos
    to_r, to_c = path[-1]
    from_idx = from_r * 8 + from_c
    to_idx = to_r * 8 + to_c
    return from_idx * 64 + to_idx

def play_self_play_game(model):
    from train import chooseGame

    env = CheckersEnv()
    state = env.reset()
    game = env
    mcts = MCTS(game, model)

    ws = websocket.WebSocket()
    ws.connect("ws://0.0.0.0:8080/python")

    root = Node(state)
    states, mcts_policies, rewards = [], [], []
    done = False

    while not done:
        # Check for back btn
        '''msg = json.loads(ws.recv()) 
        if msg["type"] == "stop":
            state = env.reset()
            push_state_ws(ws, state, env)
            chooseGame()
            break'''


        mcts.run(root)

        children_moves = list(root.children.keys())

        # Convert visit counts to probabilities
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        policy = visit_counts / visit_counts.sum()

        # Convert partial policy over legal moves to full vector
        policy_full = np.zeros(4096, dtype=np.float32)
        for prob, move in zip(policy, root.children.keys()):
            idx = move_to_index(move)
            policy_full[idx] = prob

        states.append(game.state_to_tensor(root.state))
        mcts_policies.append(policy_full)

        # Sample action proportionally to visit counts with a temperature
        move_number = len(states)
        temperature = 2 + (0.5 - 2) * min(move_number / 100, 1.0) #temperature starts at 2 and decreaces to 0.5 after 100 moves
        policy_temp = np.power(policy, 1 / temperature)
        policy_temp /= policy_temp.sum()
        action = random.choices(list(root.children.keys()), weights=policy_temp, k=1)[0]

        state, reward, done, _ = env.step(action)
        rewards.append(reward)

        #env.render()
        #send state with websockets
        legal_moves = env.get_legal_moves(env.board.copy(), env.current_player)
        ###ws.send_state(state[0], state[1], legal_moves)
        push_state_ws(ws, state, game, False)

        if done:
            break

        root = root.children[action]
        root.parent = None  # Free memory
        root.reset(state)
    ws.close()  # Close WS connection when done

    return states, mcts_policies, rewards

def play_vs_human(model):
    from train import chooseGame

    env = CheckersEnv()
    state = env.reset()
    game = env
    mcts = MCTS(game, model, n_simulations=300)

    ws = websocket.WebSocket()
    ws.connect("ws://0.0.0.0:8080/python")

    # Send initial board state to browser
    push_state_ws(ws, state, game, True)

    # Game loop
    while True:
        msg = json.loads(ws.recv())  # Wait for player move
        if msg["type"] == "player_move":
            # Apply player's move
            move_dict = msg["move"]
            move = (tuple(move_dict["from"]), [tuple(pos) for pos in move_dict["path"]])
            state, reward, done, _ = env.step(move)

            push_state_ws(ws, state, game, True)

            if done:
                break

            # AI's turn
            root = Node(state)
            mcts.run(root)

            # Pick best move by visits
            best_move = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
            state, reward, done, _ = env.step(best_move)

            push_state_ws(ws, state, game, True)

            if done:
                break
        elif msg["type"] == "stop":
            state = env.reset()
            push_state_ws(ws, state, env, False)
            chooseGame()
            break
    # Finished game loop
    while (True):
        msg = json.loads(ws.recv()) 
        push_state_ws(ws, state, game, False)
        if msg["type"] == "stop":
            state = env.reset()
            push_state_ws(ws, state, env, False)
            chooseGame()
            break

    ws.close()

def push_state_ws(ws, state, game, vs_human):
    ws.send(json.dumps({
        "type": "state",
        "board": state[0].tolist(),
        "current_player": state[1],
        "legal_moves": [{"from": mv[0], "path": mv[1]} for mv in game.get_legal_moves(game.board.copy(), game.current_player)],
        "vs_human": vs_human
    }))