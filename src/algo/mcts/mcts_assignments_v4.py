import copy
import json
from pathlib import Path
import pandas as pd
import math
import argparse
import os
import multiprocessing
import random
from dateutil.relativedelta import relativedelta
import numpy as np
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from utils import assign_aulas
from algo.data import Data

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
USER = os.getlogin()
PATH_ASSIGNMENT = Path(f'./output')

# Virtual Loss Constants (Heuristics from Chaslot et al.)
VIRTUAL_LOSS = 3
C_PARAM = math.sqrt(2)

class RoomLog(Data):
    def __init__(self, periodo: int, sede: str, data_path, room_log_path,
                 items_path, items_predict_path):
        super().__init__(
            periodo, sede, data_path, room_log_path,
            items_path, items_predict_path)

        self.sede = sede
        self.periodo = periodo
        self.data_path = data_path
        self.room_log_path = room_log_path
        self.items_path = items_path
        self.items_predict_path = items_predict_path
        self.roomlog = self.get_room_log()
        self.aulas, self.aforos = self.get_aulas_aforos(self.roomlog)
        self.items = self.get_items_predict()
        self.idx_item = 0
        self.n_items = len(self.items)
        self.n_aulas = len(self.aulas)

    def clone(self):
        g = RoomLog(
            self.periodo, self.sede, self.data_path, self.room_log_path,
            self.items_path, self.items_predict_path
        )
        g.idx_item = self.idx_item
        g.roomlog = copy.deepcopy(self.roomlog.copy())
        return g

    def step(self, action: int):
        if self.idx_item >= self.n_items:
            return None

        item = self.items[self.idx_item].copy()
        aula = self.aulas[action]
        aforo = self.aforos[action]
        roomlog = self.roomlog.copy()

        # Reward Calculation
        if (aforo - item['FORECAST_ALUMN']) < 0:
            reward = aforo - item['FORECAST_ALUMN'] - 2
        elif ((aforo - item['FORECAST_ALUMN']) >= 0) and ((aforo - item['FORECAST_ALUMN']) <= 2):
            reward = 1 + (item['FORECAST_ALUMN'] / aforo)
        else:
            reward = 0

        dias = item['DIAS']
        franjas = item['TURNOS']

        self.roomlog = assign_aulas(roomlog, franjas, dias, aula)
        self.idx_item += 1
        return reward

    def get_available_actions(self):
        if self.idx_item >= self.n_items:
            return []

        item = self.items[self.idx_item].copy()
        roomlog = self.roomlog.copy()
        dias = item['DIAS']
        franjas = item['TURNOS']
        aulas_disponibles = self.get_aulas_disponibles(roomlog, franjas, dias)

        available_collection = []
        for idx, disponible in enumerate(aulas_disponibles):
            if disponible:
                available_collection.append(idx)

        return available_collection

    def is_terminal(self):
        return (self.idx_item >= self.n_items) | (len(self.get_available_actions()) == 0)


class Node:
    def __init__(self, move=None, parent=None, untried_actions=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.w = 0.0
        self.visits = 0
        self.untried_actions = [] if untried_actions is None else untried_actions.copy()
        
        # --- 1. LOCAL MUTEX ---
        # Each node gets its own lock for thread safety during updates
        self.lock = threading.Lock()

    def uct_select_child(self, c_param=math.sqrt(2)):
        # Reads statistics. In Tree Parallelization, these stats include Virtual Loss
        # applied by other threads currently traversing the tree.
        best = max(self.children, key=lambda child: (
            float('inf') if child.visits == 0 else
            (child.w / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
        ))
        return best

    def add_child(self, move, untried_actions):
        child = Node(move=move, parent=self, untried_actions=untried_actions)
        self.untried_actions.remove(move)
        self.children.append(child)
        return child

    def update(self, reward):
        # Basic update, locking handles external consistency
        self.visits += 1
        self.w += reward


def mcts_worker(root_node, initial_state, iter_count, c_param):
    """
    Worker function for Tree Parallelization.
    Executes MCTS iterations on the SHARED tree (root_node).
    """
    
    for _ in range(iter_count):
        node = root_node
        
        # Clone state for this specific thread's simulation
        # Note: We must sync the clone state with the tree traversal
        clone = initial_state.clone()
        
        # Track the path for Virtual Loss removal later
        path_nodes = []

        # =============================
        # 1. SELECTION (w/ Virtual Loss)
        # =============================
        while True:
            # We check untried actions and children availability inside lock logic
            # to ensure thread safety during expansion decisions.
            
            is_leaf = False
            with node.lock:
                if node.untried_actions:
                    is_leaf = True # We can expand here
                elif not node.children:
                    is_leaf = True # Terminal in tree
                
                if not is_leaf:
                    # Select best child
                    node = node.uct_select_child(c_param)
                    
                    # --- 2. APPLY VIRTUAL LOSS ---
                    # We artificially boost visits and lower wins (penalty)
                    # so other threads choose different paths.
                    node.visits += VIRTUAL_LOSS
                    node.w -= VIRTUAL_LOSS
                    path_nodes.append(node)
            
            if is_leaf:
                break
                
            # If we descended, we must advance the state to match the node
            # This is computationally expensive but necessary in this domain
            clone.step(node.move)

        # =============================
        # 2. EXPANSION
        # =============================
        # We are at a leaf or a node with untried actions.
        
        # Check if state is terminal before expanding
        if not clone.is_terminal():
            with node.lock:
                # Double check untried actions inside lock (another thread might have just taken one)
                if node.untried_actions:
                    action = random.choice(node.untried_actions)
                    # Advance state
                    clone.step(action)
                    # Get available actions for the NEW state
                    available_children = clone.get_available_actions()
                    
                    # Create child
                    node = node.add_child(move=action, untried_actions=available_children)
                    
                    # Apply Virtual Loss to the new node as well
                    node.visits += VIRTUAL_LOSS
                    node.w -= VIRTUAL_LOSS
                    path_nodes.append(node)

        # =============================
        # 3. SIMULATION
        # =============================
        # Run simulation on the thread-local clone (No locks needed here)
        rewards = []
        while not clone.is_terminal():
            possible_moves = clone.get_available_actions()
            if not possible_moves:
                break
            reward = clone.step(random.choice(possible_moves))
            rewards.append(reward)

        final_reward = sum(rewards) / max(1, clone.n_items)

        # =============================
        # 4. BACKPROPAGATION (w/ Virtual Loss Removal)
        # =============================
        # Iterate backwards up the path
        for n in reversed(path_nodes):
            with n.lock:
                # Remove Virtual Loss
                n.visits -= VIRTUAL_LOSS
                n.w += VIRTUAL_LOSS
                
                # Apply Real Result
                n.update(final_reward)
        
        # Update root (root usually doesn't have virtual loss applied in some implementations, 
        # but if we track visits there, we update it)
        with root_node.lock:
            root_node.update(final_reward)


def tree_parallel_UCT(state: RoomLog, available_actions: list, iter_max=5000, c_param=math.sqrt(2)):
    """
    Tree Parallel MCTS using Threads and Shared Memory.
    """
    
    # Initialize Root
    root_node = Node(move=None, parent=None, untried_actions=available_actions)
    
    # Determine concurrency
    # We use ThreadPoolExecutor because we need to share the 'root_node' object reference.
    # Note: In Python, due to GIL, this is concurrency, not CPU parallelism. 
    # However, this implementation strictly follows the logic required for Tree Parallelization 
    # (Mutexes/Virtual Loss) which is applicable if ported to C++ or JIT compiled.
    num_threads = min(os.cpu_count() or 4, 8) 
    iters_per_thread = iter_max // num_threads
    
    # Launch Threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for _ in range(num_threads):
            # Pass the SAME root_node to all workers (Shared Memory)
            # Pass a CLONE of the state so they have a fresh starting point
            futures.append(
                executor.submit(mcts_worker, root_node, state, iters_per_thread, c_param)
            )
        
        # Wait for all threads to complete
        for f in futures:
            f.result()

    # Select best move
    print([child.visits for child in root_node.children])
    best_child = max(root_node.children, key=lambda c: c.visits)
    
    # Update the actual state outside logic
    state.step(best_child.move)
    
    return best_child.move, root_node, state


def run_mcts(periodo, sede, data_path, room_log_path, items_path, items_predict_path, iter_max: int = 5000):
    timer = time.time()
    state = RoomLog(periodo, sede, data_path, room_log_path, items_path, items_predict_path)
    aulas = []
    print("Time to load state:", time.time() - timer)
    
    while state.idx_item < len(state.items):
        print("----------------------------------")
        print("IDX:", state.idx_item)
        
        available_actions = state.get_available_actions()
        
        if len(available_actions) == 0:
            result = copy.deepcopy(state.items[state.idx_item].copy())
            result['ASSIGNMENTS'] = {'AULA': None, 'AFORO': None}
            state.idx_item += 1
            aulas.append(result)
        else:
            timer = time.time()
            
            # CALL TREE PARALLEL UCT
            move, _, state = tree_parallel_UCT(state, available_actions=available_actions, iter_max=iter_max)
            
            print("Time to UCT:", time.time() - timer)
            
            result = copy.deepcopy(state.items[state.idx_item-1].copy()) # -1 because step incremented inside UCT return
            # Note: In the original logic, state.step was called after UCT. 
            # In my modification, I perform state.step inside tree_parallel_UCT return logic 
            # to keep it consistent, but we need to capture the result for the excel.
            
            # Re-fetch current item details based on the move made
            # Since state.step was called, idx_item has increased.
            result['ASSIGNMENTS'] = {'AULA': state.aulas[move],
                                     'AFORO': state.aforos[move]}
            aulas.append(result)

    print("☰"*(len(sede) + 10))
    print(f"✅ Periodo: {periodo}")
    print(f"📌 Sede: {sede}")
    print("☰"*(len(sede) + 10))
    
    best_data_frame = pd.DataFrame(aulas)
    output_path = PATH_ASSIGNMENT / f'{periodo}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    best_data_frame.to_excel(output_path / f'asignacion_{periodo}_{sede}.xlsx', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sede", type=str, default="Ica")
    parser.add_argument("--iter_max", type=int, default=100) # Lowered for quick testing
    parser.add_argument("--periodo_franja", type=str, default="1. Lun - Vie")
    
    # Mock paths for arguments if running in isolation without valid file paths
    # You should pass actual paths or use defaults that match your file structure
    
    args = parser.parse_args()
    
    # Note: The user needs to ensure Data/RoomLog works with these paths
    # run_mcts(...)
    print("Setup complete. Call run_mcts with valid data paths to execute.")