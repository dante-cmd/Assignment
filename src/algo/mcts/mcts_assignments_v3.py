import copy
import json
from pathlib import Path
import pandas as pd
import math, random, time
import argparse
import os
import multiprocessing
import random
from dateutil.relativedelta import relativedelta
import numpy as np
import re
import time
import json
from utils import assign_aulas
import copy
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from algo.data import Data


USER = os.getlogin()
# PATH_ASSIGNMENT = Path(f'C:/Users/{USER}/apis/02_assignment')
PATH_ASSIGNMENT = Path(f'./output')


class RoomLog(Data):

    def __init__(self,  periodo: int, sede: str, data_path, room_log_path,
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

        # Get item
        item = self.items[self.idx_item].copy()

        # Get aula and aforo for action
        aula = self.aulas[action]
        aforo = self.aforos[action]
        roomlog = self.roomlog.copy()

        # Reward
        if (aforo - item['FORECAST_ALUMN']) < 0:
            reward = aforo - item['FORECAST_ALUMN'] - 2

        elif ((aforo - item['FORECAST_ALUMN']) >= 0) and ((aforo - item['FORECAST_ALUMN']) <= 2):
            reward = 1 + (item['FORECAST_ALUMN'] / aforo)

        else:
            reward = 0

        # Update roomlog
        dias = item['DIAS'] # self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        franjas = item['TURNOS'] # self.dim_horario[item['HORARIO']]

        self.roomlog = assign_aulas(roomlog, franjas, dias, aula)

        # Update idx_item
        self.idx_item += 1
        return reward

    def get_available_actions(self):
        if self.idx_item >= self.n_items:
            return []

        item = self.items[self.idx_item].copy()
        aulas = self.aulas.copy()
        roomlog = self.roomlog.copy()
        dias = item['DIAS']
        franjas = item['TURNOS']
        aulas_disponibles = self.get_aulas_disponibles(roomlog, franjas, dias)
        # dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        # franjas = self.dim_horario[item['HORARIO']]

        available_collection = []

        for idx, (aula, disponible) in enumerate(zip(aulas, aulas_disponibles)):
            if disponible:
                available_collection.append(idx)

        return available_collection

    def is_terminal(self):
        return (self.idx_item >= self.n_items) | (len(self.get_available_actions()) == 0)


class Node:
    def __init__(self, move=None, parent=None, untried_actions=None,
                 available_actions=True):
        self.move = move  # the move that led to this node (from parent)
        self.parent = parent  # parent node
        self.children = []  # list of child nodes
        self.w = 0.0  # number of wins
        self.visits = 0  # visit count
        self.untried_actions = [] if untried_actions is None else untried_actions.copy()  # moves not expanded yet
        self.available_actions = available_actions

    def uct_select_child(self, c_param=math.sqrt(2)):
        # Select a child according to UCT (upper confidence bound applied to trees)
        # If a child has 0 visits we consider its UCT value infinite to ensure it's visited.
        best = max(self.children, key=lambda child: (
            float('inf') if child.visits == 0 else
            (child.w / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
        ))
        return best

    def add_child(self, move, untried_actions, available_actions):
        child = Node(move=move, parent=self, untried_actions=untried_actions,
                     available_actions=available_actions)
        self.untried_actions.remove(move)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.w += reward


def UCT(state, iter_max=5000, c_param=math.sqrt(2)):
    # PATH = Path("project")
    # SEDE = 'Ica'
    # iter_max = 5000
    # c_param=math.sqrt(2)
    # dataset_01 = DataSet(PATH)
    # state = RoomLog(dataset_01, SEDE)
    available_actions = state.get_available_actions()
    root_node = Node(move=None,
                     parent=None,
                     untried_actions=available_actions,
                     available_actions=True if len(available_actions) > 0 else False)

    clone = state.clone()

    for i in range(iter_max):
        # i = 1
        rewards = []
        node = root_node
        clone.idx_item = state.idx_item
        clone.roomlog = copy.deepcopy(state.roomlog.copy())

        # 1. Selection: descend until we find a node with untried actions or a leaf (no children)
        while node.untried_actions == [] and node.children:
            node = node.uct_select_child(c_param)
            reward = clone.step(node.move)
            rewards.append(reward)

        # 2. Expansion: if we can expand (i.e. state not terminal) pick an untried action
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            reward = clone.step(action)
            rewards.append(reward)
            available_children = clone.get_available_actions()
            node = node.add_child(move=action,
                                  untried_actions=available_children,
                                  available_actions=True if len(available_children) > 0 else False)

        # 3. Simulation: play randomly until the game ends
        while not clone.is_terminal():
            possible_moves = clone.get_available_actions()
            reward = clone.step(random.choice(possible_moves))
            rewards.append(reward)

        # 4. Backpropagation: update node statistics with simulation result
        # n_items = min(max(clone.idx_item, 1), clone.n_items)
        n_items = clone.n_items

        while node is not None:
            node.update(sum(rewards) / n_items)
            node = node.parent

    # return the move that was most visited
    best_child = max(root_node.children, key=lambda c: c.visits)
    clone.idx_item = state.idx_item
    clone.roomlog = copy.deepcopy(state.roomlog.copy())

    return best_child.move, root_node, clone  # also return root node so the caller can inspect children statistics


def UCT_worker(args):
    state, iter_max, c_param = args
    # Clone state for isolation
    cloned_state = state.clone()
    # RoomLog(state.dataset, state.sede, state.periodo_franja)
    cloned_state.idx_item = state.idx_item
    cloned_state.roomlog = copy.deepcopy(state.roomlog)

    move, root, _ = UCT(cloned_state, iter_max=iter_max, c_param=c_param)
    return root


def parallel_UCT(state, iter_max=5000, c_param=math.sqrt(2)):
    # max_workers=12
    # get n_cores
    # n_cores = os.cpu_count()
    # split iterations across workers

    n_cores = os.cpu_count()

    assert isinstance(n_cores, int)

    iters_per_worker = iter_max // n_cores
    roots = []

    with multiprocessing.Pool(processes=n_cores) as pool:
        # run mcts
        results = pool.map(UCT_worker, [(state, iters_per_worker, c_param) for _ in range(n_cores)])
        for result in results:
            roots.append(result)

    # merge children statistics from workers
    merged_root = Node(move=None, parent=None,
                       untried_actions=state.get_available_actions())
    move_to_node = {}

    for r in roots:
        for child in r.children:
            if child.move not in move_to_node:
                move_to_node[child.move] = Node(move=child.move,
                                                parent=merged_root,
                                                untried_actions=[])
            move_to_node[child.move].visits += child.visits
            move_to_node[child.move].w += child.w

    merged_root.children = list(move_to_node.values())
    best_child = max(merged_root.children, key=lambda c: c.visits)

    return best_child.move, merged_root, state


def run_mcts(periodo, sede, data_path, room_log_path,items_path, items_predict_path, iter_max: int = 5000):
    # --- Demo: use UCT on an empty RoomLog board ---
    # path = Path("project")
    # data = DataSet(path)
    # periodo, sede, data_path, room_log_path,items_path, items_predict_path
    state = RoomLog(periodo, sede, data_path, room_log_path,items_path, items_predict_path)
    aulas = []
    total_time = time.time()
    while state.idx_item < len(state.items):
        start_time = time.time()
        result = copy.deepcopy(state.items[state.idx_item].copy())
        if len(state.get_available_actions()) == 0:
            result['ASSIGNMENTS'] = {'AULA': None,
                                     'AFORO': None}
            state.idx_item += 1
        else:
            move, root_node, state = parallel_UCT(state, iter_max=iter_max)
            # move, root_node, state = UCT(state, iter_max=5000)   # 2000 rollouts from empty board
            result['ASSIGNMENTS'] = {'AULA': state.aulas[move],
                                     'AFORO': state.aforos[move]}
            state.step(move)
        aulas.append(result)
        duration = time.time() - start_time
        total_duration = time.time() - total_time
        duration_per_item = total_duration / (state.idx_item)
        remaining_time = duration_per_item * (len(state.items) - state.idx_item)
        to_time = lambda x: time.strftime('%H:%M:%S', time.gmtime(x))
        print(
            f"Total duration: {to_time(total_duration)} | Actual duration: {to_time(duration)} | Remaining time: {to_time(remaining_time)} | {state.idx_item}/{len(state.items)}",
            end="\r")
    print()
    # return aulas

    print("☰"*(len(sede) + 10))
    print(f"✅ Periodo: {periodo}")
    print(f"📌 Sede: {sede}")
    # print(f"⏱ Time: {process_time}")
    print("☰"*(len(sede) + 10))
    
    best_data_frame = pd.DataFrame(aulas)
    best_data_frame.to_excel(PATH_ASSIGNMENT / f'{periodo}/asignacion_{periodo}_{sede}.xlsx',
                                 index=False)
    # Path('../../output').mkdir(exist_ok=True)

    # df.to_excel('output/assignments_{}.xlsx'.format(sede), index=False)


# ===============================
# Node Class (with per-node Lock)
# ===============================
class Node:
    def __init__(self, move=None, parent=None, untried_actions=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.untried_actions = untried_actions[:] if untried_actions else []
        self.visits = 0
        self.w = 0.0
        self.lock = threading.Lock()  # <-- Lock for this node

    def add_child(self, move, state):
        with self.lock:
            if move in self.untried_actions:
                self.untried_actions.remove(move)
            child_node = Node(
                move=move,
                parent=self,
                untried_actions=state.get_legal_moves()
            )
            self.children.append(child_node)
        return child_node

    def update(self, reward):
        with self.lock:
            self.visits += 1
            self.w += reward

    def get_untried_moves(self):
        with self.lock:
            return self.untried_actions[:]

    def select_child(self, c_param=math.sqrt(2)):
        with self.lock:
            return max(
                self.children,
                key=lambda child: child.w / child.visits + c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1))
            )


# ======================================
# RoomLog stub (replace with your class)
# ======================================
class RoomLog:
    def __init__(self, dataset, sede):
        self.dataset = dataset
        self.sede = sede
        self.idx_item = 0
        self.roomlog = {}

    def clone(self):
        cloned = RoomLog(self.dataset, self.sede)
        cloned.idx_item = self.idx_item
        cloned.roomlog = copy.deepcopy(self.roomlog)
        return cloned

    def get_legal_moves(self):
        # Return list of possible moves (stub)
        return [1, 2, 3, 4]

    def move(self, m):
        # Apply move (stub)
        self.idx_item += 1

    def terminal(self):
        # Terminal condition (stub)
        return self.idx_item >= 5

    def reward(self):
        # Example reward
        return random.random()


# ======================================
# MCTS (UCT) algorithm – single iteration
# ======================================
def UCT_search_iteration(root_node, state, c_param):
    node = root_node
    st = state.clone()

    # SELECTION
    while not st.terminal() and len(node.get_untried_moves()) == 0:
        node = node.select_child(c_param)
        st.move(node.move)

    # EXPANSION
    untried = node.get_untried_moves()
    if untried:
        m = random.choice(untried)
        st.move(m)
        node = node.add_child(m, st)

    # SIMULATION / ROLLOUT
    while not st.terminal():
        moves = st.get_legal_moves()
        st.move(random.choice(moves))

    # BACKPROPAGATION
    reward = st.reward()
    while node is not None:
        node.update(reward)
        node = node.parent


# ======================================
# Parallel Tree MCTS
# ======================================
def tree_parallel_UCT(state, iter_max=1000, c_param=math.sqrt(2), n_threads=4):
    root = Node(move=None, parent=None, untried_actions=state.get_legal_moves())

    def worker():
        local_state = state.clone()
        for _ in range(iter_max // n_threads):
            UCT_search_iteration(root, local_state, c_param)

    # Run multiple threads sharing the same tree
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(worker) for _ in range(n_threads)]
        for f in futures:
            f.result()  # wait for all threads to finish

    # Choose best move from root
    best_child = max(root.children, key=lambda c: c.visits) if root.children else None
    best_move = best_child.move if best_child else None
    return best_move, root, state


# ======================================
# DEMO
# ======================================
def demo():
    state = RoomLog(dataset="data", sede="A")
    move, root, _ = tree_parallel_UCT(state, iter_max=1000, n_threads=4)
    print("Best move:", move)
    for c in root.children:
        print(f"Move {c.move}: visits={c.visits}, w={c.w:.2f}")

if __name__ == "__main__":
    demo()


if __name__ == '__main__':
    # create argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sede", type=str, default="Ica")
    parser.add_argument("--iter_max", type=int, default=5000)
    parser.add_argument("--periodo_franja", type=str, default="1. Lun - Vie")
    args = parser.parse_args()
    # run_mcts(args.sede, args.periodo_franja, args.iter_max)


