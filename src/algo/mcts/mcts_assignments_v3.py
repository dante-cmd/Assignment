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
        self.reward_sedes = self.get_reward_sedes()
        self.idx_item = 0
        self.n_items = len(self.items)
        self.n_aulas = len(self.aulas)

    def clone(self):
        g = RoomLog(
            self.periodo, self.sede, self.data_path, self.room_log_path,
            self.items_path, self.items_predict_path)
        g.items = copy.deepcopy(self.items.copy())
        g.n_items = len(g.items)
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
        vac_hab = min(item['VAC_ACAD_ESTANDAR'], aforo)

        # Reward
        if (vac_hab - item['FORECAST_ALUMN']) < 0:
            reward = (vac_hab - item['FORECAST_ALUMN']) * 2

        elif ((vac_hab - item['FORECAST_ALUMN']) >= 0) and ((vac_hab - item['FORECAST_ALUMN']) <= 2):
            reward = 1 + (item['FORECAST_ALUMN'] / vac_hab)

        else:
            reward = 0
        
        reward += self.reward_sedes[aula][item['NIVEL']]

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
        # aulas = self.aulas.copy()
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
        self.move = move  # the move that led to this node (from parent)
        self.parent = parent  # parent node
        self.children = []  # list of child nodes
        self.w = 0.0  # number of wins
        self.visits = 0  # visit count
        self.untried_actions = [] if untried_actions is None else untried_actions.copy()  # moves not expanded yet
        # self.available_actions = available_actions

    def uct_select_child(self, c_param=math.sqrt(2)):
        # Select a child according to UCT (upper confidence bound applied to trees)
        # If a child has 0 visits we consider its UCT value infinite to ensure it's visited.
        best = max(self.children, key=lambda child: (
            float('inf') if child.visits == 0 else
            (child.w / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
        ))
        return best

    def add_child(self, move, untried_actions):
        child = Node(move=move, parent=self, untried_actions=untried_actions)
        # available_actions=available_actions
        self.untried_actions.remove(move)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.w += reward


def UCT(state: RoomLog, available_actions: list, iter_max=5000, c_param=math.sqrt(2)):
    # state: the representation of RoomLog
    # iter_max: number of iterations
    # c_param: parameter for UCT
    root_node = Node(move=None,
                     parent=None,
                     untried_actions=available_actions)

    node = root_node
    clone = state.clone()

    for i in range(iter_max):
        # i = 1
        rewards = []
        
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
                                  untried_actions=available_children)

        # 3. Simulation: play randomly until the game ends
        while not clone.is_terminal():
            possible_moves = clone.get_available_actions()
            reward = clone.step(random.choice(possible_moves))
            rewards.append(reward)

        # 4. Backpropagation: update node statistics with simulation result
        # n_items = min(max(clone.idx_item, 1), clone.n_items)
        n_items = clone.n_items

        # Update root node
        while True: # node is not None:
            node.update(sum(rewards) / n_items)
            parent = node.parent
            if parent is None:
                break
            node = parent
        # At the end of the iteration, node reaches the root node

    # Return the move that was most visited
    root_node = node
    print([child.visits for child  in root_node.children])
    assert isinstance(root_node.children, list)
    best_child = max(root_node.children, key=lambda c: c.visits)
    clone.idx_item = state.idx_item
    clone.roomlog = copy.deepcopy(state.roomlog.copy())

    return best_child.move, root_node, clone  # also return root node so the caller can inspect children statistics


def UCT_worker(args):
    state, available_actions, iter_max, c_param = args
    # Clone state for isolation
    cloned_state = state.clone()
    # RoomLog(state.dataset, state.sede, state.periodo_franja)
    cloned_state.idx_item = state.idx_item
    cloned_state.roomlog = copy.deepcopy(state.roomlog)

    move, root, _ = UCT(cloned_state, iter_max=iter_max, c_param=c_param, available_actions=available_actions)
    return root


def parallel_UCT(state, available_actions: list, iter_max=5000, c_param=math.sqrt(2)):
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
        results = pool.map(UCT_worker, [(state, available_actions, iters_per_worker, c_param) for _ in range(n_cores)])
        for result in results:
            roots.append(result)

    # merge children statistics from workers
    merged_root = Node(move=None, parent=None,
                       untried_actions=available_actions)
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


def mcts(periodo, sede, data_path, room_log_path,items_path, items_predict_path, periodo_franja:str|None, iter_max:int):
    # --- Demo: use UCT on an empty RoomLog board ---
    # path = Path("project")
    # data = DataSet(path)
    # periodo, sede, data_path, room_log_path,items_path, items_predict_path
    
    start = time.time()
    timer = time.time()
    state = RoomLog(periodo, sede, data_path, room_log_path,items_path, items_predict_path)
    if periodo_franja is not None:
        state.items = state.filter_items(state.items, periodo_franja)
        state.n_items = len(state.items)
    
    aulas = []
    print("Periodo Franja: ",periodo_franja)
    print("Time to load state:",time.time() - timer)
    # total_time = time.time()
    print(len(state.items))
    while state.idx_item < len(state.items):
        print("----------------------------------")
        print("IDX:", state.idx_item)
        timer = time.time()
        result = copy.deepcopy(state.items[state.idx_item].copy())
        print("Time to load item:",time.time() - timer)
        timer = time.time()
        available_actions = state.get_available_actions()
        print("Time to get available actions:",time.time() - timer)
        timer = time.time()
        if len(available_actions) == 0:
            result['ASSIGNMENTS.AULA'] = None
            result['ASSIGNMENTS.AFORO'] = None
            state.idx_item += 1
        else:
            # move, root_node, state = parallel_UCT(state, iter_max=iter_max)
            timer = time.time()
            move, _, state = UCT(state, available_actions=available_actions, iter_max=iter_max)   # 2000 rollouts from empty board
            print("Time to UCT:",time.time() - timer)
            result['ASSIGNMENTS.AULA'] = state.aulas[move]
            result['ASSIGNMENTS.AFORO'] = state.aforos[move]
            state.step(move)
        aulas.append(result)
    end = time.time()
    gap = end - start
    hours = int(gap // 3600)
    minutes = int((gap % 3600) // 60)
    seconds = int(round((gap % 3600) % 60, 0))
    if hours > 24:
        process_time = "Superior a las 24h"
    else:
        process_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    print(f"⏱ Time: {process_time}")
    return aulas
  

def run_mcts(periodo, sede, data_path, room_log_path,items_path, items_predict_path,  iter_max: int = 5000):
      
    n_cores = os.cpu_count()

    assert isinstance(n_cores, int)

    # iters_per_worker = iter_max // n_cores
    args_list = [(periodo, sede, data_path, room_log_path,items_path, items_predict_path, periodo_franja, iter_max) 
                 for periodo_franja in ['1. Lun - Vie', '2. Sab']]

    aulas = []
    with multiprocessing.Pool(processes=n_cores) as pool:
        # run mcts
        results = pool.starmap(mcts, args_list)
        for result in results:
            aulas.extend(result)

    print("☰"*(len(sede) + 10))
    print(f"✅ Periodo: {periodo}")
    print(f"📌 Sede: {sede}")
    # print(f"⏱ Time: {process_time}")
    print("☰"*(len(sede) + 10))
    
    # print(aulas)
    best_data_frame = pd.DataFrame(aulas)
    best_data_frame.to_excel(PATH_ASSIGNMENT / f'{periodo}/asignacion_{periodo}_{sede}.xlsx',
                                 index=False)
    # Path('../../output').mkdir(exist_ok=True)
    # df.to_excel('output/assignments_{}.xlsx'.format(sede), index=False)


if __name__ == '__main__':
    # create argparse
    # demo()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sede", type=str, default="Ica")
    parser.add_argument("--iter_max", type=int, default=5000)
    parser.add_argument("--periodo_franja", type=str, default="1. Lun - Vie")
    args = parser.parse_args()
    # run_mcts(args.sede, args.periodo_franja, args.iter_max)

