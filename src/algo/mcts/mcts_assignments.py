import copy
import json
from pathlib import Path
import pandas as pd
import math, random, time
import argparse
import os
import multiprocessing
from src.etl.load_json import DataLakeLoader


class DataSet:
    def __init__(self, path):
        self.path = path
        self.dim_aulas = None
        self.dim_periodo_franja = None
        self.dim_frecuencia = None
        self.dim_horario = None
        self.items = None
        self.items_bimestral = None
        self.load_all()

    def read_json(self, name: str):
        with open(self.path / name, "r") as f:
            return json.load(f)

    def dim_aulas_loader(self):
        self.dim_aulas = self.read_json("dim_aulas.json")

    def dim_periodo_franja_loader(self):
        self.dim_periodo_franja = self.read_json("dim_periodo_franja.json")

    def dim_frecuencia_loader(self):
        self.dim_frecuencia = self.read_json("dim_frecuencia.json")

    def dim_horario_loader(self):
        self.dim_horario = self.read_json("dim_horario.json")

    def items_loader(self):
        self.items = self.read_json("items.json")

    def items_bimestral_loader(self):
        self.items_bimestral = self.read_json("items_bimestral.json")

    def load_all(self):
        self.dim_horario_loader()
        self.dim_aulas_loader()
        self.dim_frecuencia_loader()
        self.dim_periodo_franja_loader()
        self.items_loader()
        self.items_bimestral_loader()


class RoomLog:
    def __init__(self, dataset, sede: str, periodo_franja: str):
        self.dataset = dataset
        self.sede = sede
        self.periodo_franja = periodo_franja
        self.dim_aulas = dataset.dim_aulas[sede].copy()
        self.dim_periodo_franja = dataset.dim_periodo_franja.copy()
        self.dim_frecuencia = dataset.dim_frecuencia.copy()
        self.dim_horario = dataset.dim_horario.copy()
        # self.items = dataset.items[sede].copy()
        self.items = self.get_items(sede)
        self.items_bimestral = self.get_items_bimestral(sede)
        self.roomlog = self.get_roomlog()
        self.idx_item = 0
        self.n_items = len(self.items)
        self.n_aulas = len(self.roomlog.keys())

    def get_items_bimestral(self, sede: str):
        if sede not in self.dataset.items_bimestral:
            return []
        return self.dataset.items_bimestral[sede]

    def get_items(self, sede: str):
        collection = []
        items = self.dataset.items[sede].copy()
        for item in items:
            if self.dim_frecuencia[item['FRECUENCIA']]['PERIODO_FRANJA'] == self.periodo_franja:
                collection.append(item)
        return collection
    
    def get_roomlog(self):
        # self = env_01
        aulas = self.dim_aulas['AULA']
        room_log = {}
        for aula in aulas:
            room_log[aula] = {}
            for periodo_franja in self.dim_periodo_franja.keys():
                franjas = self.dim_periodo_franja[periodo_franja]['FRANJAS']
                dias = self.dim_periodo_franja[periodo_franja]['DIAS']
                for dia in dias:
                    room_log[aula][dia] = {}
                    for franja in franjas:
                        room_log[aula][dia][franja] = 0

        for item in self.items_bimestral:
            assert self.dim_frecuencia[item['FRECUENCIA']]['PERIODO_FRANJA'] == '2. Sab'
            
            dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
            franjas = self.dim_horario[item['HORARIO']]
            for dia in dias:
                for franja in franjas:
                    room_log[item['AULA']][dia][franja] = 1
        return room_log

    def clone(self):
        g = RoomLog(self.dataset, self.sede, self.periodo_franja)
        g.idx_item = self.idx_item
        g.roomlog = copy.deepcopy(self.roomlog.copy())
        return g

    def step(self, action: int):
        if self.idx_item >= self.n_items:
            return None

        # Get item
        item = self.items[self.idx_item].copy()

        # Get aula and aforo for action
        aula = self.dim_aulas['AULA'][action]
        aforo = self.dim_aulas['AFORO'][action]
        roomlog = self.roomlog.copy()

        # Reward
        if (aforo - item['ALUMN']) < 0:
            reward = aforo - item['ALUMN'] - 2

        elif ((aforo - item['ALUMN']) >= 0) and ((aforo - item['ALUMN']) <= 2):
            reward = 1 + (item['ALUMN'] / aforo)

        else:
            reward = 0

        # Update roomlog
        dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        franjas = self.dim_horario[item['HORARIO']]

        for dia in dias:
            for franja in franjas:
                roomlog[aula][dia][franja] = 1

        self.roomlog = roomlog.copy()

        # Update idx_item
        self.idx_item += 1
        return reward

    def get_available_actions(self):
        if self.idx_item >= self.n_items:
            return []

        item = self.items[self.idx_item].copy()
        aulas = self.dim_aulas['AULA']
        roomlog = self.roomlog.copy()
        dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        franjas = self.dim_horario[item['HORARIO']]

        available = []
        for idx, aula in enumerate(aulas):
            conflict = []
            for dia in dias:
                for franja in franjas:
                    conflict.append(True if roomlog[aula][dia][franja] == 1 else False)
            if not any(conflict):
                available.append(idx)
        return available

    def is_terminal(self):
        return (self.idx_item >= self.n_items) | (len(self.get_available_actions()) == 0)


class Node:
    def __init__(self, move=None, parent=None, untried_actions=None,
                 available_actions=True):
        self.move = move                  # the move that led to this node (from parent)
        self.parent = parent              # parent node
        self.children = []                # list of child nodes
        self.w = 0.0                      # number of wins
        self.visits = 0                   # visit count
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
    cloned_state = RoomLog(state.dataset, state.sede, state.periodo_franja)
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

    iters_per_worker = iter_max // n_cores
    roots = []
    
    with multiprocessing.Pool(processes=n_cores) as pool:
        # run mcts
        results = pool.map(UCT_worker, [(state, iters_per_worker, c_param) for _ in range(n_cores)])
        for result in results:
            roots.append(result)

    # with concurrent.futures.ProcessPoolExecutor(
    #         max_workers=max_workers) as executor:
    #     futures = [
    #         executor.submit(
    #             UCT_worker, state, iters_per_worker, c_param)
    #         for _ in range(max_workers)]
    # 
    #     for f in concurrent.futures.as_completed(futures):
    #         roots.append(f.result())

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


def run_mcts(sede: str, periodo_franja: str, iter_max: int = 5000):
    # --- Demo: use UCT on an empty RoomLog board ---
    path = Path("project")
    data = DataSet(path)
    state = RoomLog(data, sede, periodo_franja)
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
            result['ASSIGNMENTS'] = {'AULA':state.dim_aulas['AULA'][move],
                                     'AFORO':state.dim_aulas['AFORO'][move]}
            state.step(move)
        aulas.append(result)
        duration = time.time() - start_time
        total_duration = time.time() - total_time
        duration_per_item =total_duration / (state.idx_item)
        remaining_time = duration_per_item * (len(state.items) - state.idx_item)
        to_time = lambda x: time.strftime('%H:%M:%S', time.gmtime(x))
        print(f"Total duration: {to_time(total_duration)} | Actual duration: {to_time(duration)} | Remaining time: {to_time(remaining_time)} | {state.idx_item}/{len(state.items)}", end="\r")
    print()
    # return aulas
    
    df = pd.DataFrame(aulas)
    Path('../../output').mkdir(exist_ok=True)

    df.to_excel('output/assignments_{}.xlsx'.format(sede), index=False)

# multiprocessing
# def run_mcts_parallel(sede: str, iter_max: int = 5000):
#     periodos_franjas = ['1. Lun - Vie', '2. Sab']
#     # Number of cores
#     n_cores = os.cpu_count()
#     
#     with multiprocessing.Pool(processes=n_cores) as pool:
#         # run mcts
#         results = pool.map(run_mcts, [(sede, periodo_franja, iter_max) for periodo_franja in periodos_franjas])
#     # return results
#     return results
# 


if __name__ == '__main__':
    # create argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sede", type=str, default="Ica")
    parser.add_argument("--iter_max", type=int, default=5000)
    parser.add_argument("--periodo_franja", type=str, default="1. Lun - Vie")
    args = parser.parse_args()
    run_mcts(args.sede, args.periodo_franja, args.iter_max)


