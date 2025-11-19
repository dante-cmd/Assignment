import copy
import json
from pathlib import Path
import pandas as pd
import math, random, time
import argparse
import os
import multiprocessing
from src.etl.load_json import DataLakeLoader


# from sys import path
import datetime
import random
from dateutil.relativedelta import relativedelta
import numpy as np
from pathlib import Path
import pandas as pd
import re
from itertools import product
from collections import namedtuple, deque
# from src.etl.load_json import DataLakeLoader, DataBase
import time
# import multiprocessing
import yaml
import os
import json
import multiprocess  # pyright: ignore[reportMissingImports]
from utils import assign_aulas
from algo.data import Data


USER = os.getlogin()
# PATH_ASSIGNMENT = Path(f'C:/Users/{USER}/apis/02_assignment')
PATH_ASSIGNMENT = Path(f'./output')


class Rank:
    def __init__(self, periodo: int, sede: str, data_path, room_log_path,
                 items_path, items_predict_path):
        # self.client = client
        self.data_path = data_path
        self.room_log_path = room_log_path
        self.items_path = items_path
        self.items_predict_path = items_predict_path
        self.periodo = periodo
        self.sede = sede

    def get_room_log(self) -> dict:
        # client = self.client
        # db = self.database.initialize_database()
        # collection = []
        data_path = Path(self.data_path)
        path_file = (data_path / self.room_log_path / f'{self.periodo // 100}' / f'room_log_{self.periodo}.json')
        with open(path_file, "r", encoding='utf-8') as f:
            collection = json.load(f)
            results = [col for col in collection if col['SEDE'] == self.sede]
            # print(results)
            # == ll
            assert len(results) == 1
            return results[0]

    def get_items(self) -> list:
        # collection = []
        data_path = Path(self.data_path)
        path_file = (data_path / self.items_path / f'{self.periodo // 100}' / f'items_{self.periodo}.json')
        with open(path_file, "r", encoding='utf-8') as f:
            collection = json.load(f)
            return [col for col in collection if col['SEDE'] == self.sede]

    def get_items_predict(self) -> list:
        collection = []
        data_path = Path(self.data_path)
        path_file = (
                data_path / self.items_predict_path / f'{self.periodo // 100}' / f'items_predict_{self.periodo}.json')
        with open(path_file, "r", encoding='utf-8') as f:
            collection = json.load(f)
            return [col for col in collection if col['SEDE'] == self.sede]

    def get_reward_sedes(self) -> dict:
        # collection = []
        data_path = Path(self.data_path)
        path_file = (data_path / 'dim' / 'rewards_sedes.json')
        with open(path_file, "r", encoding='utf-8') as f:
            collection = json.load(f)
            docs = [col for col in collection if col['SEDE'] == self.sede]
        # docs = self.client.db.reward_sedes.find({'SEDE':self.sede})
        result = {}

        for doc in docs:
            aula = doc["N_AULA"]
            nivel = doc["NIVEL"]
            reward = doc["REWARD"]

            if aula not in result:
                result[aula] = {}

            result[aula][nivel] = reward
        return result

    @staticmethod
    def get_aulas_disponibles(room_log: dict, turnos: list, dias: list) -> list:
        # results = self.client.db.room_log.find_one({'PERIODO':self.periodo, 'SEDE':self.sede})
        disponibles = []
        for result in room_log['AULAS']:
            collection = []
            for dia in result['DIAS']:
                for turno in dia['TURNOS']:
                    if (turno['TURNO'] in turnos) and (dia['DIA'] in dias):
                        if (turno['AVAILABLE'] == 1):
                            collection.append(True)
                        else:
                            collection.append(False)
            if all(collection):
                disponibles.append(True)
            else:
                disponibles.append(False)
        return disponibles

    @staticmethod
    def get_aulas_aforos(room_log: dict):
        aulas = []
        aforos = []
        for result in room_log['AULAS']:
            aulas.append(result['AULA'])
            aforos.append(result['AFORO'])

        return aulas, aforos

    def get_simulation(self, items, room_log: dict, aulas, aforos, reward_sedes):
        room_log_clone = room_log.copy()
        # db = self.database.initialize_database()
        # items = self.client.db.items.find({'PERIODO':self.periodo})

        assignment = namedtuple('ASIGNACION',
                                ['PERIODO', 'SEDE', 'CODIGO_DE_CURSO', 'HORARIO',
                                 'N_AULA', 'FORECAST_ALUMN', 'AFORO', 'VAC_HAB', 'REWARD'])

        # seed = np.random.randint(2000)
        # random_state = np.random.RandomState(seed)

        # idx_items = random_state.choice(range(0, len(items)), size=len(items), replace=False)
        # items = items[idx_items].copy()

        collection = []

        for item in items:
            disponibles = self.get_aulas_disponibles(
                room_log_clone, item['TURNOS'], item['DIAS'])

            if sum(disponibles) == 0:
                collection.append(
                    assignment(
                        PERIODO=item['PERIODO'],
                        SEDE=item['SEDE'],
                        CODIGO_DE_CURSO=item['CURSO_ACTUAL'],
                        HORARIO=item['HORARIO'],
                        N_AULA=np.nan,
                        FORECAST_ALUMN=item['FORECAST_ALUMN'],
                        REWARD=-20,
                        AFORO=0,  # np.NAN
                        VAC_HAB=0,  # np.NAN
                        # ID=uuid_packet
                    ))
            else:
                aforos_disponibles = [aforo for (aforo, disponible) in zip(aforos, disponibles) if disponible]
                aulas_disponibles = [aula for (aula, disponible) in zip(aulas, disponibles) if disponible]
                rewards_niveles = [reward_sedes[aula][item['NIVEL']] for (aula, disponible) in zip(aulas, disponibles)
                                   if disponible]

                vac_hab = np.minimum(aforos_disponibles, item['VAC_ACAD_ESTANDAR'])

                saldos = vac_hab - item['FORECAST_ALUMN']
                one_hot = np.where(np.abs(saldos) <= 2, 0, 1)
                # print(vac_hab, saldos, rewards_niveles)
                reward = np.where(
                    ((saldos >= 0) &
                     (saldos <= 2)), 2,
                    np.where(saldos > 2,
                             0, saldos * 2)) * one_hot
                reward += np.array(rewards_niveles)
                idxmax = np.argmax(reward)

                aula_max = aulas_disponibles[idxmax]
                room_log_clone = assign_aulas(
                    room_log_clone, item['TURNOS'], item['DIAS'], aula_max)

                # for dia, franja in product(item['DIAS'], item['TURNOS']):
                #     room_log_clone[aulas_disponibles[idxmax]]['PROGRAM'][dia][franja] = item['CURSO_ACTUAL']

                collection.append(
                    assignment(
                        PERIODO=item['PERIODO'],
                        SEDE=item['SEDE'],
                        CODIGO_DE_CURSO=item['CURSO_ACTUAL'],
                        HORARIO=item['HORARIO'],
                        N_AULA=aulas_disponibles[idxmax],
                        FORECAST_ALUMN=item['FORECAST_ALUMN'],
                        REWARD=reward[idxmax],
                        AFORO=aforos_disponibles[idxmax],
                        VAC_HAB=vac_hab[idxmax],
                        # ID=uuid_packet
                    ))
        # print(collection)
        return collection

    def get_simulations_parallel(self, items, room_log: dict, aulas, aforos, reward_sedes,
                                 num_simulations=200):
        """Run multiple simulations in parallel using Pool.map()"""

        # Create a list of identical arguments for each simulation
        args_list = [(items, room_log, aulas, aforos, reward_sedes)] * num_simulations

        # Use context manager for automatic cleanup
        start = time.time()
        # multiprocessing.Pool()

        with multiprocess.Pool(processes=6) as pool:
            # starmap is used because our function takes multiple arguments
            results = pool.starmap(self.get_simulation, args_list)

        best_data_frame = pd.DataFrame()
        best_reward = float('-inf')
        for data in results:
            data_frame = pd.DataFrame(data)
            reward = data_frame['REWARD'].sum()
            if reward > best_reward:
                best_reward = reward
                best_data_frame = data_frame.copy()

        end = time.time()
        gap = end - start
        hours = int(gap // 3600)
        minutes = int((gap % 3600) // 60)
        seconds = int(round((gap % 3600) % 60, 0))
        if hours > 24:
            process_time = "Superior a las 24h"
        else:
            process_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        if not (PATH_ASSIGNMENT / f'{self.periodo}').exists():
            (PATH_ASSIGNMENT / f'{self.periodo}').mkdir()

        print("☰"*(len(self.sede) + 10))
        print(f"✅ Periodo: {self.periodo}")
        print(f"📌 Sede: {self.sede}")
        print(f"⏱ Time: {process_time}")
        print("☰"*(len(self.sede) + 10))
        best_data_frame.to_excel(PATH_ASSIGNMENT / f'{self.periodo}/asignacion_{self.periodo}_{self.sede}.xlsx',
                                 index=False)

    def run_simulation(self, parallel: bool = True, num_simulations: None | int = 200):
        room_log = self.get_room_log()
        items = self.get_items_predict()
        reward_sedes = self.get_reward_sedes()

        aulas, aforos = self.get_aulas_aforos(room_log=room_log)
        if not parallel:
            print('No parallel')
            return self.get_simulation(
                items, room_log, aulas, aforos, reward_sedes)
        else:
            print(f'Parallel = {num_simulations}')
            return self.get_simulations_parallel(
                items, room_log, aulas, aforos, reward_sedes, num_simulations)


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


class RoomLog(Data):

    def __init__(self,  periodo: int, sede: str, data_path, room_log_path,
                 items_path, items_predict_path):
        super().__init__(
            periodo, sede, data_path, room_log_path,
            items_path, items_predict_path)

        # def __init__(self, dataset, sede: str, periodo_franja: str):
        # self.dataset = dataset
        self.sede = sede
        self.periodo = periodo
        self.data_path = data_path
        self.room_log_path = room_log_path
        self.items_path = items_path
        self.items_predict_path = items_predict_path
        # self.periodo_franja = periodo_franja
        # self.dim_aulas = dataset.dim_aulas[sede].copy()
        # self.dim_periodo_franja = dataset.dim_periodo_franja.copy()
        # self.dim_frecuencia = dataset.dim_frecuencia.copy()
        # self.dim_horario = dataset.dim_horario.copy()
        # self.items = dataset.items[sede].copy()
        # self.items_bimestral = self.get_items_bimestral(sede)
        self.roomlog = self.get_room_log()
        self.aulas, self.aforos = self.get_aulas_aforos(self.roomlog)
        self.items = self.get_items()
        self.idx_item = 0
        self.n_items = len(self.items)
        self.n_aulas = len(self.aulas)

    # def get_items_bimestral(self, sede: str):
    #     if sede not in self.dataset.items_bimestral:
    #         return []
    #     return self.dataset.items_bimestral[sede]

    # def get_items(self, sede: str):
    #     collection = []
    #     items = self.dataset.items[sede].copy()
    #     for item in items:
    #         if self.dim_frecuencia[item['FRECUENCIA']]['PERIODO_FRANJA'] == self.periodo_franja:
    #             collection.append(item)
    #     return collection

    # def get_roomlog(self):
    #     # self = env_01
    #     aulas = self.dim_aulas['AULA']
    #     room_log = {}
    #     for aula in aulas:
    #         room_log[aula] = {}
    #         for periodo_franja in self.dim_periodo_franja.keys():
    #             franjas = self.dim_periodo_franja[periodo_franja]['FRANJAS']
    #             dias = self.dim_periodo_franja[periodo_franja]['DIAS']
    #             for dia in dias:
    #                 room_log[aula][dia] = {}
    #                 for franja in franjas:
    #                     room_log[aula][dia][franja] = 0
#
    #     for item in self.items_bimestral:
    #         assert self.dim_frecuencia[item['FRECUENCIA']]['PERIODO_FRANJA'] == '2. Sab'
#
    #         dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
    #         franjas = self.dim_horario[item['HORARIO']]
    #         for dia in dias:
    #             for franja in franjas:
    #                 room_log[item['AULA']][dia][franja] = 1
    #     return room_log

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
        if (aforo - item['ALUMN']) < 0:
            reward = aforo - item['ALUMN'] - 2

        elif ((aforo - item['ALUMN']) >= 0) and ((aforo - item['ALUMN']) <= 2):
            reward = 1 + (item['ALUMN'] / aforo)

        else:
            reward = 0

        # Update roomlog
        dias = item['DIAS'] # self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        franjas = item['TURNOS'] # self.dim_horario[item['HORARIO']]

        self.roomlog = self.assign_aulas(roomlog, franjas, dias, aula)

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
            result['ASSIGNMENTS'] = {'AULA': state.dim_aulas['AULA'][move],
                                     'AFORO': state.dim_aulas['AFORO'][move]}
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


