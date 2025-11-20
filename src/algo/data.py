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
# import multiprocess  # pyright: ignore[reportMissingImports]


class Data:
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

    def get_aulas(self):
        data_path = Path(self.data_path)
        path_file = (
                data_path / self.room_log_path / f'{self.periodo // 100}' / f'room_log_{self.periodo}.json')

        with open(path_file, "r", encoding='utf-8') as f:
            collection = json.load(f)
            results = [col for col in collection if col['SEDE'] == self.sede]
            # print(results)
            # == ll
            assert len(results) == 1
            room_log = results[0]
            aulas = []
            aforos = []
            for item in room_log['AULAS']:
                aulas.append(item['AULA'])
                aforos.append(item['AFORO'])

            return aulas, aforos

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

