# from sys import path
import datetime
import random
from dateutil.relativedelta import relativedelta
import numpy as np
from pathlib import Path
import pandas as pd
from itertools import product
from collections import namedtuple, deque
# from src.etl.load_json import DataLakeLoader, DataBase
import time
# import multiprocessing
import yaml
import os
import json
import multiprocess # pyright: ignore[reportMissingImports]

USER = os.getlogin()
# PATH_ASSIGNMENT = Path(f'C:/Users/{USER}/apis/02_assignment')
PATH_ASSIGNMENT = Path(f'./output')


class Rank:
    def __init__(self, periodo:int, sede:str, data_path, room_log_path, items_path, items_predict_path):
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
        collection = []
        data_path = Path(self.data_path)
        path_file =  (data_path / self.room_log_path/f'{self.periodo//100}'/f'room_log_{self.periodo}.json')
        with open(path_file, "r") as f:
            collection = json.load(f)
            results = [col for col in collection if col['SEDE'] == self.sede]
            assert len(results) == 1
            return results[0]
    
    def get_items(self) -> list:
        collection = []
        data_path = Path(self.data_path)
        path_file =  (data_path / self.items_path/f'{self.periodo//100}'/f'items_{self.periodo}.json')
        with open(path_file, "r") as f:
            collection = json.load(f)
            return [col for col in collection if col['SEDE'] == self.sede]
    
    def get_items_predict(self) -> list:
        collection = []
        data_path = Path(self.data_path)
        path_file =  (data_path / self.items_predict_path/f'{self.periodo//100}'/f'items_predict_{self.periodo}.json')
        with open(path_file, "r") as f:
            collection = json.load(f)
            return [col for col in collection if col['SEDE'] == self.sede]
    
    def get_reward_sedes(self) -> dict:
        collection = []
        data_path = Path(self.data_path)
        path_file =  (data_path / 'dim'/'rewards_sedes.json')
        with open(path_file, "r") as f:
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

    def get_aulas_disponibles(self, room_log:dict ,turnos:list, dias:list) -> list:
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
    
    def assign_aulas(self, room_log:dict ,turnos:list, dias:list, aula:str) -> dict:
        
        room_log_clone = room_log.copy()
        # results = self.client.db.room_log.find_one({'PERIODO':self.periodo, 'SEDE':self.sede})
        for idx_result, result in enumerate(room_log_clone['AULAS']):
            if result['AULA'] == aula:
                for idx_dia, dia in enumerate(result['DIAS']):
                    for idx_turno, turno in enumerate(dia['TURNOS']):
                        if (turno['TURNO'] in turnos) and (dia['DIA'] in dias):
                            room_log_clone['AULAS'][idx_result]['DIAS'][idx_dia]['TURNOS'][idx_turno]['AVAILABLE'] = 0

        return room_log_clone
    
    def test_assign_aulas(self, room_log:dict ,turnos:list, dias:list, aula:str):
        
        room_log_clone = room_log.copy()
        # results = self.client.db.room_log.find_one({'PERIODO':self.periodo, 'SEDE':self.sede})
        for idx_result, result in enumerate(room_log_clone['AULAS']):
            if result['AULA'] == aula:
                for idx_dia, dia in enumerate(result['DIAS']):
                    for idx_turno, turno in enumerate(dia['TURNOS']):
                        if (turno['TURNO'] in turnos) and (dia['DIA'] in dias):
                            print(room_log_clone['AULAS'][idx_result]['DIAS'][idx_dia]['TURNOS'][idx_turno]['AVAILABLE'])
                            
        # return room_log_clone
        
    def get_aulas_aforos(self, room_log:dict):
        aulas = []
        aforos = []
        for result in room_log['AULAS']:
            aulas.append(result['AULA'])
            aforos.append(result['AFORO'])        
        
        return (aulas, aforos)
    
    def get_simulation(self, items, room_log:dict, aulas, aforos, reward_sedes):
        room_log_clone = room_log.copy()
        # db = self.database.initialize_database()
        # items = self.client.db.items.find({'PERIODO':self.periodo})

        assignment = namedtuple('ASIGNACION',
                                ['PERIODO', 'SEDE', 'CODIGO_DE_CURSO', 'HORARIO',
                                 'N_AULA', 'FORECAST_ALUMN', 'AFORO', 'VAC_HAB', 'REWARD'])

        seed = np.random.randint(2000)
        random_state = np.random.RandomState(seed)
        
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
                rewards_niveles = [reward_sedes[aula][item['NIVEL']] for (aula, disponible) in zip(aulas, disponibles) if disponible]
                # niveles = niveles_consol[no_conflicts].copy()
                # aforo = aula_aforo[:, 1].copy()
                vac_hab = np.minimum(aforos_disponibles, item['VAC_ACAD_ESTANDAR'])
                
                saldos = vac_hab - item['FORECAST_ALUMN']
                # print(vac_hab, saldos, rewards_niveles)
                reward = np.where(
                    ((saldos >= 0) &
                     (saldos <= 2)), 5,
                    np.where(saldos > 2,
                             0, saldos * 2))
                reward += np.array(rewards_niveles)
                idxmax = np.argmax(reward)
                
                aula_max = aulas_disponibles[idxmax]
                room_log_clone = self.assign_aulas(room_log_clone, item['TURNOS'], item['DIAS'], aula_max)

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
    
    def run_simulation(self, parallel:bool=True):
        room_log = self.get_room_log()
        items = self.get_items_predict()
        reward_sedes = self.get_reward_sedes()

        aulas, aforos = self.get_aulas_aforos(room_log)
        if not parallel:
            return self.get_simulation(items, room_log, aulas, aforos, reward_sedes)
        else:
            return self.get_simulations_parallel(items, room_log, aulas, aforos, reward_sedes, 200)
    
    def get_simulations_parallel(self, items, room_log:dict, aulas, aforos, reward_sedes, num_simulations=200):
        """Run multiple simulations in parallel using Pool.map()"""

        # Create a list of identical arguments for each simulation
        args_list = [(items, room_log, aulas,aforos, reward_sedes)] * num_simulations

        # Use context manager for automatic cleanup
        start = time.time()
        # multiprocessing.Pool()

        with multiprocess.Pool(processes=6) as pool:
            # starmap is used because our function takes multiple arguments
            results = pool.starmap(self.get_simulation, args_list)

        best_data_frame = pd.DataFrame()
        best_reward = float('-inf')
        print(len(results))
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

        if not (PATH_ASSIGNMENT/f'{self.periodo}').exists():
            (PATH_ASSIGNMENT/f'{self.periodo}').mkdir()

        print(f"Complete {self.periodo} {self.sede}!!!", process_time)
        best_data_frame.to_excel(PATH_ASSIGNMENT/f'{self.periodo}/asignacion_{self.periodo}_{self.sede}.xlsx', index=False)



if __name__ == '__main__':
    pass