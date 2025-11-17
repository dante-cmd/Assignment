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
import multiprocessing
import yaml
import os

USER = os.getlogin()
PATH_ASSIGNMENT = Path(f'C:/Users/{USER}/apis/02_assignment')


class Rank:
    def __init__(self, client, periodo):
        self.client = client
        self.periodo = periodo

    def get_available_aulas(self, sede:str, periodo:int, dias:list, franjas:list):
        query = {
            "SEDE": sede,
            "PERIODO": periodo,
            "AULAS": {
                "$elemMatch": {
                    "DIAS": {
                        "$elemMatch": {
                            "DIA": {"$in": dias},
                            "TURNOS": {
                                "$elemMatch": {
                                    "FRANJA": {"$in": franjas},
                                    "AVAILABLE": 1
                                }
                            }
                        }
                    }
                }
            }
        }
        projection = {
            "_id": 0,
            "SEDE": 1,
            "PERIODO": 1,
            "AULAS.AULA": 1,
            "AULAS.DIAS": 1,
        }    

        return list(self.client.db.room_log.find(query, projection))
    
    # ✅ Example Usage
    # python
    # Copy code


    def get_simulation(self, room_log, aula_aforo):
        # db = self.database.initialize_database()
        items = self.client.db.items.find({'PERIODO':self.periodo})

        assignment = namedtuple('ASIGNACION',
                                ['PERIODO', 'SEDE', 'CODIGO_DE_CURSO', 'HORARIO',
                                 'N_AULA', 'FORECAST_ALUMN', 'AFORO', 'VAC_HAB', 'REWARD'])

        seed = np.random.randint(2000)
        random_state = np.random.RandomState(seed)
        idx_items = random_state.choice(range(0, len(items)), size=len(items), replace=False)
        items = items[idx_items].copy()
        # list(items)
        # aulas = aula_aforo['AULA']
        collection = []

        for item in items:

            (periodo, sede, codigo_curso,
             frecuencia, nivel, horario, alumn, _) = item
            # dias = self.env.mapping_frec_to_dias[frecuencia]
            # franjas = self.env.mapping_horario_to_franjas[horario]
            # vac_acad = self.env.mapping_nivel_to_vac_acad[nivel]

            no_conflicts = np.zeros_like(aula_aforo['AFORO'])
            no_conflicts = np.where(no_conflicts == 0, False, True)
            niveles = np.zeros_like(aula_aforo['AFORO'])
            for idx, aula in enumerate(aula_aforo['AULA']):
                no_conflict = all([True if len(room_log[aula]['PROGRAM'][dia][franja]) == 0 else False
                                   for dia, franja in product(item['DIAS'], item['TURNOS'])])
                niveles[idx] = self.env.mapping_aula_nivel_to_reward[aula][nivel]
                no_conflicts[idx] = no_conflict

            if sum(no_conflicts) == 0:
                collection.append(
                    assignment(
                        PERIODO=periodo,
                        SEDE=sede,
                        CODIGO_DE_CURSO=codigo_curso,
                        HORARIO=horario,
                        N_AULA=np.nan,
                        FORECAST_ALUMN=alumn,
                        REWARD=-20,
                        AFORO=0,  # np.NAN
                        VAC_HAB=0,  # np.NAN
                        # ID=uuid_packet
                    ))
            else:
                aforo_01 = aula_aforo['AFORO'][no_conflicts].copy()
                aulas_01 = aula_aforo['AULA'][no_conflicts].copy()
                niveles_01 = niveles[no_conflicts].copy()
                # niveles = niveles_consol[no_conflicts].copy()
                # aforo = aula_aforo[:, 1].copy()
                vac_hab = np.minimum(aforo_01, item['VAC_ACAD_ESTANDAR'])
                saldos = vac_hab - alumn
                reward = np.where(
                    ((saldos >= 0) &
                     (saldos <= 2)), 5,
                    np.where(saldos > 2,
                             0, saldos * 2))
                reward += niveles_01
                idxmax = np.argmax(reward)

                for dia, franja in product(item['DIAS'], item['TURNOS']):
                    room_log[aulas_01[idxmax]]['PROGRAM'][dia][franja] = codigo_curso

                collection.append(
                    assignment(
                        PERIODO=periodo,
                        SEDE=sede,
                        CODIGO_DE_CURSO=codigo_curso,
                        HORARIO=horario,
                        N_AULA=aulas_01[idxmax],
                        FORECAST_ALUMN=alumn,
                        REWARD=reward[idxmax],
                        AFORO=aforo_01[idxmax],
                        VAC_HAB=vac_hab[idxmax],
                        # ID=uuid_packet
                    ))

            # aulas
            # util = [1 if len(self.temp_room_log[aula]['PROGRAM'][dia][franja]) != 0 else 0
            #         for dia, franja in product(all_dias, franjas)]
            #
            # rate_util = sum(util) / len(util)
        return collection


class MonteCarlo:
    def __init__(self, env):
        self.env = env

    @staticmethod
    def get_aula_aforo(room_log: dict):
        aulas = []
        aforos = []
        for aula in room_log.keys():
            aulas.append(aula)
            aforos.append(room_log[aula]['AFORO'])
        return {'AULA': np.array(aulas), 'AFORO': np.array(aforos)}

    def get_simulation(self, items, room_log, aula_aforo):

        asignacion = namedtuple('ASIGNACION',
                                ['PERIODO', 'SEDE', 'CODIGO_DE_CURSO', 'HORARIO',
                                 'N_AULA', 'FORECAST_ALUMN', 'AFORO', 'VAC_HAB', 'REWARD'])

        seed = np.random.randint(2000)
        random_state = np.random.RandomState(seed)
        idx_items = random_state.choice(range(0, len(items)), size=len(items), replace=False)
        items = items[idx_items].copy()
        # list(items)
        # aulas = aula_aforo['AULA']
        collection_asignacion = []

        for item in items:
            (periodo, sede, codigo_curso,
             frecuencia, nivel, horario, alumn, _) = item
            dias = self.env.mapping_frec_to_dias[frecuencia]
            # all_dias = self.mapping_frec_to_all_dias[frecuencia]
            franjas = self.env.mapping_horario_to_franjas[horario]
            vac_acad = self.env.mapping_nivel_to_vac_acad[nivel]

            no_conflicts = np.zeros_like(aula_aforo['AFORO'])
            no_conflicts = np.where(no_conflicts == 0, False, True)

            niveles = np.zeros_like(aula_aforo['AFORO'])

            for idx, aula in enumerate(aula_aforo['AULA']):
                no_conflict = all([True if len(room_log[aula]['PROGRAM'][dia][franja]) == 0 else False
                                   for dia, franja in product(dias, franjas)])
                niveles[idx] = self.env.mapping_aula_nivel_to_reward[aula][nivel]
                no_conflicts[idx] = no_conflict

            if sum(no_conflicts) == 0:
                collection_asignacion.append(
                    asignacion(
                        PERIODO=periodo,
                        SEDE=sede,
                        CODIGO_DE_CURSO=codigo_curso,
                        HORARIO=horario,
                        N_AULA=np.NAN,
                        FORECAST_ALUMN=alumn,
                        REWARD=-20,
                        AFORO=0,  # np.NAN
                        VAC_HAB=0,  # np.NAN
                        # ID=uuid_packet
                    ))
            else:
                aforo_01 = aula_aforo['AFORO'][no_conflicts].copy()
                aulas_01 = aula_aforo['AULA'][no_conflicts].copy()
                niveles_01 = niveles[no_conflicts].copy()
                # niveles = niveles_consol[no_conflicts].copy()
                # aforo = aula_aforo[:, 1].copy()
                vac_hab = np.minimum(aforo_01, vac_acad)
                saldos = vac_hab - alumn
                reward = np.where(
                    ((saldos >= 0) &
                     (saldos <= 2)), 5,
                    np.where(saldos > 2,
                             0, saldos * 2))
                reward += niveles_01
                idxmax = np.argmax(reward)

                for dia, franja in product(dias, franjas):
                    room_log[aulas_01[idxmax]]['PROGRAM'][dia][franja] = codigo_curso

                collection_asignacion.append(
                    asignacion(
                        PERIODO=periodo,
                        SEDE=sede,
                        CODIGO_DE_CURSO=codigo_curso,
                        HORARIO=horario,
                        N_AULA=aulas_01[idxmax],
                        FORECAST_ALUMN=alumn,
                        REWARD=reward[idxmax],
                        AFORO=aforo_01[idxmax],
                        VAC_HAB=vac_hab[idxmax],
                        # ID=uuid_packet
                    ))

            # aulas
            # util = [1 if len(self.temp_room_log[aula]['PROGRAM'][dia][franja]) != 0 else 0
            #         for dia, franja in product(all_dias, franjas)]
            #
            # rate_util = sum(util) / len(util)
        return collection_asignacion

    def run_simulations_parallel(self, items, room_log, aula_aforo, num_simulations=200):
        """Run multiple simulations in parallel using Pool.map()"""

        # Create a list of identical arguments for each simulation
        args_list = [(items, room_log, aula_aforo)] * num_simulations

        # Use context manager for automatic cleanup

        with multiprocess.Pool(processes=6) as pool:
            # starmap is used because our function takes multiple arguments
            results = pool.starmap(self.get_simulation, args_list)

        return results

    def simulate(self, mode:str, path: None | str = None):
        # import time
        start = time.time()
        # self = monte_carlo
        items = get_items_forecast(self.env.dataset, self.env.periodo,
                                   self.env.sede, 'CANT_MATRICULADOS', mode, path)

        room_log = get_room_log_total(self.env.dataset,
                                      self.env.periodo,
                                      self.env.periodo,
                                      self.env.sede)

        aula_aforo = self.get_aula_aforo(room_log)

        simulations = self.run_simulations_parallel(items, room_log, aula_aforo, 2000)

        best_data_frame = pd.DataFrame()
        best_reward = float('-inf')
        for data in simulations:
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

        if not (PATH_ASSIGNMENT/f'{self.env.periodo}').exists():
            (PATH_ASSIGNMENT/f'{self.env.periodo}').mkdir()

        print(f"Complete {self.env.periodo} {self.env.sede}!!!", process_time)
        best_data_frame.to_excel(PATH_ASSIGNMENT/f'{self.env.periodo}/asignacion_{self.env.periodo}_{self.env.sede}.xlsx', index=False)


def run_monte_carlo(dataset, periodo, training_periodos, evaluation_periodos):
    # sedes = ['Surco', 'San Miguel', 'La Molina', 'Lima Centro',
    #          'Lima Norte Satélite', 'Miraflores',
    #          'Chincha', 'Chimbote', 'Pucallpa', 'Iquitos']
    sedes = ['Lima Norte Satélite']
    year = periodo//100

    path = fr"C:\Users\dante.toribio\OneDrive - ICPNA\01_MOD_PRG\Forecast Model\{year}\forecast_{periodo}.xlsx"

    for sede in sedes:
        env = Env(dataset, periodo,
                  sede, training_periodos, evaluation_periodos)
        monte_carlo = MonteCarlo(env)
        monte_carlo.simulate('Production', path)


def run_deep_q_learning(dataset, periodo, training_periodos, evaluation_periodos):
    # dataset=dataset_01
    # periodo=PERIODO
    # training_periodos=TRAINING_PERIODOS.copy()
    # evaluation_periodos=EVALUATION_PERIODOS.copy()
    sede = 'Ica'
    env = Env(dataset, periodo,
              sede, training_periodos, evaluation_periodos)
    agent = DQAgent01(env)
    # agent.load_best_model()
    try:
        epoch = 5000
        agent.train(epoch)
        # agent.train(50)
        torch.save({
            'epoch': epoch,
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            # 'loss': loss,
        }, agent.best_model_path)
        # torch.save(agent.model.state_dict(), agent.best_model_path)
        print('Save Model')
        # ll = agent.get_assignments(200)
        assignments = agent.get_assignments(200)
        df = pd.DataFrame(assignments)
        df.to_excel(PATH_ASSIGNMENT/f'assignments_{env.sede}_{env.periodo}.xlsx', index=False)

    except KeyboardInterrupt as p:
        torch.save(agent.model.state_dict(), agent.best_model_path)


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    periodo_predict = config['periodo_predict']
    ult_periodo = config['ult_periodo']
    n_periodos = config['n_periodos']
    full = config['full']
    db_path = config['db_path']
    excel_path = config['excel_path']
    log_path = config['log_path']
    log_file = config['log_file']
    data_path = config['data_path']
    dim = config['dim']
    items_predict = config['items_predict']
    items = config['items']
    items_bim = config['items_bim']
    room_log = config['room_log']


    db = DataBase()
    db.drop_dbs()
    # db = database.initialize_database()
    data_lake = DataLakeLoader(db, data_path, dim, items_predict, items, items_bim, room_log)
    data_lake.load_all()
    ll = db.initialize_database()
    qq = ll.room_log.find({'PERIODO':202506})

    llw = list(qq)
    for n in qq:
        print(n)



    db.close()

    # run_deep_q_learning(dataset_01, PERIODO, TRAINING_PERIODOS, EVALUATION_PERIODOS)

    # run_monte_carlo(dataset_01, PERIODO, TRAINING_PERIODOS, EVALUATION_PERIODOS)

    # env_01 = Env(dataset_01, PERIODO,'Ica', TRAINING_PERIODOS, EVALUATION_PERIODOS)
    # agent_01 = DQAgent01(env_01)
    # ll = agent_01.get_assignments(200)
    # agent_01.train(1000)
    # agent_01.load_best_model()
    # ll = agent_01.get_assignments()
    # ll[0], ll[1]
    # pp = pd.DataFrame(ll[0])
