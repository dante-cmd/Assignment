from algo.hierarchy.rank import Rank
from algo.data import Data
from algo.mcts.mcts_assignments_v2 import run_mcts
from database.database import get_sqlite_session, engine
# get_monty_client
import time
import yaml
from datetime import datetime


if __name__ == '__main__':
    start = time.time()

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
    # monty_client = get_monty_client()
    periodo = 202511
    sede = 'Ica'
    # rank = Rank(periodo, sede, data_path, room_log, items, items_predict)
    # dias_ = ["LUN", "MAR"]
    # franjas_ = ["07:00 - 08:30", "08:45 - 10:15"]

    
    # result = rank.run_simulation(True, 5)

    run_mcts(periodo, sede, data_path, room_log, items, items_predict, 5000)

    
    # result = rank.get_room_log()
    # print(len(result))
    # result = rank.assign_aulas(room_log, franjas_ ,dias_, '101')
    # print(rank.test_assign_aulas(room_log, franjas_ ,dias_, '101'))
    # ww = rank.get_aulas_disponibles(result, franjas_ ,dias_)
    # ww_2 = rank.get_aulas_disponibles(room_log, franjas_ ,dias_)
    # print("get result",time.time() - start, datetime.today())
    # print(ww, ww_2)
    # print(result, datetime.today())
    # import pandas as pd
    # df = pd.DataFrame(result)
    # df.to_excel('susu.xlsx', index=False)
   # 
                
    # print(datetime.today(), result)
