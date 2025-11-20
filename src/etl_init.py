import yaml
from etl.extract_excel_to_sqlite import ExcelExtractor
from etl.transform_sqlite_to_json import DataTransformer
# from etl.load_json import DataLakeLoader
from database.database import (get_sqlite_session, engine)
# , get_monty_client
import logging

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

    logging.basicConfig(
        filename=f'{log_path}/{log_file}', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    sqlite_session = get_sqlite_session()
    # monty_client = get_monty_client()

    extractor = ExcelExtractor(periodo_predict, ult_periodo, n_periodos, full,
                               sqlite_session, excel_path)
    # extractor.fetch_all()

    data_transformer = DataTransformer(
        periodo_predict, ult_periodo, n_periodos, full, 
        sqlite_session, engine, log_path, data_path,
        dim, items_predict, items, items_bim, room_log)
    data_transformer.transform_all()
    
    # data_lake = DataLakeLoader(monty_client, data_path, dim, items_predict, items, items_bim, room_log)
    # dias = ["LUN", "MAR"]
    # franjas = ["07:00 - 08:30", "08:45 - 10:15"]
    # result = get_available_aulas(
    #     sede="Ica",
    #     periodo=202501,
    #     dias=dias,
    #     franjas=franjas
    # )
    # for aula in result:
    #     print(aula)
    # data_transformer.get_room_log()
    # data_lake.susu()
    
    # extractor.fetch_all()

    # data_transformer.transform_all()
    
    # data_lake.load_all()
    
