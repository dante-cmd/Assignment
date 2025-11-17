from .extract_excel_to_sqlite import ExcelExtractor
from .transform_sqlite_to_json import DataTransformer  
from .load_json_to_azure_blob import LakeLoader
from ..database.database import DatabaseManager
import yaml



if __name__ == '__main__':
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    excel_path = config['excel_path']
    db_path = config['db_path']
    log_path = config['log_path']
    periodo = config['periodo']
    n_periodos = config['n_periodos']
    full = config['full']
    
    drop_all = False
    db = DatabaseManager(db_path, drop_all)
    db.initialize_database()

    extractor = ExcelExtractor(periodo, n_periodos, full, db, excel_path)
    extractor.fecth_all()
    transformer = DataTransformer(db, log_path)
    transformer.transform_all()
    loader = LakeLoader()
    loader.load_all()
    