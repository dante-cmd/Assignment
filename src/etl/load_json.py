from pathlib import Path
import json
from montydb import MontyClient # type: ignore
import yaml


class DataLakeLoader:
    def __init__(self,
                 client,
                 data_path: str,
                 dim_path: str,
                 items_predict_path: str,
                 items_path: str,
                 items_bim_path: str,
                 room_log_path:str):

        self.client = client
        self.data_path = data_path
        self.dim_path = dim_path
        self.items_path = items_path
        self.items_bim_path = items_bim_path
        self.items_predict_path = items_predict_path
        self.room_log_path = room_log_path

    def get_aulas(self):
        with open(f'{self.data_path}/{self.dim_path}/aulas.json', "r") as f:
            return json.load(f)

    def get_reward_sedes(self):
        with open(f'{self.data_path}/{self.dim_path}/rewards_sedes.json', "r") as f:
            return json.load(f)

    def dim_loader(self):
        client = self.client
        # db = self.database.initialize_database()
        aulas = self.get_aulas()
        reward_sedes = self.get_reward_sedes()
        client.db.aulas.insert_many(aulas)
        client.db.reward_sedes.insert_many(reward_sedes)

    def items_loader(self):
        client = self.client
        # self = DataLakeLoader("data", "dim", "items", "items_bim", "items_predict")
        #
        # db = self.database.initialize_database()
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path/self.items_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        client.db.items.insert_many(collection)

    def items_bim_loader(self):
        client = self.client
        # db = self.database.initialize_database()
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path / self.items_bim_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        client.db.items_bim.insert_many(collection)

    def items_predict_loader(self):
        client = self.client
        # db = self.database.initialize_database()
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path / self.items_predict_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        client.db.items_predict.insert_many(collection)

    def room_log_loader(self):
        client = self.client
        # db = self.database.initialize_database()
        collection = []
        data_path = Path(self.data_path)
        for path_file in (data_path / self.room_log_path).glob('*/*.json'):
            with open(path_file, "r") as f:
                collection.extend(json.load(f))
        client.db.room_log.insert_many(collection)

    def load_all(self):
        self.dim_loader()
        self.items_loader()
        self.items_bim_loader()
        self.items_predict_loader()
        self.room_log_loader()


if __name__ == '__main__':
    pass
