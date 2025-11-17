from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import yaml

def get_n_lags(periodo: int, n: int):
    periodo_date = datetime.strptime(str(periodo), '%Y%m')
    return int((periodo_date - relativedelta(months=n)).strftime('%Y%m'))


def get_last_n_periodos(periodo: int, n: int):
    # periodo_date = datetime.strptime(str(periodo), '%Y%m')
    return [get_n_lags(periodo, lag) for lag in range(n)]

# clean logs and csv files
def clean_logs(log_path: str, log_file: str):
    # with open(log_path + '/' + log_file, 'r') as f:
    #     lines = f.readlines()
    with open(log_path + '/' + log_file, 'w') as f:
        # for line in lines:
        #     if 'INFO' in line:
        f.write('')

    # clean csv files
    for file in os.listdir(log_path):
        if file.endswith('.csv'):
            os.remove(os.path.join(log_path, file))


if __name__ == '__main__':
    pass
    # with open('config.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # log_path = config['log_path']
    # log_file = config['log_file']
    # clean_logs(log_path, log_file)
