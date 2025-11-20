from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union
import os
from unidecode import unidecode
from database.utils import insert_dataframe, insert_dataframe_by_periodo, query_to_dataframe
from database.database import get_sqlite_session #, get_monty_client
import yaml
from utils import get_last_n_periodos


class ExcelExtractor:
    def __init__(
        self, periodo_predict: int|None, ult_periodo: int|None, n_periodos: int|None, full: bool, 
        session, excel_path: str):
        # self.db = db
        self.excel_path = excel_path
        self.periodo_predict = periodo_predict
        self.ult_periodo = ult_periodo
        self.n_periodos = n_periodos
        self.full = full
        self.session = session

    @staticmethod
    def filter_prog_acad(df_prog_acad: pd.DataFrame):
        es_cancelado = df_prog_acad['ESTADO'] == 'Cur. Cancelado'
        es_vecor = df_prog_acad['SEDE'] == 'VECOR'
        es_corporate = df_prog_acad['NIVEL'] == 'Corporate'
        end_with_pv = df_prog_acad['CODIGO_DE_CURSO'].str.contains('.+[pP][Vv]$')
        end_with_p = df_prog_acad['CODIGO_DE_CURSO'].str.contains('.+[pP]$')
        break_periodo = df_prog_acad['PERIODO'] < 202409

        return (
            df_prog_acad[(~(es_cancelado | es_vecor | ((end_with_pv | end_with_p) & break_periodo) |
                                 (end_with_pv & ~break_periodo) | es_corporate)) ].copy())
    
    def fetch_fact_prog_acad(self):
        raw_path = Path(self.excel_path)
        collection = []
        assert isinstance(self.full, bool)
       
        if not self.full:
            assert isinstance(self.ult_periodo, int)
            assert isinstance(self.n_periodos, int) 
            periodos = get_last_n_periodos(self.ult_periodo, self.n_periodos)
            for periodo in periodos:
                year = periodo // 100
                path_item = raw_path / 'fact_prog_acad' / f'{year}' / f'{periodo}.xlsx'
                if not path_item.exists():
                    continue
                df_item = pd.read_excel(path_item, sheet_name='Sheet1', skiprows=1)
                collection.append(df_item)
            if len(collection) == 0:
                print(f"No data found for fact_prog_acad")
            else:
                df_prog_acad = pd.concat(collection, ignore_index=True)
                df_prog_acad.columns = df_prog_acad.columns.astype('string')
                df_prog_acad.columns = df_prog_acad.columns.str.strip().str.upper()
                df_prog_acad.columns = df_prog_acad.columns.str.replace('.', '')
                df_prog_acad.columns = df_prog_acad.columns.str.replace(' ', '_').map(unidecode)
                df_prog_acad_filtered = self.filter_prog_acad(df_prog_acad)
                insert_dataframe_by_periodo(self.session, 'fact_prog_acad', df_prog_acad_filtered, periodos)

                print(f"Successfully read fact_prog_acad")
        else:
            for path_item in raw_path.glob('fact_prog_acad/*/*.xlsx'):
                df_item = pd.read_excel(path_item, sheet_name='Sheet1', skiprows=1)
                collection.append(df_item)

            if len(collection) == 0:
                print(f"No data found for fact_prog_acad")
            else:
                df_prog_acad = pd.concat(collection, ignore_index=True)
                df_prog_acad.columns = df_prog_acad.columns.astype('string')
                df_prog_acad.columns = df_prog_acad.columns.str.strip().str.upper()
                df_prog_acad.columns = df_prog_acad.columns.str.replace('.', '')
                df_prog_acad.columns = df_prog_acad.columns.str.replace(' ', '_').map(unidecode)
                df_prog_acad_filtered = self.filter_prog_acad(df_prog_acad)
                insert_dataframe(self.session, 'fact_prog_acad', df_prog_acad_filtered)
                print(f"Successfully read fact_prog_acad")
        # return df_prog_acad_filtered

    def fetch_fact_predict(self):
        raw_path = Path(self.excel_path)
        # session = self.session()

        collection = []
        assert isinstance(self.full, bool)
        if not self.full:
            # assert isinstance(self.periodo, int)
            # assert isinstance(self.n_periodos, int)
            periodos = [self.periodo_predict]
            for periodo in periodos:
                year = periodo // 100 # type: ignore
                path_item = raw_path / 'fact_predict' / f'{year}' / f'forecast_{periodo}.xlsx'
                if not path_item.exists():
                    continue
                df_item = pd.read_excel(path_item, sheet_name='Sheet1')
                collection.append(df_item)

            if len(collection) == 0:
                print(f"No data found for fact_predict")
            else:
                df_predict = pd.concat(collection, ignore_index=True)
                df_predict.columns = df_predict.columns.astype('string')
                df_predict.columns = df_predict.columns.str.strip().str.upper()
                df_predict.columns = df_predict.columns.str.replace('.', '')
                df_predict.columns = df_predict.columns.str.replace(' ', '_').map(unidecode)
                df_predict['PERIODO'] = df_predict['PERIODO'].astype('int')
                df_predict['SEDE'] = df_predict['SEDE'].astype('string')
                df_predict['CODIGO_DE_CURSO'] = df_predict['CODIGO_DE_CURSO'].astype('string')
                df_predict['HORARIO'] = df_predict['HORARIO'].astype('string')
                insert_dataframe_by_periodo(self.session, 'fact_predict', df_predict, periodos) # type: ignore
                print(f"Successfully read fact_predict")
        else:
            for path_item in raw_path.glob('fact_predict/*/*.xlsx'):
                df_item = pd.read_excel(path_item, sheet_name='Sheet1')
                collection.append(df_item)
            
            if len(collection) == 0:
                print(f"No data found for fact_predict")
            else:
                df_predict = pd.concat(collection, ignore_index=True)
                df_predict.columns = df_predict.columns.astype('string')
                df_predict.columns = df_predict.columns.str.strip().str.upper()
                df_predict.columns = df_predict.columns.str.replace('.', '')
                df_predict.columns = df_predict.columns.str.replace(' ', '_').map(unidecode)
                df_predict['PERIODO'] = df_predict['PERIODO'].astype('int')
                df_predict['SEDE'] = df_predict['SEDE'].astype('string')
                df_predict['CODIGO_DE_CURSO'] = df_predict['CODIGO_DE_CURSO'].astype('string')
                df_predict['HORARIO'] = df_predict['HORARIO'].astype('string')
                insert_dataframe(self.session, 'fact_predict', df_predict)
                print(f"Successfully read fact_predict")
    
    def fetch_dim_horario(self):
        """
        Read Excel file from the dim directory and return a DataFrame.
        
        Args:
            base_path (str): Base path to the dim directory. Defaults to '../raw/dim'.
            
        Returns:
            pd.DataFrame: DataFrame containing the data from the Excel file.
        """
        # session = self.session()
        raw_path = Path(self.excel_path)
        df_horario = pd.read_excel(raw_path / 'dim'/'Horarios Merge.xlsx', sheet_name='Hoja1')
        
        df_horario.columns = df_horario.columns.astype('string')
        df_horario.columns = df_horario.columns.str.strip().str.upper()
        df_horario.columns = df_horario.columns.str.replace('.', '')
        df_horario.columns = df_horario.columns.str.replace(' ', '_').map(unidecode)
        df_horario['HORARIO'] = df_horario['HORARIO'].astype('string')
        df_horario['PERIODO_FRANJA'] = df_horario['PERIODO_FRANJA'].astype('string')
        
        turnos = ['TURNO_1', 'TURNO_2', 'TURNO_3', 'TURNO_4']
        df_horario[turnos] = df_horario[turnos].fillna('')
        df_horario[turnos] = df_horario[turnos].astype('string')
        # df_horario['FRANJAS'] = df_horario[turnos].apply(lambda x: list(x), axis=1)
        # df_horario['FRANJAS'] = df_horario['FRANJAS'].apply(lambda x: [i for i in x if i != ''])
        df_horario_subset = df_horario.loc[
            :, ['HORARIO', 'PERIODO_FRANJA', 'TURNO_1', 
            'TURNO_2', 'TURNO_3', 'TURNO_4']].copy()
        
        print(f"Successfully read dim_horario")
        # records = df_horario_subset.to_dict('records')
        insert_dataframe(self.session, 'dim_horario', df_horario_subset)
        # self.db.insert_dataframe_dim_horario(records)
    
    def fetch_dim_aulas(self):
        raw_path = Path(self.excel_path)
        # session = self.session()
        df_aulas = pd.read_excel(raw_path / 'dim'/'Total Aulas.xlsx', sheet_name='BBDD')
        
        df_aulas.columns = df_aulas.columns.astype('string')
        df_aulas.columns = df_aulas.columns.str.strip().str.upper()
        df_aulas.columns = df_aulas.columns.str.replace('.', '')
        df_aulas.columns = df_aulas.columns.str.replace(' ', '_').map(unidecode)
        month_columns = df_aulas.columns[df_aulas.columns.str.contains(r'\d')]

        df_aulas_normalized = df_aulas.melt(
            id_vars=['SEDE', 'N_AULA', 'YEAR'],
            value_vars=list(month_columns),
            var_name='MONTH',
            value_name='AFORO'
        )

        df_aulas_normalized['MONTH'] = df_aulas_normalized['MONTH'].astype('string')
        df_aulas_normalized['YEAR'] = df_aulas_normalized['YEAR'].astype('string')
        df_aulas_normalized['SEDE'] = df_aulas_normalized['SEDE'].astype('string')
        df_aulas_normalized['N_AULA'] = df_aulas_normalized['N_AULA'].astype('string')
        df_aulas_normalized['PERIODO'] = (df_aulas_normalized['YEAR'] + df_aulas_normalized['MONTH'].str.zfill(2)).astype('int32')

        df_aulas_normalized['AFORO'] = df_aulas_normalized['AFORO'].astype('int32')
        df_aulas_normalized_active = df_aulas_normalized.loc[
            (df_aulas_normalized['AFORO'] > 0) & (df_aulas_normalized['PERIODO'] >= 202206), ['PERIODO', 'SEDE', 'N_AULA', 'AFORO']].copy()

        print(f"Successfully read dim_aulas")
        insert_dataframe(self.session, 'dim_aulas', df_aulas_normalized_active)

    def fetch_dim_sedes(self):
        raw_path = Path(self.excel_path)
        # session = self.session()

        df_sedes = pd.read_excel(raw_path / 'dim'/'Total Aulas.xlsx', sheet_name='SEDE')
        
        df_sedes.columns = df_sedes.columns.astype('string')
        df_sedes.columns = df_sedes.columns.str.strip().str.upper()
        df_sedes.columns = df_sedes.columns.str.replace('.', '')
        df_sedes.columns = df_sedes.columns.str.replace(' ', '_').map(unidecode)
        df_sedes['HABILITADO'] = df_sedes['HABILITADO'].astype('int')
        df_sedes['SEDE'] = df_sedes['SEDE'].astype('string')
        df_sedes['REGION'] = df_sedes['REGION'].astype('string')
        df_sedes['LINEA_DE_NEGOCIO'] = df_sedes['LINEA_DE_NEGOCIO'].astype('string')
        df_sedes_subset = df_sedes.loc[:,
            # df_sedes['HABILITADO'] == 1, 
            ['SEDE',  'REGION', 'LINEA_DE_NEGOCIO']].copy()
        print(f"Successfully read dim_sedes")
        insert_dataframe(self.session, 'dim_sedes', df_sedes_subset)

    def fetch_dim_rewards_sedes(self):
        raw_path = Path(self.excel_path)
        # session = self.session()
        df_rewards_sedes = pd.read_excel( raw_path / 'dim'/'Total Aulas.xlsx', sheet_name='REWARDS')
        
        df_rewards_sedes.columns = df_rewards_sedes.columns.astype('string')
        df_rewards_sedes.columns = df_rewards_sedes.columns.str.strip().str.upper()
        df_rewards_sedes.columns = df_rewards_sedes.columns.str.replace('.', '')
        df_rewards_sedes.columns = df_rewards_sedes.columns.str.replace(' ', '_').map(unidecode)


        nivel_columns = df_rewards_sedes.columns[~df_rewards_sedes.columns.isin(['SEDE', 'N_AULA'])]

        df_rewards_sedes_normalized = df_rewards_sedes.melt(
            id_vars=['SEDE', 'N_AULA'],
            value_vars=list(nivel_columns),
            var_name='NIVEL',
            value_name='REWARD'
        )

        df_rewards_sedes_normalized['SEDE'] = df_rewards_sedes_normalized['SEDE'].astype('string')
        df_rewards_sedes_normalized['N_AULA'] = df_rewards_sedes_normalized['N_AULA'].astype('string')
        df_rewards_sedes_normalized['NIVEL'] = df_rewards_sedes_normalized['NIVEL'].astype('string')
        df_rewards_sedes_normalized['REWARD'] = df_rewards_sedes_normalized['REWARD'].astype('int')

        print(f"Successfully read dim_rewards_sedes")
        insert_dataframe(self.session, 'dim_rewards_sedes', df_rewards_sedes_normalized)

    def fetch_dim_vac_acad(self):
        raw_path = Path(self.excel_path)
        # session = self.session()
        df_vac_acad = pd.read_excel( raw_path / 'dim'/'Total Aulas.xlsx', sheet_name='VAC_ACAD_ESTANDAR')
        
        df_vac_acad.columns = df_vac_acad.columns.astype('string')
        df_vac_acad.columns = df_vac_acad.columns.str.strip().str.upper()
        df_vac_acad.columns = df_vac_acad.columns.str.replace('.', '')
        df_vac_acad.columns = df_vac_acad.columns.str.replace(' ', '_').map(unidecode)
        df_vac_acad['PERIODO'] = df_vac_acad['PERIODO'].astype('int32')
        df_vac_acad['LINEA_DE_NEGOCIO'] = df_vac_acad['LINEA_DE_NEGOCIO'].astype('string')
        df_vac_acad['NIVEL'] = df_vac_acad['NIVEL'].astype('string')
        df_vac_acad['VAC_ACAD_ESTANDAR'] = df_vac_acad['VAC_ACAD_ESTANDAR'].astype('int32')
        df_vac_acad['VAC_ACAD_MAX'] = df_vac_acad['VAC_ACAD_MAX'].astype('int32')
        print(f"Successfully read  dim_vac_acad")
        insert_dataframe(self.session, 'dim_vac_acad', df_vac_acad)

    def fetch_dim_horarios_atencion(self):
        raw_path = Path(self.excel_path)
        # session = self.session()

        df_horarios_atencion = pd.read_excel( raw_path / 'dim'/'Total Aulas.xlsx', sheet_name='HORARIOS_ATENCION')
        df_horarios_atencion.columns = df_horarios_atencion.columns.astype('string')
        df_horarios_atencion.columns = df_horarios_atencion.columns.str.strip().str.upper()
        df_horarios_atencion.columns = df_horarios_atencion.columns.str.replace('.', '')
        df_horarios_atencion.columns = df_horarios_atencion.columns.str.replace(' ', '_').map(unidecode)
        month_columns = df_horarios_atencion.columns[df_horarios_atencion.columns.str.contains(r'\d')]

        df_horarios_atencion_normalized = df_horarios_atencion.melt(
            id_vars=['SEDE', 'PERIODO_FRANJA', 'FRANJA', 'YEAR'],
            value_vars=list(month_columns),
            var_name='MONTH',
            value_name='FLAG_ACTIVA'
        )

        df_horarios_atencion_normalized['MONTH'] = df_horarios_atencion_normalized['MONTH'].astype('string')
        df_horarios_atencion_normalized['YEAR'] = df_horarios_atencion_normalized['YEAR'].astype('string')
        df_horarios_atencion_normalized['SEDE'] = df_horarios_atencion_normalized['SEDE'].astype('string')
        df_horarios_atencion_normalized['PERIODO_FRANJA'] = df_horarios_atencion_normalized['PERIODO_FRANJA'].astype('string')
        df_horarios_atencion_normalized['FRANJA'] = df_horarios_atencion_normalized['FRANJA'].astype('string')
        df_horarios_atencion_normalized['PERIODO'] = (
            df_horarios_atencion_normalized['YEAR'] + df_horarios_atencion_normalized['MONTH'].str.zfill(2)).astype('int')

        df_horarios_atencion_normalized['FLAG_ACTIVA'] = df_horarios_atencion_normalized['FLAG_ACTIVA'].astype('int')
        df_horarios_atencion_active = df_horarios_atencion_normalized.loc[
            df_horarios_atencion_normalized['FLAG_ACTIVA'] == 1, ['PERIODO', 'SEDE', 'PERIODO_FRANJA', 'FRANJA']].copy()

        print(f"Successfully read dim_horarios_atencion")
        insert_dataframe(self.session, 'dim_horarios_atencion', df_horarios_atencion_active)
    
    def fetch_dim_cursos(self):
        raw_path = Path(self.excel_path)
        # session = self.session()

        df_cursos = pd.read_excel( raw_path / 'dim'/'Cursos Acumulado.xlsx', sheet_name='BBDD')
        
        df_cursos.columns = df_cursos.columns.astype('string')
        df_cursos.columns = df_cursos.columns.str.strip().str.upper()
        df_cursos.columns = df_cursos.columns.str.replace('.', '')
        df_cursos.columns = df_cursos.columns.str.replace(' ', '_').map(unidecode)
        df_cursos['FRECUENCIA'] = df_cursos['FRECUENCIA'].astype('string')
        df_cursos['NIVEL'] = df_cursos['NIVEL'].astype('string')
        df_cursos['CURSO_ANTERIOR'] = df_cursos['CURSO_ANTERIOR'].astype('string')
        df_cursos['CURSO_ACTUAL'] = df_cursos['CURSO_ACTUAL'].astype('string')
        df_cursos_subset = df_cursos.loc[:, ['FRECUENCIA', 'NIVEL', 'CURSO_ANTERIOR', 'CURSO_ACTUAL', 'DURACION']].copy()
        print(f"Successfully read dim_cursos")
        insert_dataframe(self.session, 'dim_cursos', df_cursos_subset)

    def fetch_fact_provicional(self):
        raw_path = Path(self.excel_path)
        # session = self.session()

        df_provicional = pd.read_excel(raw_path / 'dim'/'Total Aulas.xlsx', sheet_name='PROVICIONAL')
        df_provicional.columns = df_provicional.columns.astype('string')
        df_provicional.columns = df_provicional.columns.str.strip().str.upper()
        df_provicional.columns = df_provicional.columns.str.replace('.', '')
        df_provicional.columns = df_provicional.columns.str.replace(' ', '_').map(unidecode)
        df_provicional['PERIODO'] = df_provicional['PERIODO'].astype('int')
        df_provicional['SEDE'] = df_provicional['SEDE'].astype('string')
        df_provicional['CODIGO_DE_CURSO'] = df_provicional['CODIGO_DE_CURSO'].astype('string')
        df_provicional['HORARIO'] = df_provicional['HORARIO'].astype('string')
        df_provicional['AULA'] = df_provicional['AULA'].astype('string')
        df_provicional['AULA_PROVICIONAL'] = df_provicional['AULA_PROVICIONAL'].astype('string')
        
        df_provicional_subset = df_provicional.loc[:, 
        ['PERIODO', 'SEDE', 'CODIGO_DE_CURSO', 'HORARIO', 'AULA', 'AULA_PROVICIONAL']].copy()
        print(f"Successfully read fact_provicional")
        insert_dataframe(self.session, 'fact_provicional', df_provicional_subset)
        
    def fetch_dim_dias(self):
        raw_path = Path(self.excel_path)
        # session = self.session()
        df_dias = pd.read_excel(
            raw_path / 'dim'/'Horarios Merge.xlsx', sheet_name='FRECUENCIAS')
        df_dias.columns = df_dias.columns.astype('string')
        df_dias.columns = df_dias.columns.str.strip().str.upper()
        df_dias.columns = df_dias.columns.str.replace('.', '')
        df_dias.columns = df_dias.columns.str.replace(' ', '_').map(unidecode)
        df_dias_melt = df_dias.melt(
            id_vars=['FRECUENCIA'],
            value_vars=['LUN', 'MAR', 'MIE', 'JUE', 'VIE', 'SAB', 'DOM'],
            var_name='DIA',
            value_name='FLAG_ACTIVA'
        )
        
        df_dias_melt['FLAG_ACTIVA'] = df_dias_melt['FLAG_ACTIVA'].astype('int')
        df_dias_melt['FRECUENCIA'] = df_dias_melt['FRECUENCIA'].astype('string')
        df_dias_melt['DIA'] = df_dias_melt['DIA'].astype('string')
        df_dias_melt_active = df_dias_melt.loc[df_dias_melt['FLAG_ACTIVA'] == 1, ['FRECUENCIA', 'DIA']].copy()

        print(f"Successfully read dim_dias")
        insert_dataframe(self.session, 'dim_dias', df_dias_melt_active)

    def fetch_dim_dias_turnos(self):
        raw_path = Path(self.excel_path)
        # session = self.session()
        
        df_dias_turnos = pd.read_excel(
            raw_path / 'dim' / 'Horarios Merge.xlsx', sheet_name='DIA_TURNO')
        df_dias_turnos.columns = df_dias_turnos.columns.astype('string')
        df_dias_turnos.columns = df_dias_turnos.columns.str.strip().str.upper()
        df_dias_turnos.columns = df_dias_turnos.columns.str.replace('.', '')
        df_dias_turnos.columns = df_dias_turnos.columns.str.replace(' ', '_').map(unidecode)

        print(f"Successfully read dim_dias_turnos")
        insert_dataframe(self.session, 'dim_dias_turnos', df_dias_turnos)
    
    def fetch_all(self):
        self.fetch_fact_prog_acad()
        self.fetch_fact_predict()
        self.fetch_dim_horario()
        self.fetch_dim_aulas()
        self.fetch_dim_sedes()
        self.fetch_dim_rewards_sedes()
        self.fetch_dim_vac_acad()
        self.fetch_dim_horarios_atencion()
        self.fetch_dim_cursos()
        self.fetch_fact_provicional()
        self.fetch_dim_dias()
        self.fetch_dim_dias_turnos()


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    periodo_predict = config['periodo_predict']
    ult_periodo = config['ult_periodo']
    n_periodos = config['n_periodos']
    full = config['full']
    db_path = config['db_path']
    excel_path = config['excel_path']

    # drop_all = True
    sqlite_session = get_sqlite_session()
    # db = DatabaseManager(db_path, drop_all)
    # db.initialize_database()
    extractor = ExcelExtractor(periodo_predict, ult_periodo, n_periodos, full, sqlite_session, excel_path)
    extractor.fetch_all()
