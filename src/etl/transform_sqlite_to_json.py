import json
from json import dumps
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union
import os
from unidecode import unidecode
import logging
import yaml
from uuid import uuid4
from datetime import datetime
import numpy as np
from utils import get_last_n_periodos, get_n_lags
from database.utils import query_to_dataframe


class DataTransformer:
    def __init__(
        self, periodo_predict: int|None, ult_periodo: int|None, 
        n_periodos: int|None, full: bool,session, engine,
        log_path: str, data_path: str, dim_path:str,  items_predict_path: str, items_path: str, items_bim_path: str, room_log_path: str):
        self.periodo_predict = periodo_predict
        self.ult_periodo = ult_periodo
        self.n_periodos = n_periodos
        self.full = full
        self.log_path = log_path
        self.data_path = data_path
        self.session = session
        self.engine = engine
        self.data_prog_presencial = pd.DataFrame()
        self.data_predict_presencial = pd.DataFrame()
        self.dim_path = dim_path
        self.items_predict_path = items_predict_path
        self.items_path = items_path
        self.items_bim_path = items_bim_path
        self.room_log_path = room_log_path

    def get_fact_prog_acad(self):
        if not self.full:
            assert isinstance(self.ult_periodo, int)
            assert isinstance(self.n_periodos, int)
            periodos = get_last_n_periodos(self.ult_periodo, self.n_periodos)
            
            # In the query join with fact_provicional to get the AULA_PROVICIONAL
            df_prog_acad = query_to_dataframe(
                self.session, self.engine,
                """SELECT 
                prog_acad.PERIODO, 
                prog_acad.SEDE, 
                prog_acad.CODIGO_DE_CURSO,
                prog_acad.HORARIO,
                prog_acad.CANT_MATRICULADOS,
                prog_acad.VAC_HABILITADAS,
                CASE WHEN prov.AULA_PROVICIONAL IS NOT NULL THEN prov.AULA_PROVICIONAL ELSE prog_acad.AULA END AS AULA
                FROM 
                fact_prog_acad AS prog_acad 
                LEFT OUTER JOIN 
                fact_provicional AS prov 
                ON (
                prog_acad.PERIODO = prov.PERIODO AND
                prog_acad.SEDE = prov.SEDE AND
                prog_acad.CODIGO_DE_CURSO = prov.CODIGO_DE_CURSO AND
                prog_acad.HORARIO = prov.HORARIO AND
                prog_acad.AULA = prov.AULA
                )
                WHERE prog_acad.PERIODO IN {};
                """.format(tuple(periodos))
                )
        else:
            # In the query join with fact_provicional to get the AULA_PROVICIONAL
            df_prog_acad = query_to_dataframe(self.session, self.engine,
                """SELECT 
                prog_acad.PERIODO, 
                prog_acad.SEDE, 
                prog_acad.CODIGO_DE_CURSO,
                prog_acad.HORARIO,
                prog_acad.CANT_MATRICULADOS,
                prog_acad.VAC_HABILITADAS,
                CASE WHEN prov.AULA_PROVICIONAL IS NOT NULL THEN prov.AULA_PROVICIONAL ELSE prog_acad.AULA END AS AULA
                FROM 
                fact_prog_acad AS prog_acad 
                LEFT OUTER JOIN 
                fact_provicional AS prov 
                ON (
                prog_acad.PERIODO = prov.PERIODO AND
                prog_acad.SEDE = prov.SEDE AND
                prog_acad.CODIGO_DE_CURSO = prov.CODIGO_DE_CURSO AND
                prog_acad.HORARIO = prov.HORARIO AND
                prog_acad.AULA = prov.AULA
                )
                """
                )
        # -- prog_acad.PERIODO = prov.PERIODO AND
        # print(df_prog_acad.head()    )
        df_prog_acad['START'] = df_prog_acad['HORARIO'].str.replace(
            r'(\d{2}):(\d{2}) - (\d{2}):(\d{2})', r'\1:\2:00', regex=True)
        df_prog_acad['END'] = df_prog_acad['HORARIO'].str.replace(
            r'(\d{2}):(\d{2}) - (\d{2}):(\d{2})', r'\3:\4:00', regex=True)
        # print(df_prog_acad['START'].unique())
        # print(df_prog_acad['END'].unique())
        df_prog_acad['START'] = pd.to_timedelta(df_prog_acad['START'])
        df_prog_acad['END'] = pd.to_timedelta(df_prog_acad['END'])
        df_prog_acad['SEDE'] = df_prog_acad['SEDE'].str.replace('Provincias - ', '')
        df_prog_acad['AULA'] = df_prog_acad['AULA'].str.replace(' - CAD', '')
        print(f"Successfully read fact_prog_acad")
        # df_prog_acad.to_excel('fact_prog_acad.xlsx', index=False)
        return df_prog_acad

    def get_fact_predict(self):
        if not self.full:
            # assert isinstance(self.periodo, int)
            # assert isinstance(self.n_periodos, int)
            # periodos = [self.periodo_predict]
            df_predict = query_to_dataframe(
                self.session, self.engine,
                """
                SELECT 
                PERIODO, 
                SEDE, 
                CODIGO_DE_CURSO,
                HORARIO,
                FORECAST_AULAS,
                FORECAST_ALUMN
                FROM 
                 fact_predict
                WHERE PERIODO = {};
                """.format(self.periodo_predict)
            )
        else:
            df_predict = query_to_dataframe(self.session, self.engine,
                """
                SELECT 
                PERIODO, 
                SEDE, 
                CODIGO_DE_CURSO,
                HORARIO,
                FORECAST_AULAS,
                FORECAST_ALUMN
                FROM 
                fact_predict
            """
            )

        def split(cant_alumn: int, cant_clases: int):
            cant_alumn = int(cant_alumn)
            cant_clases = int(cant_clases)
            alumn_prom = cant_alumn / cant_clases
            assert cant_clases >= 1
            if cant_clases == 1:
                return [cant_alumn]
            else:
                aulas = np.array([alumn_prom//1]*cant_clases)
                aulas_remaning = np.array([alumn_prom % 1]*cant_clases)
                total_remaning = int(round(cant_alumn - sum(aulas)))
                if sum(aulas) == cant_alumn:
                    return aulas.tolist()
                else:
                    idx = np.flip(np.argsort(aulas_remaning))
                    aulas[idx[:total_remaning]] += 1
                    return aulas.tolist()
        
        # ee = df_predict.apply(lambda row: split(row['FORECAST_ALUMN'], row['FORECAST_AULAS']), axis=1)
        # print(ee)
        df_predict['ITEMS'] = df_predict.apply(lambda row: split(row['FORECAST_ALUMN'], row['FORECAST_AULAS']), axis=1)
        df_predict_01 = df_predict.explode('ITEMS', ignore_index=True)
        df_predict_02 = df_predict_01[['PERIODO', 'SEDE', 'CODIGO_DE_CURSO', 'HORARIO', 'ITEMS']].copy()
        df_predict_03 =df_predict_02.rename(columns={'ITEMS': 'FORECAST_ALUMN'})
        df_predict_03['FORECAST_ALUMN'] = df_predict_03['FORECAST_ALUMN'].astype('int')
        df_predict_03['AULA'] = '******'
        df_predict_03['START'] = df_predict_03['HORARIO'].str.replace(
            r'(\d{2}):(\d{2}) - (\d{2}):(\d{2})', r'\1:\2:00', regex=True)
        df_predict_03['END'] = df_predict_03['HORARIO'].str.replace(
            r'(\d{2}):(\d{2}) - (\d{2}):(\d{2})', r'\3:\4:00', regex=True)
        df_predict_03['START'] = pd.to_timedelta(df_predict_03['START'])
        df_predict_03['END'] = pd.to_timedelta(df_predict_03['END'])
        print(f"Successfully read fact_predict")
        return df_predict_03

    def get_dim_horario(self):
        df_horario = query_to_dataframe(self.session, self.engine,
            """
            SELECT 
            HORARIO, 
            PERIODO_FRANJA,
            TURNO_1,
            TURNO_2,
            TURNO_3,
            TURNO_4
            FROM 
             dim_horario
            """
            )
        df_horario['HORARIO'] = df_horario['HORARIO'].astype('string')
        df_horario['PERIODO_FRANJA'] = df_horario['PERIODO_FRANJA'].astype('string')
        df_horario['TURNO_1'] = df_horario['TURNO_1'].astype('string')
        df_horario['TURNO_2'] = df_horario['TURNO_2'].astype('string')
        df_horario['TURNO_3'] = df_horario['TURNO_3'].astype('string')
        df_horario['TURNO_4'] = df_horario['TURNO_4'].astype('string')
        # list of turnos
        df_horario['TURNOS'] = df_horario.apply(lambda row: [row['TURNO_1'], row['TURNO_2'], row['TURNO_3'], row['TURNO_4']], axis=1)
        df_horario['TURNOS'] = df_horario['TURNOS'].apply(lambda x: [i for i in x if i != ''])
        df_horario_01 = df_horario[['HORARIO', 'PERIODO_FRANJA', 'TURNOS']].copy()
        print(f"Successfully read dim_horario")
        return df_horario_01
    
    def get_dim_aulas(self):
        df_aulas = query_to_dataframe(self.session, self.engine,
            """
            SELECT 
            PERIODO, 
            SEDE, 
            N_AULA,
            AFORO
            FROM 
             dim_aulas
            """
            )
        print(f"Successfully read dim_aulas")
        return df_aulas
    
    def get_dim_sedes(self):
        df_sedes = query_to_dataframe(self.session, self.engine,
            """
            SELECT 
            SEDE, 
            REGION,
            LINEA_DE_NEGOCIO
            FROM 
             dim_sedes
            """
            )
        print(f"Successfully read dim_sedes")
        return df_sedes
    
    def get_dim_horarios_atencion(self):
        df_horarios_atencion = query_to_dataframe(self.session, self.engine, 
            """
            SELECT 
            PERIODO, 
            SEDE, 
            PERIODO_FRANJA,
            FRANJA
            FROM 
             dim_horarios_atencion
            """
            )

        df_horarios_atencion['START'] = df_horarios_atencion['FRANJA'].str.replace(
            r'(\d{2}):(\d{2}) - (\d{2}):(\d{2})', r'\1:\2:00', regex=True)
        df_horarios_atencion['END'] = df_horarios_atencion['FRANJA'].str.replace(
            r'(\d{2}):(\d{2}) - (\d{2}):(\d{2})', r'\3:\4:00', regex=True)
        # print(df_horarios_atencion['START'].unique())
        # print(df_horarios_atencion['END'].unique())
        df_horarios_atencion['START'] = pd.to_timedelta(df_horarios_atencion['START'])
        df_horarios_atencion['END'] = pd.to_timedelta(df_horarios_atencion['END'])

        # MIN and MAX
        df_horarios_atencion_01 = df_horarios_atencion.groupby(['PERIODO', 'SEDE', 'PERIODO_FRANJA'], as_index=False).agg(
            MIN=pd.NamedAgg(column='START', aggfunc='min'),
            MAX=pd.NamedAgg(column='END', aggfunc='max')
        )

        print(f"Successfully read dim_horarios_atencion")
        return df_horarios_atencion_01

    def get_dim_rewards_sedes(self):
        df_rewards_sedes = query_to_dataframe(self.session, self.engine, 
            """
            SELECT 
            SEDE, 
            N_AULA,
            NIVEL,
            REWARD
            FROM 
            dim_rewards_sedes
            """
            )
        print(f"Successfully read dim_rewards_sedes")
        return df_rewards_sedes

    def get_dim_vac_acad(self):
        df_vac_acad = query_to_dataframe(self.session, self.engine,
            """
            SELECT 
            PERIODO, 
            LINEA_DE_NEGOCIO,
            NIVEL,
            VAC_ACAD_ESTANDAR,
            VAC_ACAD_MAX
            FROM 
             dim_vac_acad
            """
            )
        print(f"Successfully read  dim_vac_acad")
        return df_vac_acad

    def get_dim_cursos(self):
        df_cursos = query_to_dataframe(self.session, self.engine, 
            """
            SELECT 
            FRECUENCIA,
            NIVEL,
            CURSO_ANTERIOR,
            CURSO_ACTUAL,
            DURACION 
            FROM dim_cursos
            """
            )
        print(f"Successfully read dim_cursos")
        return df_cursos
    
    def get_dim_dias(self):
        df_dias = query_to_dataframe(self.session, self.engine,
            """
            SELECT 
            FRECUENCIA,
            DIA 
            FROM dim_dias
            """
            )

        df_dias_01 = df_dias.groupby(['FRECUENCIA'], as_index=False).agg(
            DIAS=pd.NamedAgg(column='DIA', aggfunc=lambda x: list(x))
        )
        print(f"Successfully read dim_dias")
        return df_dias_01

    def get_dim_dias_turnos(self):

        df_dias_turnos = query_to_dataframe(self.session, self.engine,
            """
            SELECT 
            DIA,
            TURNO
            FROM dim_dias_turnos
            """
            )
        
        df_dias_turnos['TURNOS'] = (
            df_dias_turnos.apply(lambda x: {'TURNO':x['TURNO'], 'AVAILABLE':1}, axis=1))
        
        df_dias_turnos_01 = df_dias_turnos.groupby(['DIA'], as_index=False).agg(
            TURNOS=pd.NamedAgg(column='TURNOS', aggfunc=lambda x: list(x))
        )
        
        print(f"Successfully read dim_dias")
        return df_dias_turnos_01
    
    def validate_fact_data(
        self, fact_data:pd.DataFrame, table_name:str, dim_curso:pd.DataFrame, 
        dim_horario:pd.DataFrame, dim_sedes:pd.DataFrame, dim_vac_acad:pd.DataFrame,
        dim_dias:pd.DataFrame):

        # Add dim_cursos to fact_data
        df_fact_data_01 = fact_data.merge(
            dim_curso.rename(columns={'CURSO_ANTERIOR': 'CODIGO_DE_CURSO'}), 
            on=['CODIGO_DE_CURSO'], how='left')

        # log if CURSO_ACTUAL has null values
        if df_fact_data_01['CURSO_ACTUAL'].isnull().sum() > 0:
            uuid = uuid4()
            # save dataframe to csv
            df_fact_data_01_null = df_fact_data_01[df_fact_data_01['CURSO_ACTUAL'].isnull()].copy()
            df_fact_data_01_null.to_csv(f'{self.log_path}/{table_name}_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"CURSO_ACTUAL has null values, saved to {table_name}_{uuid}.csv")
        else:
            logging.info(f"CURSO_ACTUAL has no null values {table_name}")

        # Add dim_horario to df_fact_data_01
        df_fact_data_02 = df_fact_data_01.merge(
            dim_horario, 
            on=['HORARIO'], how='left')

        # log if HORARIO has null values
        if df_fact_data_02['PERIODO_FRANJA'].isnull().sum() > 0:
            uuid = uuid4()
            # save dataframe to csv
            df_fact_data_02_null = df_fact_data_02[df_fact_data_02['PERIODO_FRANJA'].isnull()].copy()
            df_fact_data_02_null.to_csv(f'{self.log_path}/{table_name}_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"PERIODO_FRANJA has null values, saved to {table_name}_{uuid}.csv")
        else:
            logging.info(f"PERIODO_FRANJA has no null values {table_name}")

        # Add dim_sedes to df_fact_data_02
        df_fact_data_03 = df_fact_data_02.merge(
            dim_sedes, 
            on=['SEDE'], how='left')

        # log if SEDE has null values
        if df_fact_data_03['LINEA_DE_NEGOCIO'].isnull().sum() > 0:
            uuid = uuid4()
            # save dataframe to csv
            df_fact_data_03_null = df_fact_data_03[df_fact_data_03['LINEA_DE_NEGOCIO'].isnull()].copy()
            df_fact_data_03_null.to_csv(f'{self.log_path}/{table_name}_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"LINEA_DE_NEGOCIO has null values, saved to {table_name}_{uuid}.csv")
        else:
            logging.info(f"LINEA_DE_NEGOCIO has no null values {table_name}")

        # Add dim_vac_acad to df_fact_data_03
        df_fact_data_04 = df_fact_data_03.merge(
            dim_vac_acad, 
            on=['PERIODO', 'LINEA_DE_NEGOCIO', 'NIVEL'], how='left')

        # log if PERIODO has null values
        if df_fact_data_04['VAC_ACAD_ESTANDAR'].isnull().sum() > 0:
            uuid = uuid4()
            # save dataframe to csv
            df_fact_data_04_null = df_fact_data_04[df_fact_data_04['VAC_ACAD_ESTANDAR'].isnull()].copy()
            df_fact_data_04_null.to_csv(f'{self.log_path}/{table_name}_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"VAC_ACAD_ESTANDAR has null values, saved to {table_name}_{uuid}.csv")
        else:
            logging.info(f"VAC_ACAD_ESTANDAR has no null values {table_name}")

        # Add dim_dias to df_fact_data_04
        df_fact_data_05 = df_fact_data_04.merge(
            dim_dias, 
            on=['FRECUENCIA'], how='left')
        
        # log if FRECUENCIA has null values
        if df_fact_data_05['DIAS'].isnull().sum() > 0:
            uuid = uuid4()
            # save dataframe to csv
            df_fact_data_05_null = df_fact_data_05[df_fact_data_05['DIAS'].isnull()].copy()
            df_fact_data_05_null.to_csv(f'{self.log_path}/{table_name}_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"DIAS has null values, saved to {table_name}_{uuid}.csv")
        else:
            logging.info(f"DIAS has no null values {table_name}")

        return df_fact_data_05
    
    def validate_data(self):
        df_fact_prog_acad = self.get_fact_prog_acad()
        df_fact_predict = self.get_fact_predict()
        df_dim_cursos = self.get_dim_cursos()
        df_dim_dias = self.get_dim_dias()
        df_dim_horario = self.get_dim_horario()
        df_dim_sedes = self.get_dim_sedes()
        df_dim_vac_acad = self.get_dim_vac_acad()

        df_dim_aulas = self.get_dim_aulas()
        df_dim_rewards_sedes = self.get_dim_rewards_sedes()
        df_dim_horarios_atencion = self.get_dim_horarios_atencion()

        df_fact_prog_acad_01 = df_fact_prog_acad[
            ['PERIODO', 'SEDE', 'CODIGO_DE_CURSO', 'HORARIO', 'AULA',
             'CANT_MATRICULADOS','VAC_HABILITADAS', 'START', 'END']].copy()

        df_fact_prog_acad_02 = self.validate_fact_data(
            df_fact_prog_acad_01, 'fact_prog_acad', df_dim_cursos, df_dim_horario, df_dim_sedes, df_dim_vac_acad, df_dim_dias)
        
        df_fact_predict_01 = self.validate_fact_data(
            df_fact_predict, 'fact_predict', df_dim_cursos, df_dim_horario, df_dim_sedes, df_dim_vac_acad, df_dim_dias)

        df_fact_prog_acad_presencial = df_fact_prog_acad_02[
            df_fact_prog_acad_02['LINEA_DE_NEGOCIO'] == 'Presencial'].copy()

        df_fact_predict_presencial = df_fact_predict_01[
            df_fact_predict_01['LINEA_DE_NEGOCIO'] == 'Presencial'].copy()

        # Add dim_aulas to df_fact_prog_acad_presencial
        df_fact_prog_acad_presencial_01 = df_fact_prog_acad_presencial.merge(
            df_dim_aulas.rename(columns={'N_AULA': 'AULA'}), 
            on=['PERIODO', 'SEDE', 'AULA'], how='left')

        # log if AULA has null values
        if df_fact_prog_acad_presencial_01['AFORO'].isnull().sum() > 0:
            uuid = uuid4()
            # save dataframe to csv
            df_fact_prog_acad_presencial_01_null = df_fact_prog_acad_presencial_01[
                df_fact_prog_acad_presencial_01['AFORO'].isnull()].copy()
            df_fact_prog_acad_presencial_01_null.to_csv(
                f'{self.log_path}/fact_prog_acad_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"AFORO has null values, saved to fact_prog_acad_{uuid}.csv")
        else:
            logging.info(f"AFORO has no null values")

        # Add dim_rewards_sedes to df_fact_prog_acad_presencial
        df_fact_prog_acad_presencial_02 = df_fact_prog_acad_presencial_01.merge(
            df_dim_rewards_sedes.rename(columns={'N_AULA': 'AULA'}), 
            on=['SEDE', 'AULA', 'NIVEL'], how='left')

        # log if REWARD has null values
        if df_fact_prog_acad_presencial_02['REWARD'].isnull().sum() > 0:
            uuid = uuid4()
            # save dataframe to csv
            df_fact_prog_acad_presencial_02_null = df_fact_prog_acad_presencial_02[
                df_fact_prog_acad_presencial_02['REWARD'].isnull()].copy()
            df_fact_prog_acad_presencial_02_null.to_csv(
                f'{self.log_path}/fact_prog_acad_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"REWARD has null values, saved to fact_prog_acad_{uuid}.csv")
        else:
            logging.info(f"REWARD has no null values")

        # Add dim_horarios_atencion to df_fact_prog_acad_presencial
        df_fact_prog_acad_presencial_03 = df_fact_prog_acad_presencial_02.merge(
            df_dim_horarios_atencion, 
            on=['PERIODO', 'SEDE', 'PERIODO_FRANJA'], how='left')

        # log if START or END has null values
        if (df_fact_prog_acad_presencial_03['MIN'].isnull().sum() > 0) or (
            df_fact_prog_acad_presencial_03['MAX'].isnull().sum() > 0):
            uuid = uuid4()
            # save dataframe to csv
            df_fact_prog_acad_presencial_03_null = df_fact_prog_acad_presencial_03[
                df_fact_prog_acad_presencial_03['MIN'].isnull() | df_fact_prog_acad_presencial_03['MAX'].isnull()].copy()
            df_fact_prog_acad_presencial_03_null.to_csv(f'{self.log_path}/fact_prog_acad_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"MIN or MAX has null values, saved to fact_prog_acad_{uuid}.csv")
        else:
            logging.info(f"MIN and MAX has no null values")

        if (
            np.sum(df_fact_prog_acad_presencial_03['START']<df_fact_prog_acad_presencial_03['MIN']) or 
            np.sum(df_fact_prog_acad_presencial_03['END']>df_fact_prog_acad_presencial_03['MAX']) ):
            uuid = uuid4()
            # save dataframe to csv
            df_fact_prog_acad_presencial_03_out_time = df_fact_prog_acad_presencial_03[
                (df_fact_prog_acad_presencial_03['START'] < df_fact_prog_acad_presencial_03['MIN']) | 
                (df_fact_prog_acad_presencial_03['END'] > df_fact_prog_acad_presencial_03['MAX'])].copy()

            df_fact_prog_acad_presencial_03_out_time.to_csv(
                f'{self.log_path}/fact_prog_acad_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"START or END is out of time, saved to fact_prog_acad_{uuid}.csv")
        else:
            logging.info(f"START and END are in time")

        self.data_prog_presencial = df_fact_prog_acad_presencial_03.copy()

        # ----------------------------------------------------------------------
    

        # Add dim_horarios_atencion to df_fact_predict_presencial
        df_fact_predict_presencial_01 = df_fact_predict_presencial.merge(
            df_dim_horarios_atencion, 
            on=['PERIODO', 'SEDE', 'PERIODO_FRANJA'], how='left')

        # log if START or END has null values
        if (df_fact_predict_presencial_01['START'].isnull().sum() > 0) or (df_fact_predict_presencial_01['END'].isnull().sum() > 0):
            uuid = uuid4()
            # save dataframe to csv
            df_fact_predict_presencial_01_null = df_fact_predict_presencial_01[
                df_fact_predict_presencial_01['START'].isnull() | df_fact_predict_presencial_01['END'].isnull()].copy()
            df_fact_predict_presencial_01_null.to_csv(f'{self.log_path}/fact_predict_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"START or END has null values, saved to fact_predict_{uuid}.csv")
        else:
            logging.info(f"START and END has no null values")

        if (
            np.sum(df_fact_predict_presencial_01['START']<df_fact_predict_presencial_01['MIN']) or 
            np.sum(df_fact_predict_presencial_01['END']>df_fact_predict_presencial_01['MAX']) ):
            uuid = uuid4()
            # save dataframe to csv
            df_fact_predict_presencial_01_out_time = df_fact_predict_presencial_01[
                (df_fact_predict_presencial_01['START'] < df_fact_predict_presencial_01['MIN']) | 
                (df_fact_predict_presencial_01['END'] > df_fact_predict_presencial_01['MAX'])].copy()

            df_fact_predict_presencial_01_out_time.to_csv(
                f'{self.log_path}/fact_predict_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"START or END is out of time, saved to fact_predict_{uuid}.csv")
        else:
            logging.info(f"START and END are in time")

        self.data_predict_presencial = df_fact_predict_presencial_01.copy()

        # df_fact_predict_subset = df_fact_predict[['PERIODO', 'SEDE', 'CODIGO_DE_CURSO', 'HORARIO', 'FORECAST_ALUMN', 'FORECAST_AULAS']].copy()

    def validate_aulas_and_rewards(self):
        df_dim_aulas = self.get_dim_aulas()
        df_dim_rewards_sedes = self.get_dim_rewards_sedes()
        df_dim_rewards_sedes_01 = df_dim_rewards_sedes[['SEDE', 'N_AULA']].drop_duplicates()
        df_dim_rewards_sedes_01['FLAG_REWARD'] = 1

        df_dim_aulas_01 = df_dim_aulas.merge(
            df_dim_rewards_sedes_01, on=['SEDE', 'N_AULA'], how='left')
        
        if df_dim_aulas_01['FLAG_REWARD'].isnull().sum() > 0:
            uuid = uuid4()
            # save dataframe to csv
            df_dim_aulas_01_null = df_dim_aulas_01[df_dim_aulas_01['FLAG_REWARD'].isnull()].copy()
            df_dim_aulas_01_null.to_csv(f'{self.log_path}/dim_aulas_{uuid}.csv', index=False, encoding='utf-8')
            logging.info(f"FLAG_REWARD has null values, saved to dim_aulas_{uuid}.csv")
        else:
            logging.info(f"FLAG_REWARD has no null values")

    def get_items(self):
        data_prog_presencial = self.data_prog_presencial.copy()
        data_prog_presencial_01 = data_prog_presencial[
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO', 'AULA',  
                 'CANT_MATRICULADOS', 'FRECUENCIA', 'NIVEL', 'DURACION', 'PERIODO_FRANJA', 
                 'TURNOS', 'DIAS', 'VAC_ACAD_ESTANDAR']].copy()
              
        for group, data in data_prog_presencial_01.groupby('PERIODO'):
            assert isinstance(group, int)
            periodo = group
            year = periodo//100
            Path(f'{self.data_path}/{self.items_path}').mkdir(parents=True, exist_ok=True)
            Path(f'{self.data_path}/{self.items_path}/{year}').mkdir(parents=True, exist_ok=True)
            data_dict = data.to_dict(
                orient='records')

            if len(data_dict) == 0:
                continue
            
            with open(f'{self.data_path}/{self.items_path}/{year}/items_{periodo}.json', 'w', encoding='utf-8') as file:
                file.write(dumps(data_dict, ensure_ascii=False, indent=4))
        
    def get_items_bim(self):
        data_prog_presencial = self.data_prog_presencial.copy()
        data_prog_presencial_01 = data_prog_presencial[
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO', 'AULA',  
                 'CANT_MATRICULADOS', 'FRECUENCIA', 'NIVEL', 'DURACION', 'PERIODO_FRANJA', 
                 'TURNOS', 'DIAS', 'VAC_ACAD_ESTANDAR']].copy()

        data_prog_presencial_02 = data_prog_presencial_01[data_prog_presencial_01['DURACION'] == 'Bimensual'].copy()
        data_prog_presencial_02['PERIODO'] = data_prog_presencial_02['PERIODO'].map(lambda x:  get_n_lags(x, -1))
        for group, data in data_prog_presencial_02.groupby('PERIODO'):
            assert isinstance(group, int)
            periodo = group
            year = periodo//100
            Path(f'{self.data_path}/{self.items_bim_path}').mkdir(parents=True, exist_ok=True)
            Path(f'{self.data_path}/{self.items_bim_path}/{year}').mkdir(parents=True, exist_ok=True)
            data_dict = data.to_dict(
                orient='records')
            if len(data_dict) == 0:
                continue
            with open(f'{self.data_path}/{self.items_bim_path}/{year}/items_bim_{periodo}.json', 'w', encoding='utf-8') as file:
                file.write(dumps(data_dict, ensure_ascii=False, indent=4))
    
    def get_items_predict(self):
        data_predict_presencial = self.data_predict_presencial.copy()
        data_predict_presencial_01 = data_predict_presencial[
            ['PERIODO', 'SEDE', 'CURSO_ACTUAL', 'HORARIO', 'AULA',  
                 'FORECAST_ALUMN', 'FRECUENCIA', 'NIVEL', 'DURACION', 'PERIODO_FRANJA', 
                 'TURNOS', 'DIAS', 'VAC_ACAD_ESTANDAR']].copy()
        if self.full:
            for group, data in data_predict_presencial_01.groupby('PERIODO'):
                assert isinstance(group, int)
                periodo = group
                year = periodo//100
                Path(f'{self.data_path}/{self.items_predict_path}').mkdir(parents=True, exist_ok=True)
                Path(f'{self.data_path}/{self.items_predict_path}/{year}').mkdir(parents=True, exist_ok=True)
                data_dict = data.to_dict(
                    orient='records')
                if len(data_dict) == 0:
                    continue
                with open(f'{self.data_path}/{self.items_predict_path}/{year}/items_predict_{periodo}.json', 'w', encoding='utf-8') as file:
                    file.write(dumps(data_dict, ensure_ascii=False, indent=4))
        else:
            assert isinstance(self.periodo_predict, int)
            periodo = self.periodo_predict
            year = periodo//100
            Path(f'{self.data_path}/{self.items_predict_path}').mkdir(parents=True, exist_ok=True)
            Path(f'{self.data_path}/{self.items_predict_path}/{year}').mkdir(parents=True, exist_ok=True)
            data_predict_presencial_dict = data_predict_presencial_01.to_dict(
                orient='records')
            with open(f'{self.data_path}/{self.items_predict_path}/{year}/items_predict_{periodo}.json', 'w', encoding='utf-8') as file:
                file.write(dumps(data_predict_presencial_dict, ensure_ascii=False, indent=4))
    
    def get_periodo_franja(self):
        df_dim_horario = self.get_dim_horario()
        df_dim_horario_dict = df_dim_horario.to_dict(
            orient='records')
        with open(f'{self.data_path}/{self.dim_path}/periodo_franja.json', 'w', encoding='utf-8') as file:
            file.write(dumps(df_dim_horario_dict, ensure_ascii=False, indent=4))
    
    def get_aulas(self):
        df_dim_aulas = self.get_dim_aulas()
        df_dim_aulas_dict = df_dim_aulas.to_dict(
            orient='records')
        with open(f'{self.data_path}/{self.dim_path}/aulas.json', 'w', encoding='utf-8') as file:
            file.write(dumps(df_dim_aulas_dict, ensure_ascii=False, indent=4))
    
    def get_frecuencia(self):
        df_dim_sedes = self.data_prog_presencial.copy()
        df_dim_sedes_dict = df_dim_sedes.to_dict(
            orient='records')
        with open(f'{self.data_path}/{self.dim_path}/frecuencia.json', 'w', encoding='utf-8') as file:
            file.write(dumps(df_dim_sedes_dict, ensure_ascii=False, indent=4))
    
    def get_room_log(self):

        df_dim_aulas = self.get_dim_aulas()
        df_dim_dias_turnos = self.get_dim_dias_turnos()
        df_dim_room_log = df_dim_aulas.merge(
            df_dim_dias_turnos,
            how='cross'
        )
        
        df_dim_room_log['DIAS'] = df_dim_room_log.apply(
            lambda x: {"DIA":x['DIA'],  "TURNOS":x['TURNOS']}, axis=1
        )

        df_dim_room_log_01 = df_dim_room_log.groupby(
            ['PERIODO', 'SEDE', 'N_AULA', 'AFORO'], as_index=False
        ).agg(
            DIAS = pd.NamedAgg(column='DIAS', aggfunc=lambda x: list(x))
        )

        df_dim_room_log_01['AULAS'] = df_dim_room_log_01.apply(
            lambda x: {"AULA":x['N_AULA'], "AFORO":x['AFORO'], "DIAS":x['DIAS']}, axis=1
        )

        df_dim_room_log_02 = df_dim_room_log_01.groupby(
            ['PERIODO', 'SEDE'], as_index=False
        ).agg(
            AULAS = pd.NamedAgg(column='AULAS', aggfunc=lambda x: list(x))
        )
        # print(df_dim_room_log_02['SEDE'].unique())

        if self.full:
            for group, data in df_dim_room_log_02.groupby('PERIODO'):
                assert isinstance(group, int)
                periodo = group
                year = periodo//100
                Path(f'{self.data_path}/{self.room_log_path}').mkdir(parents=True, exist_ok=True)
                Path(f'{self.data_path}/{self.room_log_path}/{year}').mkdir(parents=True, exist_ok=True)
                data_dict = data.to_dict(
                    orient='records')
                if len(data_dict) == 0:
                    continue
                with open(f'{self.data_path}/{self.room_log_path}/{year}/room_log_{periodo}.json', 'w', encoding='utf-8') as file:
                    file.write(dumps(data_dict, ensure_ascii=False, indent=4))
        else:
            assert isinstance(self.periodo_predict, int)
            periodo = self.periodo_predict
            df_dim_room_log_03 = df_dim_room_log_02[df_dim_room_log_02['PERIODO'] == periodo].copy()
            year = periodo//100
            Path(f'{self.data_path}/{self.room_log_path}').mkdir(parents=True, exist_ok=True)
            Path(f'{self.data_path}/{self.room_log_path}/{year}').mkdir(parents=True, exist_ok=True)
            data_room_log_dict = df_dim_room_log_03.to_dict(
                orient='records')
            with open(f'{self.data_path}/{self.room_log_path}/{year}/room_log_{periodo}.json', 'w', encoding='utf-8') as file:
                file.write(dumps(data_room_log_dict, ensure_ascii=False, indent=4))
        
    def get_rewards_sedes(self):
        df_dim_rewards_sedes = self.get_dim_rewards_sedes()
        df_dim_rewards_sedes_dict = df_dim_rewards_sedes.to_dict(
            orient='records')
        with open(f'{self.data_path}/{self.dim_path}/rewards_sedes.json', 'w', encoding='utf-8') as file:
            file.write(dumps(df_dim_rewards_sedes_dict, ensure_ascii=False, indent=4))
        
    def transform_all(self):
        # falta validar dim_aulas sand dim_rewards_sedes, este se  debe incluir en validate_data
        self.validate_data()
        self.validate_aulas_and_rewards()
        self.get_items()
        self.get_items_bim()
        self.get_items_predict()
        self.get_aulas()
        self.get_rewards_sedes()
        self.get_room_log()
    

if __name__ == '__main__':
   pass
