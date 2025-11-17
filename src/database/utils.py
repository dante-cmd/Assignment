
from sqlalchemy import create_engine, text
from montydb import MontyClient  # type: ignore
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import os
from datetime import datetime
from models.models import (
            FactProgAcad, FactPredict, DimHorario, DimAulas, DimSedes,
            DimRewardsSedes, DimVacAcad, DimHorariosAtencion, DimCursos,
             FactProvicional, DimDias, DimDiasTurnos, Base
        )


# Map table names to model classes
table_model_map = {
    'fact_prog_acad': FactProgAcad,
    'fact_predict': FactPredict,
    'dim_horario': DimHorario,
    'dim_aulas': DimAulas,
    'dim_sedes': DimSedes,
    'dim_rewards_sedes': DimRewardsSedes,
    'dim_vac_acad': DimVacAcad,
    'dim_horarios_atencion': DimHorariosAtencion,
    'dim_cursos': DimCursos,
    'fact_provicional': FactProvicional,
    'dim_dias': DimDias,
    'dim_dias_turnos': DimDiasTurnos
    }

         
def insert_dataframe(session, table_name: str, df: pd.DataFrame, if_exists: str = 'replace', 
                        chunksize: int = 1000) -> None:
        """
        Insert data from a pandas DataFrame into the specified table with proper type handling.
        
        Args:
            table_name: Name of the target table
            df: DataFrame containing the data to insert
            if_exists: How to behave if the table already exists
                     - 'fail': Raise a ValueError
                     - 'replace': Drop the table before inserting new values
                     - 'append': Insert new values to the existing table
            chunksize: Number of rows to insert at a time
        """
        active_session = session()
        try:
            # active_session = session()
            # Convert datetime columns to ISO format strings
            for col in df.select_dtypes(include=['datetime64[ns]']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            if table_name not in table_model_map:
                raise ValueError(f"Table {table_name} not supported by ORM")
                
            model_class = table_model_map[table_name]
            
            # Handle different if_exists modes
            if if_exists == 'replace':
                # Delete all existing records for this table
                active_session.query(model_class).delete()
                active_session.commit()
            
            # Convert DataFrame to list of dictionaries and then to model instances
            records = df.to_dict('records')
            model_instances = []
            for record in records:
                # Remove any columns not in the model
                filtered_record = {k.upper(): v for k, v in record.items()} # type: ignore
                model_instances.append(model_class(**filtered_record))
            
            # Bulk insert the records
            active_session.bulk_save_objects(model_instances)
            active_session.commit()
            
            print(f"Successfully inserted/updated {len(df)} rows in {table_name}")
            
        except Exception as e:
            print(f"Error inserting data into {table_name}: {e}")
            active_session.rollback()
            raise
        finally:
            active_session.close()
            
def insert_dataframe_by_periodo(session, table_name: str, df: pd.DataFrame, 
                        periodos: List[int]) -> None:
        """
        Insert data from a pandas DataFrame into the specified table with proper type handling.
        
        Args:
            table_name: Name of the target table
            df: DataFrame containing the data to insert
            periodos: list of periodos to insert
        """
        active_session = session()
        try:
            # session = self.Session()
            # Convert datetime columns to ISO format strings
            for col in df.select_dtypes(include=['datetime64[ns]']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            if table_name not in table_model_map:
                raise ValueError(f"Table {table_name} not supported by ORM")
                
            model_class = table_model_map[table_name]

            # delete rows where periodo is in periodos
            active_session.query(model_class).filter(model_class.PERIODO.in_(periodos)).delete()
            active_session.commit()
            
            # Handle different if_exists modes
            # if if_exists == 'replace':
            #     # Delete all existing records for this table
            #     session.query(model_class).delete()
            #     session.commit()
            
            # Convert DataFrame to list of dictionaries and then to model instances
            records = df.to_dict('records')
            model_instances = []
            for record in records:
                # Remove any columns not in the model
                filtered_record = {k.upper(): v for k, v in record.items()} # type: ignore
                model_instances.append(model_class(**filtered_record))
            
            # Bulk insert the records
            active_session.bulk_save_objects(model_instances)
            active_session.commit()
            
            print(f"Successfully inserted/updated {len(df)} rows in {table_name}")
            
        except Exception as e:
            print(f"Error inserting data into {table_name}: {e}")
            active_session.rollback()
            raise
        finally:
            active_session.close()
    
def query_to_dataframe(session, engine, query: str, params: tuple = ()) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a pandas DataFrame.
        
        Args:
            query: SQL query to execute
            params: Parameters for the SQL query
            
        Returns:
            pd.DataFrame: Query results as a DataFrame
        """
        active_session = session()
        try:
            # Use pandas' read_sql with the SQLAlchemy engine directly
            result = pd.read_sql_query(query, con=engine, params=params)
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
        finally:
            active_session.close()


if __name__ == "__main__":
     pass
    # import argparse
    
    # parser = argparse.ArgumentParser(description='Database management for classroom scheduling')
    # parser.add_argument('--init', action='store_true', help='Initialize the database')
    # parser.add_argument('--db-path', type=str, required=True, help='Path to the database file')
    # parser.add_argument('--load-data', action='store_true', help='Load all data into the database')
    # parser.add_argument('--drop-all', action='store_true', help='Drop the database if it exists')
    # parser.add_argument('--data-dir', default='raw', help='Path to the data directory')
    
    # args = parser.parse_args()
    
    # if args.init or args.load_data:
    #     print("Initializing database...")
    #     db = DatabaseManager(args.db_path, args.drop_all)
    #     print(f"Database created at: {os.path.abspath(db.db_path)}")
    
    # if not (args.init or args.load_data):
    #     print("No action specified. Use --init to initialize the database or --load-data to load all data.")
	