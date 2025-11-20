# from montydb import MontyClient  # type: ignore
import sqlite3
from sqlalchemy import create_engine, text
from pathlib import Path
import os
from database.database import SQLALCHEMY_DATABASE_URL, MONTY_DATABASE_URL
from models.models import Base


DROP_ALL_TABLES = False


def create_database():
    try:
        # client = MontyClient(MONTY_DATABASE_URL)
        # print(client)
        # sqlite://  + /db/ 
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        
        if DROP_ALL_TABLES:
            # drop all tables
            # Establish a connection
            with engine.connect() as connection:
                # Query to get all table names
                result = connection.execute(text("""
                    SELECT name FROM sqlite_master WHERE type='table';
                """))
                
                # Get a list of all table names
                tables = result.fetchall()

                # Drop each table
                for table in tables:
                    table_name = table[0]
                    print(f"Dropping table: {table_name}")
                    connection.execute(text(f"DROP TABLE IF EXISTS \"{table_name}\";"))

                # Commit the changes
                connection.commit()

                print("Drop all Table in SQLite")
            # Base.metadata.drop_all(engine)
            Base.metadata.create_all(bind=engine)
            print("Create the tables in SQLite")
            # database_names = client.list_database_names()
            # if len(database_names) == 0:
            #     print("No databases in Monty")
            # for db_name in database_names:
            #     client.drop_database(db_name)
            #     print('Drop all tables in Monty')
        
    except Exception as e:
        print(f"Error al crear la base de datos: {e}")


if __name__ == "__main__":
    # print(Path().absolute())
    create_database()
