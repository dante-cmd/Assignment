
from sqlalchemy import create_engine, text
# from montydb import MontyClient  # type: ignore
from mongita import MongitaClientDisk
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import os
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///db/sqlite.db"
MONTY_DATABASE_URL = "./db/mongodb"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_sqlite_session():
    return SessionLocal

def get_monty_client():
    client = MongitaClientDisk(MONTY_DATABASE_URL)
    return client

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
	