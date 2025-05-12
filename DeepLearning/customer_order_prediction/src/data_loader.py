import pandas as pd
import psycopg2
from src.config import DB_CONFIG


def load_data(sql_path):
    
    with open(sql_path,"r") as file:
        query = file.read()

    connection = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql(query,connection)
    connection.close()
    return df