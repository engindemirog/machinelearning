import pandas as pd
import psycopg2
from config import DB_CONFIG

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            print("Database connection established")
        except Exception as e:
            print(f"Error connecting to the database : {e}")
            raise
    
    def disconnect(self):
        if self.conn:
            self.conn.close()
            print("Connection closed")
    
    def get_order_data(self):
        query = """
        select 
        od.order_id,
        od.product_id,
        od.unit_price,
        od.quantity,
        od.discount,
        o.customer_id,
        o.order_date,
        p.category_id,
        c.company_name
        from
        orders o inner join order_details od
        on o.order_id=od.order_id
        inner join products p
        on p.product_id=od.product_id
        inner join customers c
        on c.customer_id=o.customer_id
        """

        try:
          df = pd.read_sql_query(query,self.conn)
          return df
        except Exception as e:
          print(f"Error {e}")
          raise

    