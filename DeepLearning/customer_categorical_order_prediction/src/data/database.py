"""
Database connection and query module.
"""
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv
from src.config import DB_CONFIG

# Load environment variables
load_dotenv()

def get_database_connection() -> Engine:
    """
    Create database connection using environment variables.
    
    Returns:
        Engine: SQLAlchemy database engine
    """
    # Get database credentials from config
    db_host = DB_CONFIG['host']
    db_port = DB_CONFIG['port']
    db_name = DB_CONFIG['database']
    db_user = DB_CONFIG['user']
    db_password = DB_CONFIG['password']
    
    # Create connection string
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Create engine
    engine = create_engine(connection_string)
    
    return engine

def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame.
    
    Args:
        query (str): SQL query to execute
        params (Dict[str, Any], optional): Query parameters
        
    Returns:
        pd.DataFrame: Query results
    """
    engine = get_database_connection()
    
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query), params or {})
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        raise Exception(f"Error executing query: {str(e)}")
    finally:
        engine.dispose()

def get_customer_category_data() -> pd.DataFrame:
    """
    Get customer category purchase data.
    
    Returns:
        pd.DataFrame: Customer category data
    """
    query = """
    WITH customer_category_stats AS (
        SELECT 
            c.customer_id,
            p.category_id,
            COUNT(DISTINCT o.order_id) as order_count,
            SUM(od.unit_price * od.quantity * (1 - od.discount)) as total_amount,
            MAX(o.order_date) as last_order_date
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_details od ON o.order_id = od.order_id
        JOIN products p ON od.product_id = p.product_id
        GROUP BY c.customer_id, p.category_id
    )
    SELECT 
        ccs.customer_id,
        c.company_name,
        cat.category_name,
        ccs.order_count,
        ccs.total_amount,
        ccs.last_order_date
    FROM customer_category_stats ccs
    JOIN customers c ON ccs.customer_id = c.customer_id
    JOIN categories cat ON ccs.category_id = cat.category_id
    ORDER BY ccs.customer_id, ccs.total_amount DESC;
    """
    
    return execute_query(query)

def get_customer_order_history() -> pd.DataFrame:
    """
    Get detailed customer order history.
    
    Returns:
        pd.DataFrame: Customer order history
    """
    query = """
    SELECT 
        c.customer_id,
        c.company_name,
        o.order_id,
        o.order_date,
        p.category_id,
        cat.category_name,
        od.unit_price * od.quantity * (1 - od.discount) as total_amount
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_details od ON o.order_id = od.order_id
    JOIN products p ON od.product_id = p.product_id
    JOIN categories cat ON p.category_id = cat.category_id
    ORDER BY c.customer_id, o.order_date;
    """
    
    df = execute_query(query)
    print("Veritabanından gelen sütunlar:", df.columns.tolist())
    print("\nİlk 5 satır:")
    print(df.head())
    return df

def get_category_metrics() -> pd.DataFrame:
    """
    Get category-level metrics.
    
    Returns:
        pd.DataFrame: Category metrics
    """
    query = """
    WITH category_stats AS (
        SELECT 
            p.category_id,
            COUNT(DISTINCT o.customer_id) as unique_customers,
            COUNT(DISTINCT o.order_id) as total_orders,
            SUM(od.unit_price * od.quantity * (1 - od.discount)) as total_revenue,
            AVG(od.unit_price * od.quantity * (1 - od.discount)) as avg_order_value
        FROM orders o
        JOIN order_details od ON o.order_id = od.order_id
        JOIN products p ON od.product_id = p.product_id
        GROUP BY p.category_id
    )
    SELECT 
        cat.category_id,
        cat.category_name,
        cs.unique_customers,
        cs.total_orders,
        cs.total_revenue,
        cs.avg_order_value
    FROM categories cat
    JOIN category_stats cs ON cat.category_id = cs.category_id
    ORDER BY cs.total_revenue DESC;
    """
    
    return execute_query(query) 