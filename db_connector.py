from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Fetch SQL connection details from .env file
SQL_SERVER = os.getenv('SQL_SERVER')
SQL_DATABASE = os.getenv('SQL_DATABASE')
SQL_USERNAME = os.getenv('SQL_USERNAME')
SQL_PASSWORD = os.getenv('SQL_PASSWORD')

# Create the SQLAlchemy engine using pymssql
def get_engine():
    try:
        # Create the connection string for SQLAlchemy using pymssql
        connection_string = f"mssql+pymssql://{SQL_USERNAME}:{SQL_PASSWORD}@{SQL_SERVER}/{SQL_DATABASE}"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"Error creating SQLAlchemy engine: {e}")
        return None

# Create a function to run queries and return a Pandas DataFrame
def run_query(query):
    engine = get_engine()
    if engine is not None:
        try:
            # Execute the query and load the result into a DataFrame
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    else:
        print("Failed to create engine.")
        return None

    
# Define the query (example: listing tables in the database)
query = """

SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'FSAllAccountsExtraction'
ORDER BY COLUMN_NAME;

"""

# Run the query and get the result as a DataFrame
df = run_query(query)

# If the query was successful, print the DataFrame
if df is not None:
    print(df)
else:
    print("No data returned or query failed.")