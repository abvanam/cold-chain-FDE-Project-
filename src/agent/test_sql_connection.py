
import os
from pathlib import Path

import pyodbc
from dotenv import load_dotenv

# Load the project's .env file
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

# Read database connection settings
server = os.getenv("SQL_RO_HOST", "127.0.0.1")
port = os.getenv("SQL_RO_PORT", "11433")
database = os.getenv("SQL_RO_DATABASE", "master")
username = os.getenv("SQL_RO_USER")
password = os.getenv("SQL_RO_PASSWORD")

if not username or not password:
    raise ValueError("SQL_RO_USER or SQL_RO_PASSWORD is missing from .env")

# Set up SQL Server connection
connection_string = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={server},{port};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=10;"
)

# Connect and execute a read-only query
with pyodbc.connect(connection_string) as connection:
    cursor = connection.cursor()

    cursor.execute("""
        SELECT TOP (5)
            [Timestamp],
            [Current_Temperature_C],
            [Risk_Classification],
            [Delay_Probability]
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        ORDER BY [Timestamp] DESC
    """)

    rows = cursor.fetchall()

    print("Successfully connected to SQL Server!")
    print(f"Retrieved {len(rows)} records.\n")

    for row in rows:
        print(row)
