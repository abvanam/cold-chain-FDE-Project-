
import os
from pathlib import Path

import sqlglot
from sqlglot import exp

import pyodbc
from contextlib import closing

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Step 1: Load environment variables
project_root = Path.cwd()

if not (project_root / ".env").exists():
    project_root = Path(__file__).resolve().parents[2]

load_dotenv(project_root / ".env")

# Step 2: Initialize OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Step 3: Define the database schema
database_schema = """
Database: master
View: FDE_VIEWS.VW_ACTIVE_FLEET

Available columns:
- Timestamp
- Latitude
- Longitude
- Current_Temperature_C
- Cargo_Condition_Code
- Risk_Classification
- Delay_Probability
- Port_Congestion_Level
- Route_Risk_Index

SQL dialect: Microsoft SQL Server (T-SQL)

The data contains historical fleet observations.
"""

# Step 4: Define the business question
question = (
    "What is the highest recorded temperature?"
)

# Step 5: Instruct the LLM to generate SQL
system_prompt = f"""
You are a Microsoft SQL Server query generator.

Database schema:
{database_schema}

Generate a SQL query to answer the user's question.

Rules:
1. Use only FDE_VIEWS.VW_ACTIVE_FLEET.
2. Use only columns listed in the schema.
3. Generate only a single SELECT statement.
4. Do not generate INSERT, UPDATE, DELETE, DROP,
   ALTER, EXEC, or other modification commands.
5. Use Microsoft SQL Server syntax.
6. For queries returning individual records,
   limit the result to TOP (100).
7. Do not invent columns or tables.
8. If the question cannot be answered using this
   schema, respond with UNSUPPORTED.
9. Return SQL only, without Markdown formatting.
"""

# Step 6: Ask the LLM to generate SQL
response = llm.invoke([
    ("system", system_prompt),
    ("human", question)
])

generated_sql = response.content.strip()

# Step 7: Display the generated SQL
print("\nUSER QUESTION:")
print(question)

print("\nGENERATED SQL:")
print(generated_sql)


# STEP 3: Validate SQL structure and database objects

ALLOWED_COLUMNS = {
    "timestamp",
    "latitude",
    "longitude",
    "current_temperature_c",
    "cargo_condition_code",
    "risk_classification",
    "delay_probability",
    "port_congestion_level",
    "route_risk_index"
}


def validate_sql(sql: str):

    statements = sqlglot.parse(
        sql,
        read="tsql",
        error_level="RAISE"
    )

    # Only one SELECT statement is permitted
    if len(statements) != 1 or not isinstance(
        statements[0], exp.Select
    ):
        raise ValueError("Only one SELECT statement is allowed.")

    statement = statements[0]

    # Keep this first implementation simple:
    # no joins, CTEs, subqueries, or set operations
    blocked_types = (
        exp.Join,
        exp.Subquery,
        exp.CTE,
        exp.Union,
        exp.Intersect,
        exp.Except,
    )

    if any(
        isinstance(node, blocked_types)
        for node in statement.walk()
    ):
        raise ValueError("Complex SQL queries are not allowed yet.")

    # SELECT INTO creates a table, so reject it
    if statement.args.get("into") is not None:
        raise ValueError("SELECT INTO is not allowed.")

    # Verify the source table/view
    tables = list(statement.find_all(exp.Table))

    if len(tables) != 1:
        raise ValueError("Exactly one approved view is required.")

    table = tables[0]

    if (
        table.name.lower() != "vw_active_fleet"
        or table.db.lower() != "fde_views"
        or table.catalog
    ):
        raise ValueError("Unauthorized table or view.")

    # Verify referenced columns
    for column in statement.find_all(exp.Column):

        column_name = column.name.lower()

        if column_name not in ALLOWED_COLUMNS:
            raise ValueError(
                f"Unauthorized column: {column.name}"
            )

        # Restrict SQL functions to approved aggregations
    ALLOWED_FUNCTIONS = (
        exp.Min,
        exp.Max,
        exp.Avg,
        exp.Count,
        exp.Sum,
    )

    for function in statement.find_all(exp.Func):
        if not isinstance(function, ALLOWED_FUNCTIONS):
            raise ValueError(
                f"Unsupported SQL function: {function.sql()}"
            )

    # Do not allow SELECT *.
    # COUNT(*) is allowed for aggregation.
    for star in statement.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):
            raise ValueError("SELECT * is not allowed.")



    return True



def build_approved_temperature_sql(candidate_sql: str) -> str:
    """
    Recognize a narrow set of temperature queries
    and construct a trusted SQL statement.
    """

    statements = sqlglot.parse(
        candidate_sql,
        read="tsql",
        error_level="RAISE"
    )

    if len(statements) != 1 or not isinstance(
        statements[0], exp.Select
    ):
        raise ValueError("Expected one SELECT statement.")

    statement = statements[0]

    # For this first test, reject filters, joins,
    # subqueries, and other SQL features.
    allowed_parts = {"expressions", "from_", "order", "limit"}

    for key, value in statement.args.items():
        if key not in allowed_parts and value is not None:
            if value is not False and value != []:
                raise ValueError(f"Unsupported SQL feature: {key}")

    tables = list(statement.find_all(exp.Table))

    if len(tables) != 1:
        raise ValueError("Expected exactly one approved view.")

    table = tables[0]

    if (
        table.name.lower() != "vw_active_fleet"
        or table.db.lower() != "fde_views"
        or table.catalog
        or table.alias
    ):
        raise ValueError("Unapproved SQL source.")

    # Reject table hints or other extra table features
    for key, value in table.args.items():
        if key not in {"this", "db", "catalog"} and value is not None:
            if value is not False and value != []:
                raise ValueError("Unsupported table feature.")

    if len(statement.expressions) != 1:
        raise ValueError("Only one temperature result is supported.")

    expression = statement.expressions[0]

    if isinstance(expression, exp.Alias):
        expression = expression.this

    operation = None

    # Form 1: SELECT MAX(...) or MIN(...)
    if isinstance(expression, (exp.Max, exp.Min)):
        column = expression.this

        if not isinstance(column, exp.Column):
            raise ValueError("Expected a temperature column.")

        if statement.args.get("order") or statement.args.get("limit"):
            raise ValueError("Unexpected sorting or limit.")

        operation = "MAX" if isinstance(expression, exp.Max) else "MIN"

    # Form 2: SELECT TOP (1) ... ORDER BY ... DESC/ASC
    elif isinstance(expression, exp.Column):
        column = expression

        order = statement.args.get("order")
        limit = statement.args.get("limit")

        if order is None or limit is None:
            raise ValueError("TOP (1) and ORDER BY are required.")

        limit_value = limit.expression

        if not (
            isinstance(limit_value, exp.Literal)
            and not limit_value.is_string
            and str(limit_value.this) == "1"
        ):
            raise ValueError("Only TOP (1) is supported.")

        if len(order.expressions) != 1:
            raise ValueError("Expected one ORDER BY column.")

        sort_item = order.expressions[0]

        if not isinstance(sort_item.this, exp.Column):
            raise ValueError("Invalid sort column.")

        if sort_item.this.name.lower() != "current_temperature_c":
            raise ValueError("Only temperature sorting is allowed.")

        operation = "MAX" if sort_item.args.get("desc") else "MIN"

    else:
        raise ValueError("Unsupported temperature query.")

    if column.name.lower() != "current_temperature_c":
        raise ValueError("Only temperature analysis is supported.")

    # Rebuild the SQL from trusted, fixed components.
    # We do not execute the model-generated string.
    return (
        f"SELECT {operation}([Current_Temperature_C]) "
        f"AS Temperature_C "
        f"FROM FDE_VIEWS.VW_ACTIVE_FLEET"
    )

def execute_trusted_sql(safe_sql: str):
    """
    Execute SQL rebuilt by our approved query builder.
    Uses the read-only SQL Server account.
    """

    # Only the trusted fixed MIN/MAX template is permitted
    approved_queries = {
        "SELECT MIN([Current_Temperature_C]) AS Temperature_C "
        "FROM FDE_VIEWS.VW_ACTIVE_FLEET",

        "SELECT MAX([Current_Temperature_C]) AS Temperature_C "
        "FROM FDE_VIEWS.VW_ACTIVE_FLEET"
    }

    if safe_sql not in approved_queries:
        raise ValueError("Query is not approved for execution.")

    username = os.getenv("SQL_RO_USER")
    password = os.getenv("SQL_RO_PASSWORD")

    if username != "USR_FDE_RO" or not password:
        raise ValueError("Read-only SQL credentials are missing or incorrect.")

    connection_string = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={os.getenv('SQL_RO_HOST')},"
        f"{os.getenv('SQL_RO_PORT')};"
        f"DATABASE={os.getenv('SQL_RO_DATABASE')};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=10;"
    )

    print("\nConnecting to SQL Server on EC2...")

    with closing(
        pyodbc.connect(connection_string, timeout=10, readonly=True)
    ) as connection:

        connection.timeout = 15

        cursor = connection.cursor()
        cursor.execute(safe_sql)

        row = cursor.fetchone()

        return row[0] if row else None


print("\nVALIDATING SQL...")

try:
    validate_sql(generated_sql)
    print("SQL passed initial structural validation!")
    print("Approved view and columns verified.")

except Exception as error:
    print("SQL validation failed:", error)

print("\nNOTE: SQL has not been executed.")


# TEST THE SQL VALIDATOR

test_queries = [
    (
        "Valid temperature query",
        """
        SELECT MAX(Current_Temperature_C)
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        """,
        True
    ),
    (
        "Unauthorized table",
        """
        SELECT Salary
        FROM HR.Employees
        """,
        False
    ),
    (
        "Unauthorized column",
        """
        SELECT Employee_Salary
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        """,
        False
    ),
    (
        "Multiple SQL statements",
        """
        SELECT Current_Temperature_C
        FROM FDE_VIEWS.VW_ACTIVE_FLEET;
        DELETE FROM FDE_VIEWS.VW_ACTIVE_FLEET;
        """,
        False
    ),
    (
        "Unsupported function",
        """
        SELECT SUSER_SNAME()
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        """,
        False
    )
]

print("\n========== VALIDATION TEST RESULTS ==========")

for name, sql, expected in test_queries:

    try:
        validate_sql(sql)
        accepted = True

    except Exception:
        accepted = False

    if accepted == expected:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name}")



print("\n========== APPROVED SQL EXECUTION ==========")

try:
    # Validate and rebuild the generated SQL
    safe_sql = build_approved_temperature_sql(generated_sql)

    print("\nTrusted SQL:")
    print(safe_sql)

    # Execute only the trusted SQL
    temperature = execute_trusted_sql(safe_sql)

    print("\n========== SQL SERVER RESULT ==========")
    print("Temperature (°C):", temperature)

except Exception as error:
    print("Query was not executed successfully:", error)


