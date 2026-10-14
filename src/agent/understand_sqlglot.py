
import sqlglot
from sqlglot import exp

sql = """
SELECT TOP (1) Current_Temperature_C
FROM FDE_VIEWS.VW_ACTIVE_FLEET
ORDER BY Current_Temperature_C DESC
"""

# Parse SQL into an Abstract Syntax Tree
statement = sqlglot.parse_one(sql, read="tsql")

print("1. Statement type:")
print(type(statement).__name__)

print("\n2. Tables/views found:")
for table in statement.find_all(exp.Table):
    print("Schema:", table.db)
    print("Table/View:", table.name)

print("\n3. Columns found:")
for column in statement.find_all(exp.Column):
    print(column.name)

print("\n4. Parsed SQL structure:")
print(statement.to_s())
