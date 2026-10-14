
import os
from pathlib import Path
from typing import TypedDict
from contextlib import closing
from datetime import date

import pyodbc
import sqlglot
from sqlglot import exp
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ==========================================
# SETUP
# ==========================================

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

GROUPED_AVG_SQL = (
    "SELECT [Risk_Classification], "
    "AVG(CAST([Delay_Probability] AS FLOAT)) "
    "AS Average_Delay_Probability "
    "FROM FDE_VIEWS.VW_ACTIVE_FLEET "
    "GROUP BY [Risk_Classification]"
)

RISK_COUNT_SQL = (
    "SELECT COUNT_BIG(*) AS Observation_Count "
    "FROM FDE_VIEWS.VW_ACTIVE_FLEET "
    "WHERE [Risk_Classification] = ?"
)

RISK_COUNT_DATE_SQL = (
    "SELECT COUNT_BIG(*) AS Observation_Count "
    "FROM FDE_VIEWS.VW_ACTIVE_FLEET "
    "WHERE [Risk_Classification] = ? "
    "AND [Timestamp] >= ? "
    "AND [Timestamp] < ?"
)

# ==========================================
# LANGGRAPH STATE
# ==========================================

class AgentState(TypedDict):
    question: str
    generated_sql: str
    safe_sql: str
    sql_result: float | list[dict] | None
    answer: str
    error: str
    risk_filter: str
    date_start: str
    date_end_exclusive: str

# ==========================================
# NODE 1: GENERATE SQL
# ==========================================

def generate_sql(state: AgentState):

    print("\nNODE 1 — Generating SQL")

    
    system_prompt = """
    You generate Microsoft SQL Server T-SQL.

    Approved view:
    FDE_VIEWS.VW_ACTIVE_FLEET

    Supported questions:
    1. Highest recorded temperature
    2. Lowest recorded temperature
    3. Average delay probability grouped by risk classification
    4. Count observations for a specified risk classification
    5. Count observations for a risk classification within a date range.

    Approved risk classifications:
    - High Risk
    - Moderate Risk
    - Low Risk

    Generate SQL using these patterns:

    Highest temperature:
    SELECT MAX(Current_Temperature_C)
    FROM FDE_VIEWS.VW_ACTIVE_FLEET

    Lowest temperature:
    SELECT MIN(Current_Temperature_C)
    FROM FDE_VIEWS.VW_ACTIVE_FLEET

    Average delay probability:
    SELECT Risk_Classification,
           AVG(Delay_Probability)
    FROM FDE_VIEWS.VW_ACTIVE_FLEET
    GROUP BY Risk_Classification

    Count by risk classification:
    SELECT COUNT(*)
    FROM FDE_VIEWS.VW_ACTIVE_FLEET
    WHERE Risk_Classification = 'High Risk'

    Replace 'High Risk' with the requested approved
    risk classification when appropriate.

    Do not include joins, subqueries, or additional filters.
    If unsupported, return UNSUPPORTED.
    Return SQL only, without Markdown.

    For a date-filtered count, generate SQL in this form:

    SELECT COUNT(*)
    FROM FDE_VIEWS.VW_ACTIVE_FLEET
    WHERE Risk_Classification = 'High Risk'
    AND [Timestamp] >= '2024-08-01'
    AND [Timestamp] < '2024-09-01'

    The example covers all of August 2024.

    For a user-requested inclusive date range, use the
    requested first date as the inclusive lower boundary
    and the day after the requested last date as the
    exclusive upper boundary.

    Use ISO dates: YYYY-MM-DD.
    Never invent dates that the user has not specified.
    Do not use joins, subqueries, or additional filters.
    """


    response = llm.invoke([
        ("system", system_prompt),
        ("human", state["question"])
    ])

    generated_sql = response.content.strip()

    print("Generated SQL:")
    print(generated_sql)

    return {"generated_sql": generated_sql}


# ==========================================
# HELPER: BUILD TRUSTED SQL
# ==========================================

def build_approved_sql(candidate_sql: str):

    statements = sqlglot.parse(
        candidate_sql,
        read="tsql",
        error_level="RAISE"
    )

    if len(statements) != 1:
        raise ValueError("Expected exactly one SQL statement.")

    statement = statements[0]

    if not isinstance(statement, exp.Select):
        raise ValueError("Only SELECT is supported.")

    # Reject all unrecognized SELECT clauses.
    allowed_parts = {
        "expressions", "from_", "order", "limit"
    }

    for key, value in statement.args.items():
        if key not in allowed_parts and value is not None:
            if value is not False and value != []:
                raise ValueError(
                    f"Unsupported SQL feature: {key}"
                )

    # Require one direct source: our approved view.
    source = statement.args.get("from_")

    if source is None or not isinstance(
        source.this, exp.Table
    ):
        raise ValueError("Direct approved view required.")

    table = source.this

    if (
        table.name.lower() != "vw_active_fleet"
        or table.db.lower() != "fde_views"
        or table.catalog
        or table.alias
    ):
        raise ValueError("Unapproved database source.")

    # Reject table hints and other extra table features.
    for key, value in table.args.items():
        if key not in {"this", "db", "catalog"}:
            if value is not None and value != []:
                raise ValueError("Unsupported table feature.")

    if len(statement.expressions) != 1:
        raise ValueError("Expected one result expression.")

    expression = statement.expressions[0]

    if isinstance(expression, exp.Alias):
        expression = expression.this

    # Support MIN() and MAX()
    if isinstance(expression, (exp.Min, exp.Max)):

        column = expression.this

        if statement.args.get("order") is not None:
            raise ValueError("Unexpected ORDER BY.")

        if statement.args.get("limit") is not None:
            raise ValueError("Unexpected TOP/LIMIT.")

        operation = (
            "MAX" if isinstance(expression, exp.Max)
            else "MIN"
        )

    # Also recognize TOP (1) with temperature sorting.
    elif isinstance(expression, exp.Column):

        column = expression

        order = statement.args.get("order")
        limit = statement.args.get("limit")

        if order is None or limit is None:
            raise ValueError(
                "TOP (1) and ORDER BY are required."
            )

        limit_value = limit.expression

        if not (
            isinstance(limit_value, exp.Literal)
            and not limit_value.is_string
            and str(limit_value.this) == "1"
        ):
            raise ValueError("Only TOP (1) is supported.")

        if len(order.expressions) != 1:
            raise ValueError("Only one sort column allowed.")

        sort_item = order.expressions[0]

        if not isinstance(sort_item.this, exp.Column):
            raise ValueError("Unsupported sorting.")

        sort_column = sort_item.this

        if (
            sort_column.name.lower()
            != "current_temperature_c"
            or sort_column.table
            or sort_column.db
        ):
            raise ValueError("Invalid sort column.")

        operation = (
            "MAX" if sort_item.args.get("desc")
            else "MIN"
        )

    else:
        raise ValueError("Unsupported SQL operation.")

    if not isinstance(column, exp.Column):
        raise ValueError("Expected a temperature column.")

    if (
        column.name.lower() != "current_temperature_c"
        or column.table
        or column.db
    ):
        raise ValueError("Unapproved column.")

    # Construct trusted SQL from our own template.
    return (
        f"SELECT {operation}([Current_Temperature_C]) "
        f"AS Temperature_C "
        f"FROM FDE_VIEWS.VW_ACTIVE_FLEET"
    )



def build_approved_grouped_average_sql(candidate_sql: str):

    statements = sqlglot.parse(
        candidate_sql,
        read="tsql",
        error_level="RAISE"
    )

    if len(statements) != 1:
        raise ValueError("Expected exactly one statement.")

    statement = statements[0]

    if not isinstance(statement, exp.Select):
        raise ValueError("Only SELECT is supported.")

    # This new pattern supports only SELECT, FROM, GROUP BY
    allowed_parts = {"expressions", "from_", "group"}

    for key, value in statement.args.items():
        if key not in allowed_parts and value is not None:
            raise ValueError("Unsupported SQL clause.")

    # Verify the approved view
    source = statement.args.get("from_")

    if source is None or not isinstance(source.this, exp.Table):
        raise ValueError("Approved view required.")

    table = source.this

    if (
        table.name.lower() != "vw_active_fleet"
        or table.db.lower() != "fde_views"
        or table.catalog
        or table.alias
    ):
        raise ValueError("Unapproved data source.")

    if any(
        value is not None
        for key, value in table.args.items()
        if key not in {"this", "db", "catalog"}
    ):
        raise ValueError("Unsupported table feature.")

    # Exactly two selected expressions are required
    if len(statement.expressions) != 2:
        raise ValueError("Expected category and average.")

    category_column = statement.expressions[0]
    average_expression = statement.expressions[1]

    # Remove the display alias if present
    if isinstance(average_expression, exp.Alias):
        average_expression = average_expression.this

    # Verify the grouping column
    if not isinstance(category_column, exp.Column):
        raise ValueError("Expected risk classification.")

    if (
        category_column.name.lower() != "risk_classification"
        or category_column.table
        or category_column.db
    ):
        raise ValueError("Invalid category column.")

    # Verify AVG(Delay_Probability)
    if not isinstance(average_expression, exp.Avg):
        raise ValueError("Expected AVG calculation.")

    average_column = average_expression.this

    if not isinstance(average_column, exp.Column):
        raise ValueError("Expected delay probability column.")

    if (
        average_column.name.lower() != "delay_probability"
        or average_column.table
        or average_column.db
    ):
        raise ValueError("Invalid average column.")

    if set(average_expression.args) - {"this"}:
        raise ValueError("Unsupported AVG options.")

    # Verify GROUP BY Risk_Classification
    group = statement.args.get("group")

    if group is None or len(group.expressions) != 1:
        raise ValueError("Expected one GROUP BY column.")

    group_column = group.expressions[0]

    if not isinstance(group_column, exp.Column):
        raise ValueError("Invalid grouping expression.")

    if (
        group_column.name.lower() != "risk_classification"
        or group_column.table
        or group_column.db
    ):
        raise ValueError("Invalid GROUP BY column.")

    # Return our own trusted SQL, not the LLM's raw SQL
    return GROUPED_AVG_SQL


def build_approved_risk_count_sql(candidate_sql: str):

    statements = sqlglot.parse(
        candidate_sql,
        read="tsql",
        error_level="RAISE"
    )

    if len(statements) != 1:
        raise ValueError("Expected one SQL statement.")

    statement = statements[0]

    if not isinstance(statement, exp.Select):
        raise ValueError("Only SELECT is supported.")

    # Allow only SELECT, FROM and WHERE
    allowed_parts = {"expressions", "from_", "where"}

    for key, value in statement.args.items():
        if key not in allowed_parts and value is not None:
            raise ValueError("Unsupported SQL clause.")

    # Check the database view
    source = statement.args.get("from_")

    if source is None or not isinstance(source.this, exp.Table):
        raise ValueError("Approved view required.")

    table = source.this

    if (
        table.name.lower() != "vw_active_fleet"
        or table.db.lower() != "fde_views"
        or table.catalog
        or table.alias
    ):
        raise ValueError("Unapproved data source.")

    # Reject extra table features such as hints
    for key, value in table.args.items():
        if key not in {"this", "db", "catalog"}:
            if value is not None and value != []:
                raise ValueError("Unsupported table feature.")

    # Require exactly COUNT(*)
    if len(statement.expressions) != 1:
        raise ValueError("Expected one COUNT expression.")

    count_expression = statement.expressions[0]

    if isinstance(count_expression, exp.Alias):
        count_expression = count_expression.this

    if not isinstance(count_expression, exp.Count):
        raise ValueError("Expected COUNT(*).")

    if not isinstance(count_expression.this, exp.Star):
        raise ValueError("Only COUNT(*) is allowed.")

    for key, value in count_expression.args.items():
        if key != "this" and value not in (None, [], False):
            raise ValueError("Unsupported COUNT option.")

    # Validate WHERE Risk_Classification = 'High Risk'
    where = statement.args.get("where")

    if where is None or not isinstance(where.this, exp.EQ):
        raise ValueError("Expected a risk classification filter.")

    condition = where.this
    column = condition.left
    value = condition.right

    if not isinstance(column, exp.Column):
        raise ValueError("Invalid filter column.")

    if (
        column.name.lower() != "risk_classification"
        or column.table
        or column.db
    ):
        raise ValueError("Only risk classification filtering is allowed.")

    if not isinstance(value, exp.Literal) or not value.is_string:
        raise ValueError("Expected a quoted risk classification.")

    risk_classification = value.this

    allowed_risks = {
        "High Risk",
        "Moderate Risk",
        "Low Risk"
    }

    if risk_classification not in allowed_risks:
        raise ValueError("Unsupported risk classification.")

    # Return trusted SQL and the validated parameter
    return RISK_COUNT_SQL, risk_classification


def build_approved_dated_risk_count_sql(candidate_sql: str):

    # Parse every SQL statement
    statements = sqlglot.parse(
        candidate_sql,
        read="tsql",
        error_level="RAISE"
    )

    if len(statements) != 1:
        raise ValueError("Expected exactly one SQL statement.")

    statement = statements[0]

    if not isinstance(statement, exp.Select):
        raise ValueError("Only SELECT is supported.")

    # Only SELECT, FROM and WHERE are allowed
    allowed_parts = {"expressions", "from_", "where"}

    for key, value in statement.args.items():
        if key not in allowed_parts and value not in (None, [], False):
            raise ValueError(f"Unsupported SQL clause: {key}")

    # Validate the source view
    source = statement.args.get("from_")

    if source is None or not isinstance(source.this, exp.Table):
        raise ValueError("Approved view required.")

    table = source.this

    if (
        table.name.lower() != "vw_active_fleet"
        or table.db.lower() != "fde_views"
        or table.catalog
        or table.alias
    ):
        raise ValueError("Unapproved database source.")

    for key, value in table.args.items():
        if key not in {"this", "db", "catalog"}:
            if value not in (None, [], False):
                raise ValueError("Unsupported table feature.")

    # Require exactly COUNT(*)
    if len(statement.expressions) != 1:
        raise ValueError("Expected one COUNT expression.")

    expression = statement.expressions[0]

    if isinstance(expression, exp.Alias):
        expression = expression.this

    if not isinstance(expression, exp.Count):
        raise ValueError("Expected COUNT(*).")

    if not isinstance(expression.this, exp.Star):
        raise ValueError("Only COUNT(*) is allowed.")

    if any(
        value not in (None, [], False)
        for key, value in expression.args.items()
        if key != "this"
    ):
        raise ValueError("Unsupported COUNT option.")

    # Retrieve WHERE conditions
    where = statement.args.get("where")

    if where is None:
        raise ValueError("Date and risk filters are required.")

    # Split AND conditions into individual comparisons
    def split_and(node):
        if isinstance(node, exp.And):
            return split_and(node.left) + split_and(node.right)
        return [node]

    conditions = split_and(where.this)

    if len(conditions) != 3:
        raise ValueError("Exactly three filters are required.")

    filters = {}

    for condition in conditions:

        if isinstance(condition, exp.EQ):
            filter_name = "risk"
            expected_column = "risk_classification"

        elif isinstance(condition, exp.GTE):
            filter_name = "start"
            expected_column = "timestamp"

        elif isinstance(condition, exp.LT):
            filter_name = "end"
            expected_column = "timestamp"

        else:
            raise ValueError("Unsupported filter operator.")

        column = condition.left
        value = condition.right

        if not isinstance(column, exp.Column):
            raise ValueError("Invalid filter column.")

        if (
            column.name.lower() != expected_column
            or column.table
            or column.db
        ):
            raise ValueError("Unapproved filter column.")

        if not isinstance(value, exp.Literal) or not value.is_string:
            raise ValueError("Filter must be a quoted literal.")

        if filter_name in filters:
            raise ValueError("Duplicate filter.")

        filters[filter_name] = value.this

    if set(filters) != {"risk", "start", "end"}:
        raise ValueError("Required filters are missing.")

    # Validate the risk category
    risk_classification = filters["risk"]

    if risk_classification not in {
        "High Risk",
        "Moderate Risk",
        "Low Risk"
    }:
        raise ValueError("Invalid risk classification.")

    # Validate both dates
    start_date = date.fromisoformat(filters["start"])
    end_date = date.fromisoformat(filters["end"])

    if (
        start_date.isoformat() != filters["start"]
        or end_date.isoformat() != filters["end"]
    ):
        raise ValueError("Use YYYY-MM-DD date format.")

    if end_date <= start_date:
        raise ValueError("End date must be after start date.")

    if (end_date - start_date).days > 366:
        raise ValueError("Date range cannot exceed 366 days.")

    # Build SQL ourselves from approved components
    return (
        RISK_COUNT_DATE_SQL,
        risk_classification,
        start_date.isoformat(),
        end_date.isoformat()
    )



# ==========================================
# NODE 2: VALIDATE SQL
# ==========================================



def validate_sql_node(state: AgentState):

    print("\nNODE 2 — Validating SQL")

    generated_sql = state["generated_sql"].strip()

    if generated_sql.upper() == "UNSUPPORTED":
        return {
            "safe_sql": "",
            "risk_filter": "",
            "date_start": "",
            "date_end_exclusive": "",
            "error": "This question is not supported yet."
        }

    try:
        risk_filter = ""
        date_start = ""
        date_end_exclusive = ""

        if "GROUP BY" in generated_sql.upper():

            safe_sql = build_approved_grouped_average_sql(
                generated_sql
            )

        elif "COUNT(" in generated_sql.upper().replace(
            " ", ""
        ).replace("\n", ""):

            # Examine the parsed WHERE structure
            parsed = sqlglot.parse_one(
                generated_sql,
                read="tsql"
            )
            where = parsed.args.get("where")

            # A dated count contains multiple AND conditions
            if (
                where is not None
                and isinstance(where.this, exp.And)
            ):

                (
                    safe_sql,
                    risk_filter,
                    date_start,
                    date_end_exclusive
                ) = build_approved_dated_risk_count_sql(
                    generated_sql
                )

            else:

                safe_sql, risk_filter = (
                    build_approved_risk_count_sql(
                        generated_sql
                    )
                )

        else:

            safe_sql = build_approved_sql(
                generated_sql
            )

        print("Validation passed.")
        print("Trusted SQL:", safe_sql)

        if risk_filter:
            print("Risk filter:", risk_filter)

        if date_start:
            print("Start date:", date_start)
            print("End date (exclusive):", date_end_exclusive)

        return {
            "safe_sql": safe_sql,
            "risk_filter": risk_filter,
            "date_start": date_start,
            "date_end_exclusive": date_end_exclusive,
            "error": ""
        }

    except Exception as error:

        print("Validation rejected:", error)

        return {
            "safe_sql": "",
            "risk_filter": "",
            "date_start": "",
            "date_end_exclusive": "",
            "error": str(error)
        }



# Conditional routing after validation
def route_after_validation(state: AgentState):

    if state["error"]:
        print("ROUTER — Stopping execution")
        return "stop"

    print("ROUTER — Proceeding to SQL Server")
    return "execute"


# ==========================================
# NODE 3: EXECUTE SQL
# ==========================================



def execute_sql(state: AgentState):

    print("\nNODE 3 — Executing SQL on EC2")

    safe_sql = state["safe_sql"]

    # Step 1: Allow only trusted SQL queries
    approved_queries = {
        "SELECT MIN([Current_Temperature_C]) AS Temperature_C "
        "FROM FDE_VIEWS.VW_ACTIVE_FLEET",

        "SELECT MAX([Current_Temperature_C]) AS Temperature_C "
        "FROM FDE_VIEWS.VW_ACTIVE_FLEET",

        GROUPED_AVG_SQL,

        RISK_COUNT_SQL,

        RISK_COUNT_DATE_SQL
    }

    if safe_sql not in approved_queries:
        raise ValueError("SQL execution denied.")

    # Step 2: Get read-only credentials from .env
    username = os.getenv("SQL_RO_USER")
    password = os.getenv("SQL_RO_PASSWORD")

    if username != "USR_FDE_RO" or not password:
        raise ValueError("Read-only SQL credentials required.")

    # Step 3: Build SQL Server connection string
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

    # Step 4: Connect to SQL Server on EC2
    with closing(
        pyodbc.connect(
            connection_string,
            timeout=10,
            readonly=True
        )
    ) as connection:

        connection.timeout = 15

        cursor = connection.cursor()

        # Step 5: Execute the appropriate query

        # Query Type 1: COUNT with risk and date filters
        if safe_sql == RISK_COUNT_DATE_SQL:

            risk_filter = state["risk_filter"]
            date_start = state["date_start"]
            date_end = state["date_end_exclusive"]

            if not risk_filter or not date_start or not date_end:
                raise ValueError(
                    "Risk classification and date range are required."
                )

            print("Executing date-filtered COUNT query")
            print("Risk filter:", risk_filter)
            print("Start date:", date_start)
            print("End date (exclusive):", date_end)

            cursor.execute(
                safe_sql,
                risk_filter,
                date_start,
                date_end
            )

        # Query Type 2: COUNT by risk classification
        elif safe_sql == RISK_COUNT_SQL:

            risk_filter = state["risk_filter"]

            if not risk_filter:
                raise ValueError(
                    "Risk classification is required."
                )

            print("Executing COUNT query")
            print("Risk filter:", risk_filter)

            cursor.execute(
                safe_sql,
                risk_filter
            )

        # Query Types 3, 4, 5:
        # MIN temperature, MAX temperature, grouped AVG
        else:

            print("Executing approved analytics query")
            cursor.execute(safe_sql)

        # Step 6: Verify a result set was returned
        if cursor.description is None:
            raise RuntimeError(
                "SQL execution did not return a result set."
            )

        # Step 7: Retrieve SQL results

        # COUNT queries return one integer
        if safe_sql in (RISK_COUNT_SQL, RISK_COUNT_DATE_SQL):

            row = cursor.fetchone()

            result = int(row[0]) if row else None

        # GROUP BY query returns multiple records
        elif safe_sql == GROUPED_AVG_SQL:

            rows = cursor.fetchall()

            result = [
                {
                    "risk_classification": row[0],
                    "average_delay_probability": (
                        round(float(row[1]), 4)
                        if row[1] is not None
                        else None
                    )
                }
                for row in rows
            ]

        # MIN/MAX temperature returns one number
        else:

            row = cursor.fetchone()

            result = (
                float(row[0])
                if row and row[0] is not None
                else None
            )

    # Step 8: Return results to LangGraph State
    print("SQL Server result:", result)

    return {"sql_result": result}




# ==========================================
# NODE 4: GENERATE ANSWER
# ==========================================

def generate_answer(state: AgentState):

    print("\nNODE 4 — Generating AI answer")

    response = llm.invoke([
        (
            "system",
            """
            You are a Cold Chain Logistics analyst.

            Explain the SQL result clearly.

            Rules:
            - Use only the provided SQL result.
            - The observations are historical.
            - Do not claim this is live shipment data.
            - Do not invent shipment IDs or temperature limits.
            - Do not claim a temperature excursion occurred.
            - If the result is unavailable, say so.
            """
        ),
        (
            "human",
            f"""
            Business question:
            {state["question"]}

            Approved SQL executed:
            {state["safe_sql"]}

            Verified SQL Server result:
            {state["sql_result"]}

            Instructions:
            1. Answer using only the verified SQL result.
            2. Identify the metric being measured.
            3. Use Celsius only for temperature values.
            4. For delay probabilities, show the decimal
                value and optionally its percentage equivalent.
            5. If there are multiple groups, use a table.
            6. Do not invent explanations for risk classifications.
            7. Clearly state that results are historical.
            8. Do not confuse average predicted delay probability
            with the actual observed delay rate.
            """
        )
    ])

    return {"answer": response.content}


# ==========================================
# BUILD THE FOUR-NODE LANGGRAPH
# ==========================================

builder = StateGraph(AgentState)

builder.add_node("generate_sql", generate_sql)
builder.add_node("validate_sql", validate_sql_node)
builder.add_node("execute_sql", execute_sql)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "validate_sql")

builder.add_conditional_edges(
    "validate_sql",
    route_after_validation,
    {
        "execute": "execute_sql",
        "stop": END
    }
)

builder.add_edge("execute_sql", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    question = "How many High Risk observations occurred in August 2024?"

    initial_state = {
        "question": question,
        "generated_sql": "",
        "safe_sql": "",
        "sql_result": None,
        "answer": "",
        "error": "",
        "risk_filter": "",
        "date_start": "",
        "date_end_exclusive": ""
    }

    result = graph.invoke(initial_state)

    if result["error"]:
        print("\nREQUEST REJECTED:")
        print(result["error"])
    else:
        print("\n========== FINAL AI ANSWER ==========")
        print(result["answer"])
