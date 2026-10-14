
import sys
from pathlib import Path

import pytest

# Locate our existing agent code
project_root = Path(__file__).resolve().parents[1]
agent_folder = project_root / "src" / "agent"

if str(agent_folder) not in sys.path:
    sys.path.insert(0, str(agent_folder))

# Import functions we already developed
from langgraph_text_to_sql import (
    build_approved_sql,
    build_approved_grouped_average_sql,
    build_approved_risk_count_sql,
    build_approved_dated_risk_count_sql,
    RISK_COUNT_SQL,
    RISK_COUNT_DATE_SQL,
    GROUPED_AVG_SQL,
)


# TEST 1: Highest temperature query should work
def test_max_temperature():

    sql = """
        SELECT MAX(Current_Temperature_C)
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
    """

    safe_sql = build_approved_sql(sql)

    assert "MAX([Current_Temperature_C])" in safe_sql


# TEST 2: Lowest temperature query should work
def test_min_temperature():

    sql = """
        SELECT MIN(Current_Temperature_C)
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
    """

    safe_sql = build_approved_sql(sql)

    assert "MIN([Current_Temperature_C])" in safe_sql


# TEST 3: High Risk count should work
def test_count_high_risk():

    sql = """
        SELECT COUNT(*)
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        WHERE Risk_Classification = 'High Risk'
    """

    safe_sql, risk_filter = build_approved_risk_count_sql(sql)

    assert safe_sql == RISK_COUNT_SQL
    assert risk_filter == "High Risk"


# TEST 4: Average delay probability should work
def test_grouped_average():

    sql = """
        SELECT
            Risk_Classification,
            AVG(Delay_Probability)
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        GROUP BY Risk_Classification
    """

    safe_sql = build_approved_grouped_average_sql(sql)

    assert safe_sql == GROUPED_AVG_SQL


# TEST 5: Date-filtered count should work
def test_date_filtered_count():

    sql = """
        SELECT COUNT(*)
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        WHERE Risk_Classification = 'High Risk'
          AND [Timestamp] >= '2024-08-01'
          AND [Timestamp] < '2024-09-01'
    """

    (
        safe_sql,
        risk_filter,
        start_date,
        end_date
    ) = build_approved_dated_risk_count_sql(sql)

    assert safe_sql == RISK_COUNT_DATE_SQL
    assert risk_filter == "High Risk"
    assert start_date == "2024-08-01"
    assert end_date == "2024-09-01"


# TEST 6: Unauthorized table must be rejected
def test_unauthorized_table():

    sql = """
        SELECT MAX(Salary)
        FROM HR.Employees
    """

    with pytest.raises(ValueError):
        build_approved_sql(sql)


# TEST 7: Unsupported risk category must be rejected
def test_invalid_risk_category():

    sql = """
        SELECT COUNT(*)
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        WHERE Risk_Classification = 'Unknown Risk'
    """

    with pytest.raises(ValueError):
        build_approved_risk_count_sql(sql)


# TEST 8: Invalid date range must be rejected
def test_invalid_date_range():

    sql = """
        SELECT COUNT(*)
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        WHERE Risk_Classification = 'High Risk'
          AND [Timestamp] >= '2024-09-01'
          AND [Timestamp] < '2024-08-01'
    """

    with pytest.raises(ValueError):
        build_approved_dated_risk_count_sql(sql)


# TEST 9: Multiple SQL statements must be rejected
def test_multiple_statements():

    sql = """
        SELECT MAX(Current_Temperature_C)
        FROM FDE_VIEWS.VW_ACTIVE_FLEET;

        DELETE FROM FDE_VIEWS.VW_ACTIVE_FLEET;
    """

    with pytest.raises(ValueError):
        build_approved_sql(sql)
