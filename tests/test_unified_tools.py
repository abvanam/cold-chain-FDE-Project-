
import sys
import json
from pathlib import Path
from unittest.mock import patch

import pytest

# Locate our existing agent files
project_root = Path(__file__).resolve().parents[1]
agent_folder = project_root / "src" / "agent"

if str(agent_folder) not in sys.path:
    sys.path.insert(0, str(agent_folder))

from unified_cold_chain_agent import (
    query_fleet_analytics,
    tools
)


# ==========================================
# TEST 1: SQL TOOL HANDLES SUCCESS
# ==========================================

def test_sql_tool_success():

    mock_result = {
        "safe_sql": (
            "SELECT COUNT_BIG(*) AS Observation_Count "
            "FROM FDE_VIEWS.VW_ACTIVE_FLEET"
        ),
        "risk_filter": "High Risk",
        "date_start": "2024-08-01",
        "date_end_exclusive": "2024-09-01",
        "sql_result": 524,
        "answer": "524 observations",
        "error": ""
    }

    with patch(
        "unified_cold_chain_agent.text_to_sql_graph.invoke",
        return_value=mock_result
    ) as mock_invoke:

        response = query_fleet_analytics.invoke({
            "question": (
                "How many High Risk observations "
                "occurred in August 2024?"
            )
        })

    data = json.loads(response)

    assert data["status"] == "success"
    assert data["sql_result"] == 524
    assert data["risk_filter"] == "High Risk"
    assert data["date_start"] == "2024-08-01"

    # Confirm the nested graph was called
    mock_invoke.assert_called_once()


# ==========================================
# TEST 2: SQL VALIDATION REJECTION
# ==========================================

def test_sql_tool_rejected():

    mock_result = {
        "error": "Unsupported SQL query"
    }

    with patch(
        "unified_cold_chain_agent.text_to_sql_graph.invoke",
        return_value=mock_result
    ):

        response = query_fleet_analytics.invoke({
            "question": "Show employee salaries"
        })

    data = json.loads(response)

    assert data["status"] == "rejected"
    assert "Unsupported" in data["message"]


# ==========================================
# TEST 3: SQL CONNECTION FAILURE
# ==========================================


def test_sql_tool_connection_failure():

    # Simulate SQL Server being unavailable
    with patch(
        "unified_cold_chain_agent.text_to_sql_graph.invoke",
        side_effect=ConnectionError(
            "SQL Server unavailable: internal details"
        )
    ):

        response = query_fleet_analytics.invoke({
            "question": (
                "How many High Risk observations "
                "are there?"
            )
        })

    # Convert the returned JSON into Python data
    data = json.loads(response)

    # The tool should return an error status
    assert data["status"] == "error"

    # The user should receive a helpful message
    assert "temporarily unavailable" in data["message"]

    # Internal technical details should not be exposed
    assert "internal details" not in response

# ==========================================
# TEST 4: BOTH TOOLS ARE REGISTERED
# ==========================================

def test_unified_tool_registration():

    tool_names = [tool.name for tool in tools]

    assert "query_fleet_analytics" in tool_names
    assert "search_sop" in tool_names
