
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import tools_condition


# ==========================================
# 1. IMPORT OUR UNIFIED AGENT
# ==========================================

project_root = Path(__file__).resolve().parents[1]
agent_folder = project_root / "src" / "agent"

if str(agent_folder) not in sys.path:
    sys.path.insert(0, str(agent_folder))

import unified_cold_chain_agent as agent_module


# ==========================================
# 2. HELPER: CREATE MOCK TOOL CALLS
# ==========================================

def make_mock_response(tool_names):

    tool_calls = []

    for index, name in enumerate(tool_names):

        if name == "query_fleet_analytics":

            args = {
                "question": (
                    "How many High Risk observations "
                    "occurred in August 2024?"
                )
            }

        elif name == "search_sop":

            args = {
                "question": (
                    "What SOP should the operations "
                    "team follow for High Risk observations?"
                ),
                "topic": "high_risk"
            }

        else:
            raise ValueError(
                f"Unknown test tool: {name}"
            )

        tool_calls.append({
            "name": name,
            "args": args,
            "id": f"test_call_{index}",
            "type": "tool_call"
        })

    return AIMessage(
        content="",
        tool_calls=tool_calls
    )


# ==========================================
# 3. TEST MOCKED TOOL ROUTING
# ==========================================

@pytest.mark.parametrize(
    "selected_tools",
    [
        ["query_fleet_analytics"],
        ["search_sop"],
        ["query_fleet_analytics", "search_sop"]
    ]
)
def test_mocked_tool_routing(selected_tools):

    fake_response = make_mock_response(selected_tools)

    # Replace the entire LLM binding object.
    # Do NOT patch llm_with_tools.invoke directly.
    with patch.object(
        agent_module,
        "llm_with_tools"
    ) as mock_llm:

        mock_llm.invoke.return_value = fake_response

        result = agent_module.unified_agent_node({
            "messages": [
                HumanMessage(
                    content="Test question"
                )
            ]
        })

        # Verify the agent called the mocked LLM
        mock_llm.invoke.assert_called_once()

    actual_tools = [
        call["name"]
        for call in result["messages"][-1].tool_calls
    ]

    # Verify the expected tools were returned
    assert set(actual_tools) == set(selected_tools)

    # Verify LangGraph routes to the tool node
    assert tools_condition(result) == "tools"


# ==========================================
# 4. TEST ROUTING WHEN NO TOOL IS REQUIRED
# ==========================================

def test_no_tool_call_ends_routing():

    fake_response = AIMessage(
        content="Hello! How can I help you?"
    )

    with patch.object(
        agent_module,
        "llm_with_tools"
    ) as mock_llm:

        mock_llm.invoke.return_value = fake_response

        result = agent_module.unified_agent_node({
            "messages": [
                HumanMessage(content="Hello")
            ]
        })

        mock_llm.invoke.assert_called_once()

    # Agent should finish without executing tools
    assert tools_condition(result) == "__end__"


# ==========================================
# 5. OPTIONAL LIVE OPENAI TEST SETUP
# ==========================================

# Live tests remain disabled by default.
# Enable them using RUN_LIVE_ROUTING_TESTS=1.

LIVE_TESTS_ENABLED = (
    os.getenv("RUN_LIVE_ROUTING_TESTS") == "1"
)

requires_live = pytest.mark.skipif(
    not LIVE_TESTS_ENABLED,
    reason="Live routing tests are disabled"
)


def get_live_tool_calls(question, history=None):

    messages = list(history or [])

    messages.append(
        HumanMessage(content=question)
    )

    # Calls the real OpenAI model.
    # Does not execute SQL or Pinecone tools.
    result = agent_module.unified_agent_node({
        "messages": messages
    })

    return result["messages"][-1].tool_calls


# ==========================================
# 6. LIVE TEST: SQL ONLY
# ==========================================

@requires_live
def test_live_sql_only():

    calls = get_live_tool_calls(
        "How many High Risk observations "
        "occurred in August 2024?"
    )

    tool_names = {
        call["name"]
        for call in calls
    }

    assert tool_names == {
        "query_fleet_analytics"
    }


# ==========================================
# 7. LIVE TEST: PINECONE SOP ONLY
# ==========================================

@requires_live
def test_live_sop_only():

    calls = get_live_tool_calls(
        "What SOP procedures should the "
        "operations team follow during a "
        "temperature excursion?"
    )

    tool_names = {
        call["name"]
        for call in calls
    }

    assert tool_names == {
        "search_sop"
    }


# ==========================================
# 8. LIVE TEST: SQL + PINECONE
# ==========================================

@requires_live
def test_live_combined_question():

    calls = get_live_tool_calls(
        "How many High Risk observations "
        "occurred in August 2024, and what "
        "SOP procedures should the "
        "operations team follow?"
    )

    tool_names = {
        call["name"]
        for call in calls
    }

    assert tool_names == {
        "query_fleet_analytics",
        "search_sop"
    }


# ==========================================
# 9. LIVE TEST: CONVERSATIONAL FOLLOW-UP
# ==========================================

@requires_live
def test_live_follow_up_question():

    previous_messages = [
        HumanMessage(
            content=(
                "How many High Risk observations "
                "occurred in August 2023?"
            )
        ),
        AIMessage(
            content=(
                "There were 560 historical "
                "High Risk observations "
                "in August 2023."
            )
        )
    ]

    calls = get_live_tool_calls(
        question="What about Moderate Risk?",
        history=previous_messages
    )

    sql_calls = [
        call
        for call in calls
        if call["name"] == "query_fleet_analytics"
    ]

    # Agent should select the SQL analytics tool
    assert len(sql_calls) == 1

    sql_question = (
        sql_calls[0]["args"]["question"].lower()
    )

    # Agent should remember the risk change
    assert "moderate risk" in sql_question

    # Agent should preserve the previous date
    assert (
        "august 2023" in sql_question
        or "2023-08" in sql_question
    )
