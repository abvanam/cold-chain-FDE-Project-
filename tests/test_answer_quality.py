
import os
import sys
import json
import re
from pathlib import Path

import pytest

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)


# ==========================================
# 1. IMPORT OUR EXISTING AGENT
# ==========================================

project_root = Path(__file__).resolve().parents[1]
agent_folder = project_root / "src" / "agent"

if str(agent_folder) not in sys.path:
    sys.path.insert(0, str(agent_folder))

import unified_cold_chain_agent as agent_module


# ==========================================
# 2. EXPECTED TEST DATA
# ==========================================

EXPECTED_COUNT = 524

SOP_EXCERPT = """
Source: cold_chain_sop.md
Section 3: High-Risk Fleet Observation

1. Review the observation timestamp, temperature,
   delay probability, congestion and route risk.

2. Identify elevated indicators.

3. Have an operations analyst verify whether
   the historical observation relates to an
   active shipment.

4. Review operational status and handling
   requirements using authorized systems.

5. Escalate when approved risk-handling
   criteria are met.

6. Document the review and actions taken.

No numeric temperature limit is specified
in this excerpt.
"""


# ==========================================
# 3. ANSWER QUALITY CHECKERS
# ==========================================

def assert_correct_count(answer):

    # Match 524 as a standalone number,
    # not as part of a larger number.
    match = re.search(
        r"(?<![\d,])524(?!\d)",
        answer
    )

    assert match is not None, (
        "Expected count 524 was not found."
    )


def assert_no_invented_temperature_limit(answer):

    # Detect numeric values such as:
    # 8°C, 2 degrees C, 5 Celsius
    temperature_pattern = (
        r"\b-?\d+(?:\.\d+)?\s*"
        r"(?:°\s*C|degrees?\s*C|Celsius)\b"
    )

    assert re.search(
        temperature_pattern,
        answer,
        flags=re.IGNORECASE
    ) is None, (
        "The answer included a numeric temperature "
        "limit not present in the test evidence."
    )


def assert_sql_failure_disclosed(answer):

    text = answer.lower()

    failure_phrases = [
        "unavailable",
        "unable to",
        "could not",
        "couldn't",
        "cannot verify",
        "can't verify",
        "not available",
        "failed"
    ]

    assert any(
        phrase in text
        for phrase in failure_phrases
    ), "The answer did not disclose the SQL failure."

    # In this test, 524 is a known value from
    # a different scenario and must not be reused.
    assert re.search(
        r"(?<![\d,])524(?!\d)",
        answer
    ) is None, (
        "The answer reused an unverified SQL count."
    )


# ==========================================
# 4. DETERMINISTIC CHECKER TESTS
# ==========================================

def test_checker_accepts_correct_count():

    answer = (
        "There were 524 historical High Risk "
        "observations in August 2024."
    )

    assert_correct_count(answer)


def test_checker_detects_incorrect_count():

    answer = (
        "There were 650 historical High Risk "
        "observations."
    )

    with pytest.raises(AssertionError):
        assert_correct_count(answer)


def test_checker_detects_invented_temperature():

    answer = (
        "The approved temperature limit is 8°C."
    )

    with pytest.raises(AssertionError):
        assert_no_invented_temperature_limit(answer)


def test_checker_detects_unverified_failure_answer():

    answer = (
        "SQL Server is unavailable, but there "
        "were 524 High Risk observations."
    )

    with pytest.raises(AssertionError):
        assert_sql_failure_disclosed(answer)


# ==========================================
# 5. BUILD SIMULATED TOOL MESSAGES
# ==========================================

def make_evidence_messages(
    question,
    include_sql=False,
    sql_success=True,
    include_sop=False
):

    tool_calls = []
    tool_messages = []

    if include_sql:

        tool_calls.append({
            "name": "query_fleet_analytics",
            "args": {
                "question": (
                    "How many High Risk observations "
                    "occurred in August 2024?"
                )
            },
            "id": "sql_test_call",
            "type": "tool_call"
        })

        if sql_success:

            sql_data = {
                "status": "success",
                "sql_result": EXPECTED_COUNT,
                "risk_filter": "High Risk",
                "date_start": "2024-08-01",
                "date_end_exclusive": "2024-09-01",
                "data_type": "historical"
            }

        else:

            sql_data = {
                "status": "error",
                "message": (
                    "Fleet analytics are temporarily "
                    "unavailable. Please try again later."
                )
            }

        tool_messages.append(
            ToolMessage(
                content=json.dumps(sql_data),
                name="query_fleet_analytics",
                tool_call_id="sql_test_call"
            )
        )

    if include_sop:

        tool_calls.append({
            "name": "search_sop",
            "args": {
                "question": (
                    "What procedures apply to "
                    "High Risk fleet observations?"
                ),
                "topic": "high_risk"
            },
            "id": "sop_test_call",
            "type": "tool_call"
        })

        tool_messages.append(
            ToolMessage(
                content=SOP_EXCERPT,
                name="search_sop",
                tool_call_id="sop_test_call"
            )
        )

    return [
        HumanMessage(content=question),

        # Pretend the agent already called these tools
        AIMessage(
            content="",
            tool_calls=tool_calls
        ),

        # Provide simulated tool responses
        *tool_messages
    ]


# ==========================================
# 6. CALL THE REAL LLM FOR FINAL ANSWER
# ==========================================

def generate_test_answer(messages):

    # Call only the unified agent decision node.
    # We do not execute SQL Server or Pinecone tools.
    result = agent_module.unified_agent_node({
        "messages": messages
    })

    response = result["messages"][-1]

    # The model already has tool results.
    # It should now produce a final answer.
    assert not response.tool_calls, (
        "The agent requested another tool call "
        "instead of producing a final response."
    )

    assert isinstance(response.content, str)
    assert response.content.strip()

    return response.content


# ==========================================
# 7. OPTIONAL LIVE TEST CONFIGURATION
# ==========================================

LIVE_TESTS_ENABLED = (
    os.getenv("RUN_LIVE_ANSWER_TESTS") == "1"
)

requires_live = pytest.mark.skipif(
    not LIVE_TESTS_ENABLED,
    reason="Live answer-quality tests are disabled"
)


# ==========================================
# 8. LIVE TEST: CORRECT SQL COUNT
# ==========================================

@requires_live
def test_live_sql_answer_accuracy():

    messages = make_evidence_messages(
        question=(
            "How many High Risk observations "
            "occurred in August 2024?"
        ),
        include_sql=True
    )

    answer = generate_test_answer(messages)

    print("\nSQL ANSWER:\n", answer)

    assert_correct_count(answer)

    # Avoid describing historical records as
    # verified live shipments.
    assert "active shipments" not in answer.lower()


# ==========================================
# 9. LIVE TEST: COMBINED SQL + SOP
# ==========================================

@requires_live
def test_live_combined_answer():

    messages = make_evidence_messages(
        question=(
            "How many High Risk observations "
            "occurred in August 2024, and what "
            "SOP procedures should we follow?"
        ),
        include_sql=True,
        include_sop=True
    )

    answer = generate_test_answer(messages)

    print("\nCOMBINED ANSWER:\n", answer)

    assert_correct_count(answer)

    assert (
        "sop" in answer.lower()
        or "procedure" in answer.lower()
    )

    assert (
        "review" in answer.lower()
        or "escalat" in answer.lower()
    )

    assert_no_invented_temperature_limit(answer)


# ==========================================
# 10. LIVE TEST: SOP ONLY
# ==========================================

@requires_live
def test_live_sop_no_invented_threshold():

    messages = make_evidence_messages(
        question=(
            "What SOP procedures should the "
            "operations team follow for "
            "High Risk observations?"
        ),
        include_sop=True
    )

    answer = generate_test_answer(messages)

    print("\nSOP ANSWER:\n", answer)

    assert (
        "sop" in answer.lower()
        or "procedure" in answer.lower()
    )

    assert_no_invented_temperature_limit(answer)


# ==========================================
# 11. LIVE TEST: SQL CONNECTION FAILURE
# ==========================================

@requires_live
def test_live_sql_failure_no_fabrication():

    messages = make_evidence_messages(
        question=(
            "How many High Risk observations "
            "occurred in August 2024?"
        ),
        include_sql=True,
        sql_success=False
    )

    answer = generate_test_answer(messages)

    print("\nFAILURE ANSWER:\n", answer)

    assert_sql_failure_disclosed(answer)
