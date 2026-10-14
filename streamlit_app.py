
import sys
import traceback
from pathlib import Path
import uuid
import json
import re

import streamlit as st


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Cold Chain Logistics AI",
    page_icon="🚚",
    layout="centered"
)


# ==========================================
# 2. LOCATE OUR LANGGRAPH AGENT
# ==========================================

project_root = Path(__file__).resolve().parent

agent_folder = project_root / "src" / "agent"

# Make our existing Python agent files importable
if str(agent_folder) not in sys.path:
    sys.path.insert(0, str(agent_folder))


# ==========================================
# 3. LOAD THE UNIFIED LANGGRAPH AGENT
# ==========================================

@st.cache_resource(show_spinner="Loading AI agent...")
def load_agent():

    from unified_cold_chain_agent import graph

    return graph


# ==========================================
# 4. PAGE HEADER
# ==========================================

st.title("🚚 Cold Chain Logistics AI Assistant")

st.write(
    "Ask questions about historical fleet observations "
    "and approved operational SOP procedures."
)

st.caption(
    "Powered by LangGraph, OpenAI, SQL Server and Pinecone"
)

st.info(
    "The logistics dataset contains historical observations, "
    "not live shipment information."
)


# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("About this application")

    st.write("Available capabilities:")

    st.write("📊 Historical fleet analytics")
    st.write("📚 SOP knowledge retrieval")
    st.write("🤖 Intelligent AI tool selection")

    st.divider()

    if st.button("Clear chat history"):

    # Clear the messages shown in Streamlit
        st.session_state.chat_history = []

    # Start a new LangGraph conversation
        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()


# ==========================================
# 6. INITIALIZE CHAT HISTORY
# ==========================================

# Store chat messages displayed in the browser
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Give this conversation a unique LangGraph thread ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ==========================================
# 7. DISPLAY PREVIOUS MESSAGES
# ==========================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_sources(message.get("sources", []))


# Extract sources used in the most recent agent response
def extract_sources(messages):

    # Keep only messages from the latest user turn
    current_turn = []

    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            break
        current_turn.append(message)

    current_turn.reverse()

    sources = []

    for message in current_turn:

        if getattr(message, "type", "") != "tool":
            continue

        tool_name = getattr(message, "name", "")
        content = str(message.content)

        # SQL Server tool
        if tool_name == "query_fleet_analytics":

            try:
                data = json.loads(content)

                sources.append({
                    "type": "sql",
                    "data": data
                })

            except json.JSONDecodeError:
                sources.append({
                    "type": "sql",
                    "data": {
                        "status": "error",
                        "message": "Unable to read SQL evidence."
                    }
                })

        # Pinecone SOP tool
        elif tool_name == "search_sop":

            documents = re.findall(
                r"Source:\s*([^\n]+)",
                content
            )

            sections = re.findall(
                r"(?m)^#{1,6}\s+(.+)$",
                content
            )

            sources.append({
                "type": "sop",
                "documents": list(dict.fromkeys(documents)),
                "sections": list(dict.fromkeys(sections))
            })

    return sources


# Display the evidence beneath an AI answer
def show_sources(sources):

    if not sources:
        return

    with st.expander("🔎 View Sources & SQL Details"):

        for source in sources:

            if source["type"] == "sql":

                data = source["data"]

                st.subheader("SQL Server")

                if data.get("status") == "success":

                    st.write("**Approved SQL query:**")

                    st.code(
                        data.get("approved_sql", ""),
                        language="sql"
                    )

                    st.write("**Applied parameters:**")

                    st.json({
                        "risk_classification": data.get("risk_filter"),
                        "date_start": data.get("date_start"),
                        "date_end_exclusive": data.get(
                            "date_end_exclusive"
                        )
                    })

                    st.write("**Verified SQL result:**")
                    st.write(data.get("sql_result"))

                else:
                    st.warning(
                        data.get(
                            "message",
                            "SQL query was not completed."
                        )
                    )

            elif source["type"] == "sop":

                st.subheader("Pinecone SOP")

                st.write(
                    "**Source documents:**",
                    source["documents"] or ["Not identified"]
                )

                st.write(
                    "**Retrieved sections:**",
                    source["sections"] or ["Not identified"]
                )


# ==========================================
# 8. ACCEPT USER INPUT
# ==========================================

user_question = st.chat_input(
    "Ask about fleet data or SOP procedures..."
)

if user_question:

    # Save the user's question
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })

    # Display the user's question
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate an AI response
    with st.chat_message("assistant"):

        with st.spinner("Analyzing your question..."):

            try:

                agent = load_agent()

                # Call our existing unified LangGraph

                result = agent.invoke(
                    {
                        # Send only the NEW user message
                        "messages": [
                            ("user", user_question)
                        ]
                    },
                    config={
                        # Identify the current conversation
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        },

                        "recursion_limit": 12
                    }
                )


                # Read the final AI response
                answer = result["messages"][-1].content

                if not isinstance(answer, str) or not answer:
                    answer = (
                        "The agent did not return a text answer."
                    )

                # Extract SQL and Pinecone evidence
                sources = extract_sources(result["messages"])

                # Display the response
                st.markdown(answer)
                show_sources(sources)

                # Save the response to UI history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception:

                # Print technical details only to the
                # local VS Code terminal
                traceback.print_exc()

                st.error(
                    "Unable to process the question. "
                    "Check the VS Code terminal for details "
                    "and verify the required connections."
                )
