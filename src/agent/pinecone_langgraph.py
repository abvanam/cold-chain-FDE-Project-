
import os
from pathlib import Path
from typing import TypedDict
import json
import pyodbc

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore


# Load credentials from the project root
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

# Connect to the Pinecone knowledge base
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Connect to Pinecone to retrieve SOP documents
vector_store = PineconeVectorStore(
    index_name="fde-sop-index-openai",
    embedding=embeddings
)

# Initialize the OpenAI chat model to generate answers
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)



# Information shared between LangGraph nodes
class AgentState(TypedDict):
    question: str
    category: str
    sop_content: str
    sql_results: str
    answer: str


# Node 1: Identify the relevant SOP topic
def classify_question(state: AgentState):
    question = state["question"].lower()

    if "temperature" in question or "refrigeration" in question:
        category = "temperature_excursion"
    elif "delay" in question or "late" in question:
        category = "delivery_delay"
    elif "route" in question or "congestion" in question:
        category = "route_risk"
    elif "high risk" in question:
        category = "high_risk"
    else:
        category = "general"

    print(f"Node 1 — Classified topic: {category}")
    return {"category": category}


# Decide which node should execute next
def route_question(state: AgentState):

    category = state["category"]

    if category == "high_risk":
        print("Router — SQL Server and Pinecone required")
        return "query_sql_server"

    print("Router — Pinecone only")
    return "retrieve_sop"



# Node 2: Retrieve historical observations from SQL Server
def query_sql_server(state: AgentState):

    print("Node 2 — Querying SQL Server")

    # Only run this predefined query for High Risk questions
    if state["category"] != "high_risk":
        print("SQL query skipped: not a High Risk question")
        return {"sql_results": "No SQL query requested."}

    connection_string = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={os.getenv('SQL_RO_HOST')},"
        f"{os.getenv('SQL_RO_PORT')};"
        f"DATABASE={os.getenv('SQL_RO_DATABASE')};"
        f"UID={os.getenv('SQL_RO_USER')};"
        f"PWD={os.getenv('SQL_RO_PASSWORD')};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=10;"
    )

    # Fixed read-only query against our approved SQL view
    sql_query = """
        SELECT TOP (5)
            [Timestamp],
            [Current_Temperature_C],
            [Risk_Classification],
            [Delay_Probability]
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        WHERE [Risk_Classification] = 'High Risk'
        ORDER BY [Timestamp] DESC
    """

    with pyodbc.connect(connection_string, timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute(sql_query)

        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()

        records = [
            dict(zip(columns, row))
            for row in rows
        ]

    print(f"Node 2 — Retrieved {len(records)} SQL records")

    return {
        "sql_results": json.dumps(records, default=str)
    }


# Node 2: Retrieve matching SOP chunks
def retrieve_sop(state: AgentState):
    topic = state["category"]

    search_filter = (
        {"topic": {"$eq": topic}}
        if topic != "general"
        else None
    )

    documents = vector_store.similarity_search(
        state["question"],
        k=3,
        filter=search_filter
    )

    sop_content = "\n\n".join(
        doc.page_content for doc in documents
    )

    print(f"Node 3 — Retrieved {len(documents)} SOP chunks")

    return {"sop_content": sop_content}


# Node 3: Generate an answer using the retrieved SOP
def generate_answer(state: AgentState):

    print("Node 4 — Generating answer with OpenAI")

    if not state["sop_content"].strip():
        return {
            "answer": "No relevant SOP was found. Please consult the operations team."
        }

    system_prompt = """
    You are a Cold Chain Logistics SOP assistant.

    Answer the user's question using only the supplied SOP context.

    Rules:
    - Do not invent temperature thresholds.
    - Do not claim historical data represents live shipment status.
    - Do not claim that operational actions have been performed.
    - Explain when approved product-specific limits are required.
    - Identify the SOP section used.
    - If the SOP does not contain the answer, say so.
    """


    user_prompt = f"""
    Business question:
    {state["question"]}

    HISTORICAL SQL OBSERVATIONS:
    {state["sql_results"]}

    RELEVANT SOP:
    {state["sop_content"]}

    Answer requirements:

    1. Show the historical SQL observations in a Markdown table
       when records are available.

    2. Include their timestamps, temperatures, risk
       classifications, and delay probabilities.

    3. Explain the relevant operational SOP procedures.

    4. Clearly identify the observations as historical.

    5. Do not invent risk classification rules,
       temperature thresholds, or shipment identities.

    6. Distinguish recorded facts from SOP recommendations.

    7. Provide a concise summary of findings.
    """




    response = llm.invoke([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    return {"answer": response.content}




# Build LangGraph
builder = StateGraph(AgentState)

# Register nodes
builder.add_node("classify_question", classify_question)
builder.add_node("query_sql_server", query_sql_server)
builder.add_node("retrieve_sop", retrieve_sop)
builder.add_node("generate_answer", generate_answer)

# Start with classification
builder.add_edge(START, "classify_question")

# Conditional routing
builder.add_conditional_edges(
    "classify_question",
    route_question,
    {
        "query_sql_server": "query_sql_server",
        "retrieve_sop": "retrieve_sop"
    }
)

# SQL results flow to Pinecone
builder.add_edge("query_sql_server", "retrieve_sop")

# Pinecone results flow to OpenAI
builder.add_edge("retrieve_sop", "generate_answer")

# Finish the workflow
builder.add_edge("generate_answer", END)

graph = builder.compile()



# Execute
if __name__ == "__main__":
    result = graph.invoke({
        "question": "What is the average recorded temperature for High Risk fleet observations?",
        "category": "",
        "sql_results": "",
        "sop_content": "",
        "answer": ""
    })

    print("\n========== FINAL AI ANSWER ==========")
    print(result["answer"])