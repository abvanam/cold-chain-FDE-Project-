
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# STEP 1: Define the information shared between nodes
class AgentState(TypedDict):
    question: str
    category: str
    answer: str


# STEP 2: Node 1 - Classify the question
def classify_question(state: AgentState):
    question = state["question"].lower()

    if "temperature" in question:
        category = "Temperature"
    elif "delay" in question:
        category = "Delivery Delay"
    elif "risk" in question:
        category = "Risk"
    else:
        category = "General"

    print("Node 1 executed: classify_question")

    return {"category": category}


# STEP 3: Node 2 - Prepare a simple response
def generate_answer(state: AgentState):
    category = state["category"]

    answer = (
        f"Your question belongs to the '{category}' category. "
        "In the next phase, we will retrieve the relevant SOP from Pinecone."
    )

    print("Node 2 executed: generate_answer")

    return {"answer": answer}


# STEP 4: Create the graph
builder = StateGraph(AgentState)

# Register both nodes
builder.add_node("classify_question", classify_question)
builder.add_node("generate_answer", generate_answer)

# STEP 5: Define the execution order
builder.add_edge(START, "classify_question")
builder.add_edge("classify_question", "generate_answer")
builder.add_edge("generate_answer", END)

# STEP 6: Compile the graph
graph = builder.compile()


# STEP 7: Run the graph
if __name__ == "__main__":
    initial_state = {
        "question": "Why are some deliveries delayed?",
        "category": "",
        "answer": ""
    }

    result = graph.invoke(initial_state)

    print("\nFinal Result:")
    print("Question:", result["question"])
    print("Category:", result["category"])
    print("Answer:", result["answer"])
