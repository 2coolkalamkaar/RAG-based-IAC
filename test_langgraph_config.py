from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig

class State(TypedDict):
    val: int

def my_node(state: State, config: RunnableConfig):
    print("CONFIG:", config)
    return {"val": state["val"] + 1}

workflow = StateGraph(State)
workflow.add_node("my_node", my_node)
workflow.add_edge(START, "my_node")
workflow.add_edge("my_node", END)
app = workflow.compile()

app.invoke({"val": 1}, config={"callbacks": []})
