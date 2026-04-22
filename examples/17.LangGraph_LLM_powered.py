"""
Exercise 17: LangGraph Orchestration Architecture with LLMs
This example demonstrates an Orchestration Architecture using a state machine
(directed graph) built with LangGraph, where individual nodes are powered by LLMs.

How it works:
- Centralized State: All inputs, LLM outputs, and errors are stored in a single `SalesReportState`.
- Decoupled LLM Workers: Each node acts as a specialized agent (collector, analyst, etc.)
  executing specific prompts. They mutate the state but have no knowledge of the overall workflow.
- Central Orchestrator: The graph configuration dictates the execution flow. It uses
  conditional edges to inspect the state (e.g., checking for exceptions or LLM parsing errors)
  and dynamically routes to the next step or a centralized error handler.
"""

import utils
from dotenv import load_dotenv, find_dotenv
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv(usecwd=True))

MODEL_NAME = "openai/gpt-4o-mini"

# =========================
# 1. LLM INIT
# =========================
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.2,
    base_url="https://openrouter.ai/api/v1",
    api_key=utils.get_api_key("OPENROUTER_API_KEY"),
)


# =========================
# 2. STATE
# =========================
class SalesReportState(TypedDict):
    request: str
    raw_data: Optional[Dict[str, Any]]
    processed_data: Optional[str]  # LLM-generated text
    chart_config: Optional[str]  # LLM-generated JSON text
    report: Optional[str]
    errors: List[str]


# =========================
# 3. AGENTS (LLM-POWERED)
# =========================


def data_collector_agent(state: SalesReportState) -> SalesReportState:
    prompt = ChatPromptTemplate.from_template("""
You are a data collection agent.

Generate realistic e-commerce sales data for:
Request: {request}

Return ONLY valid JSON like:
{{
  "orders": [
    {{"date": "YYYY-MM-DD", "product": "...", "category": "...", "revenue": number}}
  ]
}}
""")

    chain = prompt | llm
    response = chain.invoke({"request": state["request"]})

    try:
        import json

        state["raw_data"] = json.loads(response.content)
    except Exception as e:
        state["errors"].append(str(e))

    return state


def data_processor_agent(state: SalesReportState) -> SalesReportState:
    prompt = ChatPromptTemplate.from_template("""
You are a data analyst.

Given this raw data:
{raw_data}

Compute:
- total revenue
- revenue by category
- monthly trends

Return structured analysis in JSON.
""")

    try:
        chain = prompt | llm
        response = chain.invoke({"raw_data": state["raw_data"]})
        state["processed_data"] = response.content
    except Exception as e:
        state["errors"].append(str(e))
    return state


def chart_generator_agent(state: SalesReportState) -> SalesReportState:
    prompt = ChatPromptTemplate.from_template("""
You are a data visualization expert.

Based on this analysis:
{processed_data}

Generate a chart configuration in JSON:
- type (bar/line)
- title
- x-axis labels
- y-axis values
""")

    try:
        chain = prompt | llm
        response = chain.invoke({"processed_data": state["processed_data"]})
        state["chart_config"] = response.content
    except Exception as e:
        state["errors"].append(str(e))
    return state


def report_generator_agent(state: SalesReportState) -> SalesReportState:
    prompt = ChatPromptTemplate.from_template("""
You are a senior business analyst.

Using:
Raw Data: {raw_data}
Analysis: {processed_data}
Chart: {chart_config}

Write a professional executive sales report with:
- summary
- insights
- recommendations
""")

    try:
        chain = prompt | llm
        response = chain.invoke(
            {
                "raw_data": state["raw_data"],
                "processed_data": state["processed_data"],
                "chart_config": state["chart_config"],
            }
        )
        state["report"] = response.content
    except Exception as e:
        state["errors"].append(str(e))
    return state


def error_handler_agent(state: SalesReportState) -> SalesReportState:
    state["report"] = f"Workflow failed:\n{state['errors']}"
    return state


# =========================
# 5. BUILD GRAPH
# =========================
def build_graph():
    graph = StateGraph(SalesReportState)

    graph.add_node("data_collector", data_collector_agent)
    graph.add_node("data_processor", data_processor_agent)
    graph.add_node("chart_generator", chart_generator_agent)
    graph.add_node("report_generator", report_generator_agent)
    graph.add_node("error_handler", error_handler_agent)

    graph.set_entry_point("data_collector")

    graph.add_conditional_edges(
        "data_collector", lambda s: "error_handler" if s["errors"] else "data_processor"
    )
    graph.add_conditional_edges(
        "data_processor",
        lambda s: "error_handler" if s["errors"] else "chart_generator",
    )
    graph.add_conditional_edges(
        "chart_generator",
        lambda s: "error_handler" if s["errors"] else "report_generator",
    )
    graph.add_conditional_edges(
        "report_generator", lambda s: "error_handler" if s["errors"] else END
    )
    graph.add_edge("error_handler", END)

    return graph.compile()


# =========================
# 6. RUN
# =========================
def run():
    app = build_graph()

    state: SalesReportState = {
        "request": "Q1-Q2 2025 E-commerce Sales Analysis",
        "raw_data": None,
        "processed_data": None,
        "chart_config": None,
        "report": None,
        "errors": [],
    }

    final = app.invoke(state)

    print("\n===== FINAL REPORT =====\n")
    print(final["report"])

    print("\n===== CHART CONFIG =====\n")
    print(final["chart_config"])


if __name__ == "__main__":
    run()
