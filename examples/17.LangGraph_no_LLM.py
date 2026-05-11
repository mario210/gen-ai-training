"""
Exercise 16: LangGraph Orchestration Architecture without LLM
This example demonstrates a pure Orchestration Architecture using a state
machine (directed graph) built with LangGraph and plain Python functions.

How it works:
- Centralized State: All data is stored and passed via a single `SalesReportState`.
- Decoupled Workers: Each node performs a specific task (collect, process, etc.)
  and mutates the state. Workers have no knowledge of the overall workflow.
- Central Orchestrator: The graph configuration dictates the execution flow. It uses
  conditional edges to inspect the state (like checking for errors) and routes to
  the next node accordingly, keeping workflow logic out of the worker functions.
"""

from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END


# =========================
# 1. STATE DEFINITION
# =========================
class SalesReportState(TypedDict):
    request: str
    raw_data: Optional[Dict[str, Any]]
    processed_data: Optional[Dict[str, Any]]
    chart_config: Optional[Dict[str, Any]]
    report: Optional[str]
    errors: List[str]


# =========================
# 2. AGENT NODES
# =========================


def data_collector_agent(state: SalesReportState) -> SalesReportState:
    try:
        # Simulated dataset
        state["raw_data"] = {
            "orders": [
                {
                    "date": "2025-01-10",
                    "product": "Laptop",
                    "category": "Electronics",
                    "revenue": 1200,
                },
                {
                    "date": "2025-02-15",
                    "product": "Phone",
                    "category": "Electronics",
                    "revenue": 800,
                },
                {
                    "date": "2025-03-01",
                    "product": "Shoes",
                    "category": "Fashion",
                    "revenue": 150,
                },
            ]
        }
    except Exception as e:
        state["errors"].append(str(e))
    return state


def data_processor_agent(state: SalesReportState) -> SalesReportState:
    try:
        orders = state.get("raw_data", {}).get("orders", [])

        total_revenue = sum(o["revenue"] for o in orders)

        by_category = {}
        monthly_revenue = {}

        for o in orders:
            # category aggregation
            cat = o["category"]
            by_category[cat] = by_category.get(cat, 0) + o["revenue"]

            # monthly aggregation (simplified)
            month = o["date"][:7]
            monthly_revenue[month] = monthly_revenue.get(month, 0) + o["revenue"]

        state["processed_data"] = {
            "total_revenue": total_revenue,
            "by_category": by_category,
            "monthly_revenue": monthly_revenue,
        }

    except Exception as e:
        state["errors"].append(str(e))

    return state


def chart_generator_agent(state: SalesReportState) -> SalesReportState:
    try:
        monthly = state["processed_data"]["monthly_revenue"]

        state["chart_config"] = {
            "type": "bar",
            "title": "Monthly Revenue",
            "x": list(monthly.keys()),
            "y": list(monthly.values()),
        }

    except Exception as e:
        state["errors"].append(str(e))

    return state


def report_generator_agent(state: SalesReportState) -> SalesReportState:
    try:
        data = state["processed_data"]

        report = f"""
Sales Analysis Report

Total Revenue: ${data['total_revenue']}

Revenue by Category:
"""

        for k, v in data["by_category"].items():
            report += f"- {k}: ${v}\n"

        report += "\nMonthly Trend:\n"

        for k, v in data["monthly_revenue"].items():
            report += f"- {k}: ${v}\n"

        report += """
Insight:
Electronics dominate revenue, while Fashion remains underdeveloped.
Recommendation: diversify marketing toward Fashion category.
"""

        state["report"] = report

    except Exception as e:
        state["errors"].append(str(e))

    return state


def error_handler_agent(state: SalesReportState) -> SalesReportState:
    state["report"] = "Workflow failed due to errors:\n" + "\n".join(state["errors"])
    return state


# =========================
# 4. BUILD GRAPH
# =========================
def create_workflow():
    workflow = StateGraph(SalesReportState)

    workflow.add_node("data_collector", data_collector_agent)
    workflow.add_node("data_processor", data_processor_agent)
    workflow.add_node("chart_generator", chart_generator_agent)
    workflow.add_node("report_generator", report_generator_agent)
    workflow.add_node("error_handler", error_handler_agent)

    workflow.set_entry_point("data_collector")

    workflow.add_conditional_edges(
        "data_collector", lambda s: "error_handler" if s["errors"] else "data_processor"
    )
    workflow.add_conditional_edges(
        "data_processor",
        lambda s: "error_handler" if s["errors"] else "chart_generator",
    )
    workflow.add_conditional_edges(
        "chart_generator",
        lambda s: "error_handler" if s["errors"] else "report_generator",
    )
    workflow.add_conditional_edges(
        "report_generator", lambda s: "error_handler" if s["errors"] else END
    )
    workflow.add_edge("error_handler", END)

    return workflow.compile()


# =========================
# 5. RUN WORKFLOW
# =========================
def run():
    app = create_workflow()

    initial_state: SalesReportState = {
        "request": "Q1-Q2 2025 Sales Analysis",
        "raw_data": None,
        "processed_data": None,
        "chart_config": None,
        "report": None,
        "errors": [],
    }

    final_state = app.invoke(initial_state)

    print("\n================ FINAL REPORT ================\n")
    print(final_state["report"])

    print("\n================ CHART CONFIG ================\n")
    print(final_state["chart_config"])


if __name__ == "__main__":
    run()
