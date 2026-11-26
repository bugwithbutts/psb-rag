"""Сборка production-графа агентского сервиса."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.config import Settings
from agents.nodes import (
    make_clarify_node,
    make_finalize_node,
    make_generate_node,
    make_grade_node,
    make_parse_dialog_node,
    make_refuse_node,
    make_retrieve_node,
    make_rewrite_node,
    make_understand_node,
    make_verify_node,
    route_after_grade,
    route_after_understand,
    route_after_verify,
)
from agents.state import AgentState
from rag.rag_pipeline import RAG

__all__ = ["build_graph"]


def build_graph(
    rag_client: RAG,
    settings: Settings,
) -> CompiledStateGraph:
    """Собрать production-граф с переформулировкой запроса и controlled finalize.

    Текущая схема:

        START
          -> parse_dialog
          -> understand
             -> refuse   -> END
             -> clarify  -> END
             -> retrieve
                -> grade
                   -> rewrite  -> retrieve
                   -> generate -> verify -> finalize -> END
                   -> clarify  -> END
                   -> finalize -> END
    """
    graph = StateGraph(AgentState)
    graph.add_node("parse_dialog", make_parse_dialog_node())
    graph.add_node("understand", make_understand_node(settings))
    graph.add_node("retrieve", make_retrieve_node(rag_client))
    graph.add_node("grade", make_grade_node(settings))
    graph.add_node("rewrite", make_rewrite_node(settings))
    graph.add_node("generate", make_generate_node(settings))
    graph.add_node("verify", make_verify_node(settings))
    graph.add_node("clarify", make_clarify_node())
    graph.add_node("finalize", make_finalize_node())
    graph.add_node("refuse", make_refuse_node())

    graph.add_edge(START, "parse_dialog")
    graph.add_edge("parse_dialog", "understand")
    graph.add_conditional_edges(
        "understand",
        route_after_understand,
        {
            "retrieve": "retrieve",
            "refuse": "refuse",
            "clarify": "clarify",
        },
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        lambda state: route_after_grade(state, settings),
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "finalize": "finalize",
            "clarify": "clarify",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda state: route_after_verify(state, settings),
        {
            "generate": "generate",
            "finalize": "finalize",
        },
    )
    graph.add_edge("clarify", END)
    graph.add_edge("finalize", END)
    graph.add_edge("refuse", END)

    return graph.compile()
