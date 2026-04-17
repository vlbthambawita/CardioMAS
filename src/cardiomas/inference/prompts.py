from __future__ import annotations

import json

from cardiomas.inference.base import ChatMessage
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.tools import ToolSpec


def planner_messages(
    query: str,
    tools: list[ToolSpec],
    dataset_path: str,
    urls: list[str],
    expression: str,
) -> list[ChatMessage]:
    tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools) or "- none"
    context = {
        "query": query,
        "default_dataset_path": dataset_path,
        "urls_in_query": urls,
        "expression_hint": expression,
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You are the CardioMAS planner. Select only from the provided tools. "
                "Return strict JSON with keys: strategy, steps, notes. "
                "Each step must contain tool_name, reason, and args. "
                "Do not invent tools or filesystem paths."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Available tools:\n"
                f"{tool_lines}\n\n"
                "Planning context:\n"
                f"{json.dumps(context, indent=2)}\n\n"
                "Rules:\n"
                "- STRICT: When dataset_mode is script_only and a dataset path is configured, "
                "select generate_python_artifact as the ONLY tool. "
                "Do NOT add retrieve_corpus, inspect_dataset, fetch_webpage, or calculate for dataset queries.\n"
                "- When dataset_mode is agentic, use retrieve_corpus for grounded answers when available.\n"
                "- Prefer generate_python_artifact for local dataset reading, file inspection, metadata extraction, and statistical analysis.\n"
                "- Use inspect_dataset only when generate_python_artifact is not available.\n"
                "- Use generate_shell_artifact only when the user asks for a shell or bash script.\n"
                "- Use calculate only for explicit arithmetic.\n"
                "- Use fetch_webpage only for direct URLs already present in the query.\n"
                "- For generate_python_artifact args use only: task, dataset_path, target_path, artifact_name.\n"
                "- Prefer the default_dataset_path when a dataset tool is needed.\n"
                "- Return compact JSON only."
            ),
        ),
    ]


def responder_messages(
    query: str,
    evidence: list[EvidenceChunk],
    aggregate: dict,
    warnings: list[str],
) -> list[ChatMessage]:
    evidence_lines = []
    for index, chunk in enumerate(evidence, start=1):
        evidence_lines.append(
            json.dumps(
                {
                    "id": index,
                    "chunk_id": chunk.chunk_id,
                    "source_label": chunk.source_label,
                    "locator": chunk.metadata.get("chunk_label") or chunk.title or chunk.uri,
                    "content": _trim(chunk.content, 500),
                },
                ensure_ascii=True,
            )
        )

    tool_context = {
        "dataset_inspection": aggregate.get("dataset_inspection"),
        "calculations": aggregate.get("calculations", []),
        "generated_python_artifacts": aggregate.get("generated_python_artifacts", []),
        "generated_shell_artifacts": aggregate.get("generated_shell_artifacts", []),
        "web_pages": aggregate.get("web_pages", []),
        "warnings": warnings,
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You are the grounded CardioMAS responder. Answer only with support from the provided evidence "
                "and tool outputs. If the evidence is insufficient, say so plainly. "
                "Return strict JSON with keys: answer, citations, warnings. "
                "Citations must be a list of integer evidence ids."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Question:\n{query}\n\n"
                "Evidence items:\n"
                f"{chr(10).join(evidence_lines) if evidence_lines else '(none)'}\n\n"
                "Tool context:\n"
                f"{json.dumps(tool_context, indent=2, default=str)}\n\n"
                "Write a concise grounded answer. Return JSON only."
            ),
        ),
    ]


def orchestrator_messages(
    query: str,
    tools: list[ToolSpec],
    observations: list[dict],
) -> list[ChatMessage]:
    tool_lines = "\n".join(
        f"- {t.name}: {t.description}" for t in tools
    ) or "- none"
    obs_text = ""
    if observations:
        parts = []
        for i, obs in enumerate(observations, 1):
            tool = obs.get("tool", "?")
            observation = obs.get("observation", obs.get("error", ""))
            parts.append(f"  Step {i}: called {tool!r} → {_trim(observation, 200)}")
        obs_text = "\nPrevious steps:\n" + "\n".join(parts) + "\n"
    return [
        ChatMessage(
            role="system",
            content=(
                "You are CardioMAS, a medical dataset analysis agent. "
                "Decide the next action to take given the query and what you have observed so far. "
                "Return strict JSON with keys: thought, action, args. "
                "'thought' is your reasoning. 'action' is the tool name or 'answer' when done. "
                "'args' is a dict of arguments for the tool (empty dict when action is 'answer'). "
                "Do not repeat the same tool call with identical args. "
                "Say action='answer' when you have enough information."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Query: {query}\n"
                f"{obs_text}\n"
                "Available tools:\n"
                f"{tool_lines}\n\n"
                "What is your next action? Return JSON only."
            ),
        ),
    ]


def router_messages(query: str, tools: list[ToolSpec]) -> list[ChatMessage]:
    tool_names = ", ".join(t.name for t in tools)
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a query router for a medical dataset analysis system. "
                "Classify the user query into one of these routes: "
                "'code' (needs computation or data analysis), "
                "'retrieval' (needs document lookup), "
                "'web' (needs a live URL fetch), "
                "'orchestrate' (complex, needs multiple steps). "
                "Return strict JSON with keys: route, reason."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Query: {query}\n"
                f"Available tools: {tool_names}\n\n"
                "Classify this query. Return JSON only."
            ),
        ),
    ]


def decomposer_messages(query: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You decompose complex user queries into atomic sub-queries for a medical dataset analysis system. "
                "Each sub-query has a type: 'factual' (needs document lookup), "
                "'computational' (needs data analysis or code), or 'exploratory' (open-ended). "
                "Return strict JSON with key 'sub_queries': a list of objects each with 'text' and 'query_type'. "
                "If the query is already simple, return a single sub-query with the original text. "
                "Cap at 4 sub-queries."
            ),
        ),
        ChatMessage(
            role="user",
            content=f"Query: {query}\n\nDecompose into atomic sub-queries. Return JSON only.",
        ),
    ]


def retrieval_grader_messages(query: str, chunks_text: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You grade the relevance of retrieved evidence for a user query. "
                "Given the query and retrieved chunk summaries, decide if the evidence is: "
                "'sufficient' (clearly answers the query), "
                "'partial' (partially relevant, proceed but note gaps), "
                "or 'insufficient' (irrelevant, need different retrieval). "
                "Return strict JSON with keys: verdict, relevant_count (int), reason."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Query: {query}\n\n"
                f"Retrieved chunks:\n{chunks_text}\n\n"
                "Grade relevance. Return JSON only."
            ),
        ),
    ]


def answer_grader_messages(query: str, answer: str, evidence_text: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You assess the quality of an AI-generated answer to a medical dataset query. "
                "Check whether the answer is grounded in the provided evidence. "
                "Return strict JSON with keys: verdict ('grounded', 'hallucinated', or 'incomplete'), reason."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Query: {query}\n\n"
                f"Answer: {answer}\n\n"
                f"Evidence:\n{_trim(evidence_text, 800)}\n\n"
                "Is the answer grounded? Return JSON only."
            ),
        ),
    ]


def prompt_preview(messages: list[ChatMessage], limit: int = 400) -> str:
    text = "\n".join(f"{message.role.upper()}: {message.content}" for message in messages)
    return _trim(text, limit)


def _trim(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    trimmed = compact[:limit].rsplit(" ", 1)[0]
    return f"{trimmed}..."
