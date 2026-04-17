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
    # Separate script-output chunks (highest priority) from corpus/tool chunks.
    script_chunks = [c for c in evidence if c.source_type == "script_output"]
    other_chunks = [c for c in evidence if c.source_type != "script_output"]

    # Build numbered evidence list for citation purposes (non-script first, then script).
    ordered = other_chunks[:5] + script_chunks[:2]
    evidence_lines = []
    for index, chunk in enumerate(ordered, start=1):
        evidence_lines.append(
            json.dumps(
                {
                    "id": index,
                    "source": chunk.source_label,
                    "content": _trim(chunk.content, 400),
                },
                ensure_ascii=True,
            )
        )

    # Script execution output — show separately so the LLM knows what to interpret.
    script_section = ""
    if script_chunks:
        script_section = (
            "\nScript execution output (interpret this to answer the question):\n"
            + "\n---\n".join(_trim(c.content, 1500) for c in script_chunks[:2])
            + "\n"
        )

    # Compact tool context — only include fields that add value; trim large blobs.
    tool_context: dict = {}
    if aggregate.get("calculations"):
        tool_context["calculations"] = aggregate["calculations"]
    if aggregate.get("dataset_inspection"):
        di = aggregate["dataset_inspection"]
        tool_context["dataset_inspection"] = {
            "total_files": di.get("total_files"),
            "extension_counts": di.get("extension_counts"),
            "csv_headers": di.get("csv_headers"),
        }
    if aggregate.get("web_pages"):
        tool_context["web_pages"] = [
            {"title": p.get("title", ""), "key_facts": p.get("key_facts", {}),
             "summary": _trim(p.get("summary", ""), 300)}
            for p in aggregate["web_pages"][:2]
        ]
    if aggregate.get("generated_python_artifacts"):
        tool_context["generated_python_artifacts"] = aggregate["generated_python_artifacts"][:1]

    has_script_output = bool(script_chunks)
    system_instruction = (
        "You are the grounded CardioMAS responder. "
        + (
            "A Python script was executed and its output is provided below. "
            "Interpret the output to directly answer the user's question in plain language. "
            if has_script_output else
            "Answer only with support from the provided evidence and tool outputs. "
            "If the evidence is insufficient, say so plainly. "
        )
        + "Return strict JSON with exactly these keys: answer (string), citations (list of int evidence ids), "
        "warnings (list of strings). No markdown, no extra keys."
    )

    user_content = f"Question: {query}\n"
    if evidence_lines:
        user_content += "\nEvidence:\n" + "\n".join(evidence_lines)
    user_content += script_section
    if tool_context:
        user_content += "\nTool context:\n" + json.dumps(tool_context, indent=2, default=str)
    user_content += "\n\nReturn JSON only."

    return [
        ChatMessage(role="system", content=system_instruction),
        ChatMessage(role="user", content=user_content),
    ]


def orchestrator_messages(
    query: str,
    tools: list[ToolSpec],
    observations: list[dict],
    scratchpad_text: str = "",
    step_reflection: bool = False,
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

    if step_reflection:
        json_spec = (
            "Return strict JSON with keys: thought, reflection, action, args. "
            "'reflection' must be one of: 'making_progress', 'stuck', 'sufficient'. "
            "Use 'stuck' if the last two steps gave no new information. "
            "Use 'sufficient' when you have enough to answer (equivalent to action='answer')."
        )
    else:
        json_spec = (
            "Return strict JSON with keys: thought, action, args."
        )

    system_content = (
        "You are CardioMAS, a medical dataset analysis agent. "
        "Decide the next action to take given the query and what you have observed so far. "
        + json_spec + " "
        "'thought' is your reasoning. 'action' is the tool name or 'answer' when done. "
        "'args' is a dict of arguments for the tool (empty dict when action is 'answer'). "
        "Do not repeat the same tool call with identical args. "
        "Say action='answer' when you have enough information."
    )

    user_parts = [f"Query: {query}"]
    if scratchpad_text:
        user_parts.append(scratchpad_text)
    user_parts.append(obs_text)
    user_parts.append(f"Available tools:\n{tool_lines}")
    user_parts.append("What is your next action? Return JSON only.")

    return [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content="\n".join(user_parts)),
    ]


def react_planner_messages(query: str, tools: list[ToolSpec]) -> list[ChatMessage]:
    tool_names = ", ".join(t.name for t in tools)
    return [
        ChatMessage(
            role="system",
            content=(
                "You are the CardioMAS pre-planner. Given a user query and available tools, "
                "create a concise ordered plan (2-5 steps) of tool names to call to answer the query. "
                "Put dataset exploration tools first (list_folder_structure, read_wfdb_dataset, read_dataset_website), "
                "then analysis tools (inspect_dataset, retrieve_corpus), "
                "then compute tools (generate_python_artifact) if calculation is needed. "
                "Return strict JSON: {\"plan\": [\"tool1\", \"tool2\", ...], \"reasoning\": \"one sentence\"}. "
                "Only use tool names from the provided list. No extra keys."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Query: {query}\n\n"
                f"Available tools: {tool_names}\n\n"
                "What is the best 2-5 step plan? Return JSON only."
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
