# Agentic RAG Redesign Plan

## What Agentic RAG Actually Means

Standard RAG retrieves once and generates once. Agentic RAG adds a **reasoning loop** around that:

- The agent decides **whether** to retrieve and **what** to retrieve
- After retrieving, it checks if the evidence is **sufficient** — if not, it retrieves again differently
- For complex questions it **decomposes** the query into atomic sub-queries first
- After generating an answer it **grades its own output** for groundedness — if not grounded, it loops
- Multiple **specialised agents** handle different task types (retrieval, computation, web, synthesis)
- **Memory** persists findings across sessions so the same work isn't repeated

---

## Current State: What Works and What Doesn't

### Keep (solid foundations)

| Component | Why keep |
|---|---|
| Retrieval (BM25 / dense / hybrid) | Correct, tested, modular — just needs to be callable in a loop |
| Tool registry | Good abstraction — just needs to be exposed to the agent loop |
| Code generation (`llm_coder.py`) | Stays as a specialised tool, already LLM-driven |
| Knowledge / corpus build | Fine as-is |
| `OllamaChatClient` + streaming | Multi-turn and streaming already supported |
| Safety / permissions | Keep, extend with per-agent policies |
| `EvidenceChunk`, `ToolResult` schemas | Good data structures |
| `AgentEvent` streaming protocol | Extend, not replace |

### Remove or Collapse

| Component | Why remove |
|---|---|
| `agentic/aggregator.py` | Evidence accumulation should be part of the agent's observation log, not a separate stage |
| Linear `planner.py` + `executor.py` combo | Replace with a ReAct loop; static upfront planning is the core of the problem |
| Heuristic planner fallback | Dead weight once the LLM drives the loop — if LLM is absent, return early with a clear message |
| `dataset_mode: script_only` as a config branch | Replace with a "code agent" that the orchestrator can call as a tool like any other |
| `responder.py` single-pass synthesis | Fold into the agent loop's final step with an optional self-reflection gate |
| `SCRIPT_FIRST_PLAN.md` / `LLM_CODEGEN_PLAN.md` | These were stepping stones; the new architecture supersedes them |

---

## New Architecture

```
User query
    │
    ▼
[Query Decomposer]          ← optional: splits "compare X and Y" into sub-queries
    │  sub-queries[]
    ▼
[ReAct Orchestrator]  ◄─────────────────────────────┐
    │  thinks: "what do I need next?"                │
    ▼                                                │
[Tool Router] ──────► retrieval | code | web | calc  │
    │  ToolResult                                    │
    ▼                                                │
[Retrieval Grader]          ← scores evidence for relevance
    │  "sufficient?" ──────────────────────────────── No (loop back)
    │  Yes
    ▼
[Answer Synthesizer]        ← single LLM call grounded in accumulated evidence
    │
    ▼
[Answer Grader]             ← optional: "is this hallucinated?"
    │  "grounded?" ────────────────────────────────── No (loop back)
    │  Yes
    ▼
QueryResult
```

---

## Components to Build

### 1. `agentic/react_agent.py` — ReAct Orchestrator

Replaces `planner.py` + `executor.py` + `aggregator.py` with a single loop:

```
for iteration in range(max_iterations):
    thought = llm("Given: query + observations so far, what tool should I call next and why?")
    if thought.action == "answer":
        break
    tool_result = registry.execute(thought.tool_name, **thought.args)
    observations.append(tool_result)
```

Key design decisions:
- **Thought format**: Free text "thought" + structured `{"action": "tool_name", "args": {...}}` — not JSON-only. The LLM reasons in natural language, then commits to a structured action.
- **Observation log**: Each tool result is appended as an observation. The next thought sees the full history.
- **Max iterations**: Configurable (`agent.max_iterations`, default 5). If reached, synthesise from what's accumulated.
- **Stop conditions**: LLM emits `action: "answer"`, or evidence is graded sufficient, or max iterations hit.

LLM prompts for this stage are **free text in, structured action out** — not the current JSON-constrained approach that forces a single upfront plan.

### 2. `agentic/query_decomposer.py` — Query Decomposition

For complex queries (multiple questions, comparisons, multi-hop), break into atomic sub-queries before the ReAct loop.

```python
def decompose(query: str, chat_client: ChatClient, config: AgentConfig) -> list[SubQuery]:
    ...
```

`SubQuery` has a `text` and a `query_type: Literal["factual", "computational", "exploratory"]`.

- `factual` → route to retrieval agent first
- `computational` → route to code agent first
- `exploratory` → route to both, synthesise

Simple queries (single question, no comparison, no "and") skip this step entirely — checked by a lightweight heuristic before hitting the LLM.

### 3. `agentic/retrieval_grader.py` — Relevance Grading

After retrieval, before feeding chunks to the synthesiser, score each chunk:

```python
def grade_chunks(query: str, chunks: list[EvidenceChunk], chat_client: ChatClient) -> GradedEvidence:
    ...
```

Returns:
- `relevant_chunks`: high-confidence chunks used for synthesis
- `partial_chunks`: lower-confidence chunks kept as backup
- `verdict: Literal["sufficient", "insufficient", "partial"]`

The orchestrator uses `verdict`:
- `sufficient` → proceed to synthesis
- `insufficient` → trigger web search or reformulate the retrieval query
- `partial` → proceed to synthesis but note low confidence in the result

This is the **Corrective RAG (CRAG)** pattern. The grader is a lightweight LLM call (small model, short prompt, binary grade per chunk).

### 4. `agentic/answer_grader.py` — Self-Reflection

After synthesising an answer, optionally check it:

```python
def grade_answer(query: str, answer: str, evidence: list[EvidenceChunk], chat_client: ChatClient) -> AnswerVerdict:
    ...
```

Returns `Literal["grounded", "hallucinated", "incomplete"]`.

- `grounded` → return answer to user
- `hallucinated` → discard answer, loop back with stricter retrieval prompt
- `incomplete` → loop back with a note about what's missing

This is the **Self-RAG** pattern. It should be **optional** (`agent.self_reflection: true/false`) because it adds latency and a second LLM call. Default off for speed, on for high-stakes queries.

### 5. `agentic/router.py` — Query Type Router

Classifies incoming queries and routes to the right specialised sub-agent:

| Query type | Routed to | Example |
|---|---|---|
| Dataset computation | Code agent | "How many unique patients?" |
| Knowledge lookup | Retrieval agent | "What is the scp_codes column?" |
| External research | Web agent | "What does this paper say?" |
| Complex / multi-hop | Orchestrator with decomposition | "Compare NORM vs AFIB distribution by age" |

The router is a single LLM call (or a keyword heuristic if no LLM configured) that returns a `RouteDecision`.

The key improvement: the **code agent path** is no longer a separate `dataset_mode` config — it's just a route that the orchestrator can take based on query content.

### 6. `memory/persistent.py` — Cross-Session Memory

Simple file-backed store (JSON) keyed by query hash. Stores:
- Query text + result summary
- Which evidence chunks were used
- Whether the answer was graded as grounded

On new queries:
1. Embed the query, find similar past queries (cosine similarity on stored embeddings)
2. If a highly similar past query exists and was grounded → surface it as a candidate answer
3. Agent still retrieves fresh evidence but past grounded answers reduce LLM calls

This is a lightweight alternative to a vector database. Single JSON file per config's `output_dir`, with a configurable max entry count.

### 7. New Prompts (`inference/prompts.py` additions)

Each agent role needs its own prompt system:

| Prompt function | Role | Output format |
|---|---|---|
| `orchestrator_messages()` | ReAct thought generation | Free text + `{"action": ..., "args": ...}` |
| `decomposer_messages()` | Query decomposition | `[{"text": ..., "type": ...}]` JSON |
| `retrieval_grader_messages()` | Per-chunk relevance | `{"grade": "relevant/partial/irrelevant", "reason": "..."}` JSON |
| `answer_grader_messages()` | Answer quality check | `{"verdict": "grounded/hallucinated/incomplete", "reason": "..."}` JSON |
| `router_messages()` | Query classification | `{"route": "code/retrieval/web/orchestrate", "reason": "..."}` JSON |
| `synthesiser_messages()` | Answer generation | `{"answer": "...", "citations": [...], "confidence": "high/medium/low"}` JSON |

The planner and responder prompts can be removed or kept as legacy.

---

## Config Changes

### Remove
- `autonomy.dataset_mode` — no longer needed; code generation is just a tool
- `autonomy.execute_for_answer` — execution is always part of the code tool's behaviour
- `autonomy.require_approval_for_shell_execution` — keep in safety config but not autonomy

### Add

```yaml
agent:
  mode: react                  # react | multi_agent | linear (linear = current behaviour, for migration)
  max_iterations: 5            # ReAct loop cap
  query_decomposition: false   # decompose complex queries before the loop
  self_reflection: false        # grade answer after synthesis
  retrieval_grading: true       # grade retrieved chunks before synthesis
  memory_mode: session          # session | persistent | none
  persistent_memory_max: 200    # max entries in persistent memory store
  router_model: ""              # model for routing/grading (can be smaller than code_model)
```

The `linear` mode preserves current behaviour exactly — useful for migration and testing.

### New `LLMConfig` field
```yaml
llm:
  router_model: ""             # lightweight model for grading/routing (defaults to model)
  router_max_tokens: 200       # grading prompts are short; don't waste tokens
```

---

## File Map: What Changes Where

```
src/cardiomas/
├── agentic/
│   ├── runtime.py              MODIFY — dispatch to react_agent or linear pipeline based on agent.mode
│   ├── react_agent.py          NEW    — ReAct orchestrator loop
│   ├── query_decomposer.py     NEW    — query decomposition
│   ├── retrieval_grader.py     NEW    — chunk relevance grading
│   ├── answer_grader.py        NEW    — answer self-reflection
│   ├── router.py               NEW    — query type routing
│   ├── planner.py              DEPRECATE (keep for linear mode compatibility)
│   ├── executor.py             DEPRECATE (keep for linear mode compatibility)
│   ├── aggregator.py           DEPRECATE (keep for linear mode compatibility)
│   └── responder.py            KEEP   — used by both linear and react modes
├── memory/
│   ├── session.py              KEEP
│   └── persistent.py           NEW    — file-backed cross-session memory
├── inference/
│   └── prompts.py              EXTEND — add 5 new prompt functions
├── schemas/
│   ├── config.py               MODIFY — add AgentConfig, new LLMConfig fields
│   └── runtime.py              MODIFY — add ReActStep, Thought, GradedEvidence schemas
└── tools/
    └── registry.py             MODIFY — expose tool descriptions in agent-friendly format
```

---

## Implementation Phases

### Phase 1 — ReAct Core (highest value, enables everything else)

1. Add `AgentConfig` to `schemas/config.py`
2. Write `agentic/react_agent.py` with the loop, observation log, stop conditions
3. Modify `agentic/runtime.py` to dispatch: `agent.mode == "react"` → `react_agent`, else existing linear path
4. Add `orchestrator_messages()` to `prompts.py`
5. Smoke-test: same queries as before, but now the agent can retrieve twice if first retrieval is sparse

### Phase 2 — Retrieval Grading (eliminates irrelevant chunks reaching the synthesiser)

6. Write `agentic/retrieval_grader.py`
7. Add `retrieval_grader_messages()` to `prompts.py`
8. Wire into ReAct loop: after each retrieval tool call, grade chunks; if insufficient, loop
9. Tests: mock LLM grades, verify orchestrator loops back correctly

### Phase 3 — Query Decomposition (enables multi-hop questions)

10. Write `agentic/query_decomposer.py`
11. Add `decomposer_messages()` to `prompts.py`
12. Wire into `react_agent.py`: if query is complex, decompose first, then run sub-queries through the loop
13. Tests: "compare X and Y" splits into two sub-queries

### Phase 4 — Routing (enables specialised sub-agents)

14. Write `agentic/router.py`
15. Add `router_messages()` to `prompts.py`
16. Wire into `runtime.py`: router runs before decomposer; direct simple computational queries straight to code tool without retrieval
17. Remove `dataset_mode: script_only` special-casing from config; the router handles it

### Phase 5 — Self-Reflection (optional, latency trade-off)

18. Write `agentic/answer_grader.py`
19. Add `answer_grader_messages()` to `prompts.py`
20. Wire as optional post-synthesis step behind `agent.self_reflection: true`
21. Configurable: hallucinated → loop back; incomplete → note in warnings; grounded → return

### Phase 6 — Persistent Memory (avoids repeated work)

22. Write `memory/persistent.py` with file-backed JSON store + embedding similarity search
23. Wire into `runtime.py`: before running the agent loop, check persistent memory for a similar past query
24. If found and grounded: surface as candidate, still run retrieval but with reduced depth

---

## What This Looks Like in Practice

**Before (current):**
```
cardiomas query "What is the age distribution in PTB-XL by diagnostic label?"

1. Planner (once): "use generate_python_artifact"
2. Executor (once): calls generate_python_artifact
3. Aggregator: deduplicates evidence
4. Responder (once): synthesises answer
```
If the script produces incomplete output, there's no recovery within the pipeline.

**After (Agentic RAG):**
```
cardiomas query "What is the age distribution in PTB-XL by diagnostic label?"

1. Router: "this is computational + factual" → route to orchestrator with code + retrieval tools
2. Decomposer: splits into:
   - "what diagnostic labels exist?" (factual → retrieval)
   - "what is the age distribution per label?" (computational → code)
3. ReAct loop, iteration 1:
   - Thought: "I need the label list first"
   - Action: retrieve_corpus("diagnostic labels")
   - Observe: 5 relevant chunks found
   - Grader: "sufficient" (label list confirmed)
4. ReAct loop, iteration 2:
   - Thought: "Now I need to compute the distribution"
   - Action: generate_python_artifact("age distribution by scp_codes label", dataset_path)
   - Observe: script written and executed, JSON output with mean/std per label
   - Grader: "sufficient"
5. Synthesiser: combines chunk evidence + script output → grounded answer
6. Answer Grader (if enabled): "grounded" → return
```

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| ReAct loop runs too many iterations | `max_iterations` cap + observation length limit |
| LLM doesn't emit valid action JSON | Parse with fallback to `action: "answer"` on malformed output |
| Grader model adds too much latency | `router_model` can be a smaller/faster model; grading prompts are short |
| Query decomposition creates too many sub-queries | Cap at 4 sub-queries; merge sub-results before synthesis |
| Persistent memory grows unbounded | `persistent_memory_max` cap with LRU eviction |
| Migration breaks existing users | `agent.mode: linear` preserves current behaviour exactly |
| Self-reflection loops forever | Hard cap: max 2 reflection loops regardless of verdict |

---

## Summary: Before vs. After

| Capability | Current | After |
|---|---|---|
| Retrieval | Once, fixed top_k | Iterative, graded, stops when sufficient |
| Planning | Static upfront plan | Think-act-observe loop |
| Query handling | Whole query as-is | Decomposes complex queries |
| Evidence quality | Unfiltered (score threshold only) | LLM-graded relevance |
| Answer quality | No check | Optional self-reflection |
| Dataset computation | Special `script_only` mode | Router-selected code tool, same loop |
| Memory | Session only | Session + optional persistent cross-session |
| Multi-source | One retrieval call | Agent decides which source per iteration |
| Streaming | Events per stage | Events per iteration + per thought |
