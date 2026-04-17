# CardioMAS Ollama Integration Plan

Date: 2026-04-17  
Scope: next-stage improvement of the fresh Agentic RAG rebuild on `master`

## 1. Current State

The current CardioMAS runtime is structured like an agentic system, but it is not yet model-driven.

- `agentic/planner.py` is heuristic.
- `agentic/responder.py` builds answers from templates.
- `retrieval/dense.py` is token-overlap scoring, not embeddings.
- There is no provider layer, no model config, and no local inference integration.

This plan upgrades the system to a local-first Ollama-backed runtime without reintroducing old workflow code.

## 2. Target Outcome

CardioMAS should support:

- local LLM inference through Ollama,
- local embedding generation through Ollama,
- grounded answer synthesis from retrieved evidence,
- optional LLM-based planning and tool selection,
- explicit model-role configuration in the runtime config,
- safe fallback to non-LLM behavior when Ollama is unavailable.

The first production target is a single-node local setup:

- Ollama running at `http://localhost:11434`
- one chat/instruct model for reasoning and answer generation
- one embedding model for retrieval

## 3. Design Principles

1. Keep Ollama integration behind a provider boundary.
2. Prefer Ollama native APIs first, not a heavyweight SDK dependency.
3. Do not hard-code model names in runtime logic.
4. Keep deterministic fallback behavior for testing and offline debugging.
5. Preserve grounded answers by forcing citation-aware prompts and evidence limits.

## 4. Proposed Architecture Changes

## New packages

```text
src/cardiomas/
  inference/
    __init__.py
    base.py
    ollama.py
    prompts.py
```

## Existing packages to extend

- `src/cardiomas/schemas/config.py`
- `src/cardiomas/retrieval/`
- `src/cardiomas/agentic/`
- `src/cardiomas/tools/registry.py`
- `src/cardiomas/knowledge/corpus.py`
- `src/cardiomas/cli/main.py`
- `src/cardiomas/api.py`

## 5. Config Model Changes

Add explicit provider and model settings to the runtime config.

Example shape:

```yaml
llm:
  provider: ollama
  base_url: http://localhost:11434
  planner_model: llama3.2
  responder_model: llama3.2
  temperature: 0.1
  max_tokens: 800
embeddings:
  provider: ollama
  base_url: http://localhost:11434
  model: embeddinggemma
  batch_size: 32
```

Notes:

- Model names above are examples, not enforced defaults.
- The config should allow one shared model or separate planner/responder models.
- Missing `llm` or `embeddings` sections should keep the current heuristic fallback path available.

## 6. Implementation Phases

## Phase 1: Provider Foundation

Goal:
Create a small inference layer for Ollama chat and embeddings.

Tasks:

- Add `InferenceClient` and `EmbeddingClient` interfaces in `inference/base.py`.
- Implement `OllamaChatClient` and `OllamaEmbeddingClient` in `inference/ollama.py`.
- Support:
  - model listing,
  - health check,
  - chat completion,
  - embedding generation.
- Add structured error handling for:
  - connection refused,
  - missing model,
  - timeout,
  - malformed response.

Acceptance criteria:

- CardioMAS can confirm whether Ollama is reachable.
- The runtime can request a completion and embeddings through one internal interface.

## Phase 2: Embedding Retrieval

Goal:
Replace the fake dense retriever with real vector retrieval.

Tasks:

- Add embedding-aware corpus/index artifacts under `output_dir`.
- Generate embeddings during `build-corpus`.
- Store:
  - chunk text,
  - metadata,
  - embedding vector,
  - embedding model name.
- Implement cosine similarity retrieval over stored vectors.
- Update `retrieval/hybrid.py` to combine BM25 plus true embedding similarity.

Acceptance criteria:

- `retrieval.mode: dense` uses Ollama embeddings, not token overlap.
- `retrieval.mode: hybrid` combines lexical and vector retrieval.

## Phase 3: LLM Grounded Response Generation

Goal:
Replace template-only responses with LLM-generated grounded answers.

Tasks:

- Add prompt templates in `inference/prompts.py`.
- Update `agentic/responder.py` to:
  - pass retrieved evidence into the prompt,
  - require citation markers or source references,
  - keep answers constrained to available evidence,
  - fall back to deterministic response building if no model is configured.
- Add response validation:
  - non-empty answer,
  - citation count limit,
  - warning if answer is unsupported by retrieved evidence.

Acceptance criteria:

- With Ollama enabled, `query()` returns a model-generated grounded answer.
- Without Ollama, the current deterministic answer path still works.

## Phase 4: LLM-Based Planning and Tool Use

Goal:
Move from keyword routing to model-assisted planning.

Tasks:

- Add a planning prompt that selects among enabled tools.
- Keep tool execution in local Python code.
- Preserve current safety checks in `safety/`.
- Support two planner modes:
  - `heuristic`
  - `ollama`
- Start with constrained structured output:
  - strategy,
  - ordered tool steps,
  - arguments.

Acceptance criteria:

- The planner can choose retrieval, dataset inspection, calculation, and web fetch from a prompt.
- Invalid tool plans are rejected and retried or downgraded to heuristic planning.

## Phase 5: Runtime Safety and Guardrails

Goal:
Prevent local-model freedom from weakening grounding and safety.

Tasks:

- Add prompt rules that forbid unsupported claims.
- Refuse action tools unless explicitly enabled.
- Add model-output validation before tool execution.
- Record prompt, model, and tool traces in the query result for debugging.

Acceptance criteria:

- Unsafe or malformed tool plans do not execute.
- Query traces show which model and prompt path were used.

## Phase 6: Evaluation

Goal:
Measure whether Ollama actually improves the system.

Tasks:

- Expand `evaluation/benchmarks.py` with:
  - retrieval tests,
  - grounded answer tests,
  - dataset inspection QA cases.
- Add benchmark runs for:
  - heuristic baseline,
  - Ollama responder only,
  - Ollama planner + responder.
- Track:
  - citation precision,
  - answer completeness,
  - unsupported claim rate,
  - latency.

Acceptance criteria:

- The Ollama path beats the baseline on answer quality without unacceptable regressions in grounding.

## Phase 7: CLI, Docs, and Examples

Goal:
Make the Ollama path runnable without code edits.

Tasks:

- Extend `cardiomas show-config` to display model/provider settings.
- Add a CLI health check command such as `cardiomas check-ollama --config ...`.
- Add example configs:
  - `examples/ollama/runtime_local.yaml`
  - `examples/ollama/runtime_ptbxl.yaml`
- Update `README.md` with:
  - Ollama install/pull steps,
  - config examples,
  - corpus build flow,
  - query flow,
  - troubleshooting.

Acceptance criteria:

- A user can install, pull models, build a corpus, and run grounded queries using documented commands alone.

## 7. Testing Plan

Add tests for:

- config validation for `llm` and `embeddings`,
- Ollama client HTTP behavior with mocked responses,
- corpus build with embedding persistence,
- dense and hybrid retrieval correctness,
- responder fallback when Ollama is unavailable,
- planner structured-output validation,
- end-to-end CLI runs with mocked Ollama responses.

## 8. Recommended Delivery Order

1. Provider config and Ollama client
2. Embedding-backed dense retrieval
3. LLM responder with grounded prompts
4. Planner upgrade with structured tool selection
5. CLI health check and documentation
6. Benchmarks and evaluation report

## 9. Definition of Done

This plan is complete when CardioMAS can:

- load an Ollama-enabled config,
- verify Ollama connectivity,
- build a vector-backed corpus using Ollama embeddings,
- answer questions with a grounded Ollama-generated response,
- optionally plan tool use with an Ollama model,
- and pass a test suite that still covers non-LLM fallback behavior.
