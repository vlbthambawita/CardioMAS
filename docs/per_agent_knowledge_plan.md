# Per-Agent Knowledge Base — Design Plan

## Goal

Allow users to define **named agents** in the YAML config, each with an optional private knowledge base (PDFs, web links) that is built and retrieved alongside the global corpus. The global corpus remains the shared foundation; per-agent knowledge supplements it without replacing it.

---

## Current State (what exists)

| Component | Location | Role |
|---|---|---|
| `KnowledgeSource` | `schemas/config.py` | Describes a single source (dataset_dir, pdf, web_page, …) |
| `RuntimeConfig.sources` | `schemas/config.py` | Global list of sources — one shared corpus |
| `build_corpus()` | `knowledge/corpus.py` | Builds a single `corpus.jsonl` from all global sources |
| `retrieve_corpus` tool | `tools/retrieval_tools.py` | Retrieves from the single global corpus |
| `AgentConfig` | `schemas/config.py` | Per-query ReAct loop settings (mode, max_iterations, …) |

`KnowledgeSource` already supports `pdf`, `web_page`, `local_file`, `local_dir`, `dataset_dir` — these can be reused for per-agent sources with no changes to the loader layer.

---

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Per-agent corpus storage | Separate `corpus.jsonl` per agent under `output_dir/agents/{name}/` | Avoids polluting global corpus; allows independent rebuilds |
| Retrieval merge strategy | Retrieve from both corpora, deduplicate, re-rank with RRF | Agent knowledge scores highest for its domain; global fills gaps |
| Agent selection | Explicit `--agent <name>` CLI flag or `default_agent` config key | Simple; avoids costly LLM-based routing for now |
| Fallback | If named agent has no private corpus built yet, use global only | Graceful degradation |
| Feature gate | `knowledge.enabled: false` (default) per agent — zero cost when off | Opt-in; no change to existing behaviour |
| Scraping control | New top-level `knowledge_scraping.enabled` flag in `RuntimeConfig` | Single switch user controls |

---

## New YAML Config Structure

```yaml
# ── Global (unchanged) ──────────────────────────────────────────────
system_name: CardioMAS
output_dir: runtime_output/ptbxl

sources:                          # global corpus sources
  - kind: dataset_dir
    path: /data/ptbxl-plus/1.0.1
    label: ptbxl

# ── Optional feature gate ────────────────────────────────────────────
knowledge_scraping:
  enabled: true                   # master switch; false = ignore all agent knowledge blocks

# ── Named agents (new) ──────────────────────────────────────────────
named_agents:
  - name: guidelines_expert
    description: "Answers questions grounded in ACC/AHA clinical ECG guidelines"
    knowledge:
      enabled: true
      sources:
        - kind: pdf
          path: docs/aha_ecg_guidelines_2023.pdf
          label: aha-2023
        - kind: web_page
          url: https://www.ahajournals.org/doi/10.1161/CIR.0000000000001123
          label: aha-stemi-guidelines
      retrieval:
        top_k: 5                  # agent-level override (falls back to global retrieval config)

  - name: ptbxl_analyst
    description: "Deep expert on PTB-XL dataset structure and statistics"
    knowledge:
      enabled: true
      sources:
        - kind: pdf
          path: docs/ptbxl_paper.pdf
          label: ptbxl-paper
        - kind: web_page
          url: https://physionet.org/content/ptb-xl/1.0.3/
          label: ptbxl-physionet
      retrieval:
        top_k: 3

# ── LLM / agent settings (global; can be overridden per agent) ───────
llm:
  provider: ollama
  model: gemma4:26b
  ...

agent:
  mode: react
  max_iterations: 10
  ...
```

---

## Schema Changes — `schemas/config.py`

### New models

```python
class AgentKnowledgeConfig(BaseModel):
    enabled: bool = False
    sources: list[KnowledgeSource] = Field(default_factory=list)
    retrieval: RetrievalConfig | None = None   # None = inherit global


class NamedAgentConfig(BaseModel):
    name: str                                  # unique slug used as directory name
    description: str = ""
    knowledge: AgentKnowledgeConfig = Field(default_factory=AgentKnowledgeConfig)
    # Optional per-agent overrides (None = inherit global)
    agent: AgentConfig | None = None
    llm: LLMConfig | None = None
    tools: ToolPolicyConfig | None = None

    @property
    def corpus_dir(self) -> str:              # resolved at RuntimeConfig level
        return ""                             # set by RuntimeConfig after construction


class KnowledgeScrapingConfig(BaseModel):
    enabled: bool = False                     # master opt-in switch
```

### Additions to `RuntimeConfig`

```python
class RuntimeConfig(BaseModel):
    ...
    knowledge_scraping: KnowledgeScrapingConfig = Field(
        default_factory=KnowledgeScrapingConfig
    )
    named_agents: list[NamedAgentConfig] = Field(default_factory=list)

    def agent_corpus_path(self, agent_name: str) -> Path:
        return Path(self.output_dir) / "agents" / agent_name / "corpus.jsonl"

    def agent_manifest_path(self, agent_name: str) -> Path:
        return Path(self.output_dir) / "agents" / agent_name / "corpus_manifest.json"

    def active_named_agent(self, name: str | None) -> NamedAgentConfig | None:
        if name is None:
            return None
        return next((a for a in self.named_agents if a.name == name), None)
```

---

## Corpus Build Pipeline Changes

### `knowledge/corpus.py` — two new functions

```python
def build_agent_corpus(
    agent: NamedAgentConfig,
    config: RuntimeConfig,
    force: bool = False,
) -> list[EvidenceChunk]:
    """Build and persist a private corpus for one named agent."""
    corpus_path = config.agent_corpus_path(agent.name)
    manifest_path = config.agent_manifest_path(agent.name)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    # Reuse existing load_source / chunk / persist logic — same as global build_corpus
    ...

def load_agent_corpus(agent_name: str, config: RuntimeConfig) -> list[EvidenceChunk]:
    """Load a pre-built agent corpus from disk (empty list if not built)."""
    path = config.agent_corpus_path(agent_name)
    if not path.exists():
        return []
    ...
```

### `agentic/runtime.py` — extend `build_corpus()`

```python
def build_corpus(self, force: bool = False) -> None:
    # 1. Build global corpus (unchanged)
    self._chunks = build_corpus(self._config, force=force)

    # 2. Build per-agent corpora if knowledge_scraping is enabled
    if self._config.knowledge_scraping.enabled:
        for agent_cfg in self._config.named_agents:
            if agent_cfg.knowledge.enabled:
                build_agent_corpus(agent_cfg, self._config, force=force)
```

---

## Retrieval Layer Changes

### `tools/retrieval_tools.py` — merged retrieval

When an active named agent has a private corpus, the `retrieve_corpus` tool merges results from both corpora using Reciprocal Rank Fusion (RRF — already implemented for hybrid retrieval):

```python
def retrieve_corpus(
    chunks: list[EvidenceChunk],          # global chunks
    query: str,
    config: RuntimeConfig,
    embedding_client: EmbeddingClient | None = None,
    top_k: int | None = None,
    agent_chunks: list[EvidenceChunk] | None = None,   # NEW — per-agent chunks
) -> ToolResult:
    global_results = _retrieve(chunks, query, config, embedding_client, top_k)
    if agent_chunks:
        agent_cfg_retrieval = ...         # per-agent retrieval config or global fallback
        agent_results = _retrieve(agent_chunks, query, ...)
        merged = _rrf_merge(global_results, agent_results)
        return _format_result(merged[:top_k])
    return _format_result(global_results)
```

### `tools/registry.py` — pass agent_chunks when building registry

```python
def build_registry(
    config: RuntimeConfig,
    chunks: list[EvidenceChunk],
    agent_chunks: list[EvidenceChunk] | None = None,   # NEW
    embedding_client: ...,
    autonomy_manager: ...,
) -> ToolRegistry:
    ...
    # retrieve_corpus lambda captures agent_chunks
    lambda query, top_k=None: retrieve_corpus(
        chunks=chunks,
        query=query,
        config=config,
        embedding_client=embedding_client,
        top_k=top_k,
        agent_chunks=agent_chunks,        # injected transparently
    )
```

The agent calls `retrieve_corpus` with exactly the same interface — it is unaware of the corpus merge happening underneath.

---

## Runtime Changes — `agentic/runtime.py`

```python
class AgenticRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        ...
        self._agent_chunks: dict[str, list[EvidenceChunk]] = {}   # name → chunks

    def build_corpus(self, force: bool = False) -> None:
        # global corpus (unchanged)
        self._chunks = build_corpus(self._config, force=force)
        # per-agent corpora
        if self._config.knowledge_scraping.enabled:
            for agent_cfg in self._config.named_agents:
                if agent_cfg.knowledge.enabled:
                    build_agent_corpus(agent_cfg, self._config, force=force)
                    self._agent_chunks[agent_cfg.name] = load_agent_corpus(
                        agent_cfg.name, self._config
                    )

    def query_stream(
        self,
        query: str,
        agent_name: str | None = None,    # NEW optional parameter
    ) -> Iterator[AgentEvent]:
        named = self._config.active_named_agent(agent_name)
        agent_chunks = self._agent_chunks.get(agent_name) if agent_name else None

        # Resolve effective config: named agent overrides global where set
        effective_agent_cfg  = named.agent or self._config.agent   if named else self._config.agent
        effective_llm_cfg    = named.llm   or self._config.llm     if named else self._config.llm
        effective_tools_cfg  = named.tools or self._config.tools   if named else self._config.tools

        registry = build_registry(
            config=self._config,
            chunks=self._chunks,
            agent_chunks=agent_chunks,    # merged transparently in retrieve_corpus
            embedding_client=self._embedding_client,
            autonomy_manager=...,
        )
        ...
```

---

## CLI Changes — `cli/main.py`

```python
# cardiomas build-corpus — rebuild one or all agent corpora
@app.command()
def build_corpus(
    config: str = typer.Option(...),
    force: bool = False,
    agent: str | None = typer.Option(None, help="Build only this named agent's corpus"),
):
    ...

# cardiomas query — select active agent
@app.command()
def query(
    question: str,
    config: str = typer.Option(...),
    agent: str | None = typer.Option(None, help="Named agent to use"),
    live: bool = False,
    json_output: bool = False,
):
    result = runtime.query(question, agent_name=agent)
    ...

# cardiomas list-agents — new command
@app.command()
def list_agents(config: str = typer.Option(...)):
    """List configured named agents and whether their corpus is built."""
    ...
```

**Usage examples:**
```bash
cardiomas build-corpus --config runtime.yaml                          # global + all agents
cardiomas build-corpus --config runtime.yaml --agent guidelines_expert  # only one agent
cardiomas query "What is LBBB criteria?" --config runtime.yaml --agent guidelines_expert
cardiomas list-agents --config runtime.yaml
```

---

## File Change Summary

| File | Change type | What changes |
|---|---|---|
| `schemas/config.py` | Add | `AgentKnowledgeConfig`, `NamedAgentConfig`, `KnowledgeScrapingConfig`; add fields to `RuntimeConfig` |
| `knowledge/corpus.py` | Add | `build_agent_corpus()`, `load_agent_corpus()` |
| `agentic/runtime.py` | Modify | `build_corpus()` builds per-agent corpora; `query_stream()` accepts `agent_name`; holds `_agent_chunks` dict |
| `tools/retrieval_tools.py` | Modify | `retrieve_corpus()` accepts optional `agent_chunks` and RRF-merges |
| `tools/registry.py` | Modify | `build_registry()` accepts `agent_chunks`, injects into `retrieve_corpus` lambda |
| `cli/main.py` | Modify | `--agent` flag on `query` and `build-corpus`; new `list-agents` command |
| `api.py` | Modify | `query()` / `query_stream()` accept `agent_name` keyword arg |

No changes needed to:
- `knowledge/loaders.py` — `load_source()` already handles pdf and web_page
- `knowledge/chunking.py` — chunking is source-agnostic
- `retrieval/` BM25/dense/hybrid retrievers — same interface, called separately per corpus
- All existing YAML configs — zero-change backward compatibility (`named_agents` defaults to `[]`, `knowledge_scraping.enabled` defaults to `false`)

---

## Phased Implementation

### Phase 1 — Schema & config parsing (no behaviour change)
Add the new Pydantic models and `RuntimeConfig` fields. Existing configs still load without change. Write schema unit tests.

### Phase 2 — Corpus build
Implement `build_agent_corpus()` and `load_agent_corpus()`. Wire into `runtime.build_corpus()`. Extend `build-corpus` CLI with `--agent` flag. Test with a small PDF and one web page.

### Phase 3 — Retrieval merge
Extend `retrieve_corpus()` tool with `agent_chunks` blending. Pass through `build_registry()` → `runtime.query_stream()`. Add `--agent` flag to CLI `query` command.

### Phase 4 — Per-agent config overrides
Honor `named.agent`, `named.llm`, `named.tools` overrides in `query_stream()`. Add `list-agents` command. Update `api.py`.

### Phase 5 — `_resolve_relative_paths` extension
Ensure PDF `path` values inside `named_agents[*].knowledge.sources` are resolved relative to the config file (same logic as top-level `sources`).
