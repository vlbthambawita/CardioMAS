# CardioMAS V3 — Improvement Plan

> **Status:** Approved for implementation  
> **Branch:** `feature/v3-rag-structured-reasoning`  
> **Phases:** 4 (independently mergeable)

---

## Table of Contents

1. [Root-Cause Diagnosis](#1-root-cause-diagnosis)
2. [SOTA Techniques — Applicability Matrix](#2-sota-techniques--applicability-matrix)
3. [Improvement Strategy Overview](#3-improvement-strategy-overview)
4. [Phase 1 — Structured Output Contracts](#phase-1--structured-output-contracts)
5. [Phase 2 — Dataset-Aware Semantic Mapper](#phase-2--dataset-aware-semantic-mapper)
6. [Phase 3 — RAG-Augmented Paper Agent](#phase-3--rag-augmented-paper-agent)
7. [Phase 4 — ReAct Verification Loops](#phase-4--react-verification-loops)
8. [Step-by-Step Implementation Checklist](#8-step-by-step-implementation-checklist)
9. [Expected Improvements by Phase](#9-expected-improvements-by-phase)
10. [What Is Not Changing](#10-what-is-not-changing)
11. [Branch & Review Protocol](#11-branch--review-protocol)

---

## 1. Root-Cause Diagnosis

### 1.1 Poor Reasoning Capability

| Gap | Root cause | Location in code |
|---|---|---|
| Agents return free-form text that downstream code must parse ad-hoc | `run_agent()` returns `response.content` (raw string); each agent does `json.loads(...)` or string search | `agents/base.py`, `agents/discovery.py`, `agents/nl_requirement.py` |
| No self-verification after output generation | Agent writes to state and returns immediately; no post-generation validation | All agents |
| Paper agent hallucinates when paper not found | Falls back to empty `paper_findings`; no grounding signal for downstream agents | `agents/paper.py` — fallback block |
| Orchestrator routing is a fixed `if/elif` chain, not adaptive | Despite claiming "dynamic routing", decisions are hardcoded in Python, not informed by state quality | `agents/orchestrator.py` |
| Context compression silently drops information | Second LLM call compresses context > 6000 chars with no verification that critical facts survived | `agents/base.py` — compress branch |

### 1.2 Poor Dataset Directory & Structure Understanding

| Gap | Root cause |
|---|---|
| Analysis agent reads only the first CSV header + 5 sample rows | `read_csv_metadata` called once on one file; dataset structure assumed uniform |
| WFDB, HDF5, EDF, and numpy formats ignored | `read_wfdb_header` tool exists but is never called by the analysis agent |
| ID field identification is LLM guesswork | Discovery agent scrapes webpage and asks LLM to infer `ecg_id_field`; not validated against real files |
| Patient ↔ record mapping never constructed | `check_patient_leakage` requires a `patient_mapping` dict, but no agent builds it; leakage check always runs with `None` |
| Multi-label diagnosis not handled | Analysis prompts look for a single label column; real ECG datasets (PTB-XL `scp_codes`, MIMIC-IV) use dicts/lists |

### 1.3 Unreliable / Inconsistent Outputs

| Gap | Root cause |
|---|---|
| Splits backed by synthetic IDs when real IDs fail to load | Splitter silently falls back to `[f"record_{i:05d}" for i in range(n)]`; checksum becomes meaningless |
| PII regex produces false positives and false negatives | `\d{3}-\d{2}-\d{4}` matches any hyphenated number; name pattern matches abbreviations |
| Generated script SHA-256 verification silently skipped | Coder agent logs "skipped" if `SPLITS_SHA256=` absent from stdout instead of failing |
| JSON parse errors in agents swallowed as warnings | `nl_requirement.py` catches `json.JSONDecodeError`, stores raw text, continues; downstream uses wrong ratios |
| `stratify_by` field name never validated against actual data | NL requirement produces a field name; splitter uses it directly; silent fallback if column missing |

---

## 2. SOTA Techniques — Applicability Matrix

| Technique | Addresses | Fit for CardioMAS |
|---|---|---|
| **Structured output enforcement** (Ollama JSON mode + Pydantic `.with_structured_output()`) | Output inconsistency | **High** — eliminates fragile ad-hoc JSON parsing across all agents |
| **Hybrid RAG** (BM25 + dense embeddings, LanceDB) | Paper reasoning, dataset grounding | **High** — grounds paper agent on retrieved chunks with citations |
| **ReAct verification loop** (Thought → Action → Observation → Verify) | Reasoning reliability | **High** — each agent verifies output satisfies a checklist before returning |
| **Dataset semantic mapper** (recursive FS walk + format-aware readers) | Data structure understanding | **High** — builds a structured manifest of every file, format, and field |
| **Pydantic inter-agent contracts** | Output consistency | **High** — state transitions validated at every agent boundary |
| **Episodic memory** (SQLite + vector store of past successful runs) | Cross-dataset consistency | **Medium** — useful after ≥5 datasets processed; deferred to V4 |
| **Graph RAG** (entity-relation extraction over papers) | Paper methodology extraction | **Medium** — high benefit but adds complexity; deferred to V4 |
| **Self-consistency sampling** (majority vote over k outputs) | Reasoning reliability | **Medium** — triples LLM cost; reserved for high-stakes decisions only |
| **XGrammar / Outlines constrained decoding** | Output consistency | **Medium** — requires vLLM or custom Ollama build; blocked by local model constraint |
| **Tree-of-Thoughts for split planning** | Reasoning | **Low** — overkill; SHA-256 seeding already handles determinism |

---

## 3. Improvement Strategy Overview

Four self-contained phases, each independently deployable:

```
Phase 1  →  Structured Output Contracts      (eliminates inconsistency)
Phase 2  →  Dataset-Aware Semantic Mapper    (fixes data understanding)
Phase 3  →  RAG-Augmented Paper Agent        (fixes paper reasoning)
Phase 4  →  ReAct Verification Loops         (eliminates silent failures)
```

**Architecture after all phases:**

```
CLI / Python API
       │
       ▼
  Orchestrator (hub)
       │
       ├──► NL Requirement  →  StructuredOutput(ParsedRequirement)
       │
       ├──► Discovery       →  StructuredOutput(DiscoveryOutput)
       │                         └── validated against DatasetMap
       │
       ├──► Analysis        →  DatasetMapper (multi-format)
       │                         └── StructuredOutput(AnalysisOutput)
       │                              └── field-existence verified
       │
       ├──► Paper           →  RAG indexer (PDF→chunks→LanceDB)
       │                         └── Hybrid retrieval (BM25+dense)
       │                              └── StructuredOutput(PaperOutput+citations)
       │
       ├──► Splitter        →  real IDs from DatasetMap (no synthetic fallback)
       │                         └── overlap/completeness verified
       │
       ├──► Security        →  real patient_record_map from DatasetMap
       │
       ├──► Coder           →  script SHA-256 verified against manifest
       │
       └──► Publisher       →  unchanged
```

---

## Phase 1 — Structured Output Contracts

**Goal:** Eliminate the entire class of failures caused by free-form LLM text parsed with ad-hoc string manipulation.

### Mechanism

- Switch every agent's LLM call to `ChatOllama(format="json")` + LangChain `.with_structured_output(PydanticModel)`.
- Define one **output DTO per agent** in `src/cardiomas/schemas/agent_outputs.py`.
- Add `run_structured_agent(llm, skill_name, message, output_schema)` to `agents/base.py`.
- On `ValidationError`: send a correction prompt with the error message and retry once.
- On second failure: raise `AgentOutputError` — orchestrator routes to `end_with_error`.

### New Agent Output Schemas

```python
# src/cardiomas/schemas/agent_outputs.py

class DiscoveryOutput(BaseModel):
    dataset_name: str
    source_type: DatasetSource
    ecg_id_field: str
    num_records: Optional[int]
    official_splits: bool
    paper_url: Optional[str]
    notes: str

class PaperOutput(BaseModel):
    found: bool
    split_methodology: Optional[str]
    patient_level: bool
    stratify_by: Optional[str]
    official_ratios: Optional[dict[str, float]]
    evidence: list[str]          # cited sentences from paper

class AnalysisOutput(BaseModel):
    num_records: int
    id_field: str
    patient_id_field: Optional[str]
    label_field: Optional[str]
    label_type: Literal["single", "multi", "none"]
    recommended_strategy: str
    missing_data_fraction: float
    notes: str

class NLRequirementOutput(BaseModel):
    split_ratios: dict[str, float]
    stratify_by: Optional[str]
    exclusion_filters: list[str]
    patient_level: bool
    seed: Optional[int]
    notes: str
```

### Files Changed

| File | Change |
|---|---|
| `src/cardiomas/schemas/agent_outputs.py` | **New** — per-agent output DTOs |
| `src/cardiomas/agents/base.py` | Add `run_structured_agent()` + retry-with-correction |
| `src/cardiomas/agents/discovery.py` | Use `run_structured_agent(DiscoveryOutput)` |
| `src/cardiomas/agents/nl_requirement.py` | Use `run_structured_agent(NLRequirementOutput)` |
| `src/cardiomas/agents/paper.py` | Use `run_structured_agent(PaperOutput)` |
| `src/cardiomas/agents/analysis.py` | Use `run_structured_agent(AnalysisOutput)` |
| `tests/test_structured_output.py` | **New** — schema validation unit tests |

---

## Phase 2 — Dataset-Aware Semantic Mapper

**Goal:** Replace shallow CSV-only file inspection with a deep, multi-format understanding of the entire dataset directory — and use real record IDs everywhere.

### Mechanism

A new `DatasetMapper` runs as the first step of the analysis agent. It recursively walks the dataset directory, dispatches to format-specific readers, and produces a single `DatasetMap` object stored in `GraphState`.

### `DatasetMap` Schema

```python
# src/cardiomas/mappers/schemas.py

class FileInventory(BaseModel):
    path: str
    format: Literal["wfdb", "hdf5", "edf", "csv", "numpy", "unknown"]
    size_bytes: int
    record_ids: list[str]
    fields: list[str]
    sample_values: dict[str, Any]

class DatasetMap(BaseModel):
    root_path: str
    total_files: int
    format_distribution: dict[str, int]
    all_record_ids: list[str]            # deduplicated, sorted
    patient_record_map: dict[str, list[str]]  # patient_id → [record_ids]
    id_field: str                        # confirmed from data, not guessed
    patient_id_field: Optional[str]
    label_field: Optional[str]
    label_values: list[Any]              # unique label values
    label_type: Literal["single", "multi", "none"]
    metadata_files: list[str]
    signal_files: list[str]
    missing_data_fraction: float
    dataset_checksum: str                # SHA-256 of sorted record IDs
```

### Format Readers

| Reader | Handles | Key extraction |
|---|---|---|
| `csv_reader.py` | Annotation CSVs | ID field detection, patient mapping, label column, value_counts |
| `wfdb_reader.py` | PhysioNet WFDB | Record ID from filename, header fields, sampling rate |
| `hdf5_reader.py` | MIMIC-IV-ECG HDF5 | Dataset keys as record IDs, attribute fields |
| `edf_reader.py` | European EDF | Signal channel names, patient info from header |
| `numpy_reader.py` | Numpy arrays | Shape, dtype (record count inference) |

### Key Integration Points

- `GraphState` gains `dataset_map: Optional[DatasetMap]` field.
- `splitter_agent` reads `state.dataset_map.all_record_ids` — **synthetic ID fallback removed entirely** (pipeline halts with clear error if IDs cannot be extracted).
- `security_agent` uses `state.dataset_map.patient_record_map` for real patient leakage detection.
- `DatasetMap.dataset_checksum` replaces the previous checksum (which was based on whatever IDs happened to load) in `ReproducibilityConfig`.

### Files Added / Changed

| File | Change |
|---|---|
| `src/cardiomas/mappers/__init__.py` | **New** |
| `src/cardiomas/mappers/dataset_mapper.py` | **New** — orchestrates all readers |
| `src/cardiomas/mappers/schemas.py` | **New** — `DatasetMap`, `FileInventory` |
| `src/cardiomas/mappers/format_readers/csv_reader.py` | **New** |
| `src/cardiomas/mappers/format_readers/wfdb_reader.py` | **New** |
| `src/cardiomas/mappers/format_readers/hdf5_reader.py` | **New** |
| `src/cardiomas/mappers/format_readers/edf_reader.py` | **New** |
| `src/cardiomas/mappers/format_readers/numpy_reader.py` | **New** |
| `src/cardiomas/schemas/state.py` | Add `dataset_map` field to `GraphState` |
| `src/cardiomas/agents/analysis.py` | Call `DatasetMapper.build()` before LLM |
| `src/cardiomas/agents/splitter.py` | Use `dataset_map.all_record_ids`; remove synthetic fallback |
| `src/cardiomas/agents/security.py` | Use `dataset_map.patient_record_map` |
| `tests/test_mapper.py` | **New** — mapper unit tests with fixture data |

---

## Phase 3 — RAG-Augmented Paper Agent

**Goal:** Replace single-shot full-text LLM analysis of papers with chunk-and-retrieve RAG, enabling small local models to reliably extract split methodology with cited evidence.

### Mechanism

```
PDF text
   │
   ▼
Chunk (400-token, overlapping paragraphs)
   │
   ├──► BM25 index  (rank_bm25)
   └──► Dense embeddings  (nomic-embed-text via Ollama)
              │
              ▼
         LanceDB  (local, zero-server)
              │
         Query: "train test split stratification patient"
              │
   ├──► BM25 top-10 results
   └──► Dense top-10 results
              │
              ▼
       Reciprocal Rank Fusion (RRF)
              │
       Top-5 chunks  →  injected as <evidence> in prompt
              │
              ▼
       LLM (structured output)  →  PaperOutput + evidence citations
```

### RAG Module

```
src/cardiomas/rag/
    __init__.py
    paper_indexer.py   # PDF → chunks → embed → LanceDB
    retriever.py       # BM25 + dense + RRF merge
```

**LanceDB rationale:** Pure Python, zero-server, stores locally in `output/{dataset}/rag/`, works fully offline after indexing, compatible with gemma4-scale embedding models.

**Embedding model:** `nomic-embed-text` (pulled via Ollama) — 137M params, runs locally, 768-dim embeddings.

### Prompt Change

System prompt (`skills/paper_analysis.md`) gains:

```
You will be given evidence blocks extracted from the paper.
Each block is labeled [0], [1], ... [4].

Rules:
- Only make claims supported by the provided evidence blocks.
- For each claim, cite the evidence index: e.g. "patient-level splits [2]".
- If no evidence supports a claim, write "not found in paper".
- Do not use prior knowledge about the dataset.
```

### Files Added / Changed

| File | Change |
|---|---|
| `src/cardiomas/rag/__init__.py` | **New** |
| `src/cardiomas/rag/paper_indexer.py` | **New** — chunking + embedding + LanceDB |
| `src/cardiomas/rag/retriever.py` | **New** — BM25 + dense + RRF |
| `src/cardiomas/agents/paper.py` | Use RAG pipeline; inject evidence into prompt |
| `src/cardiomas/skills/paper_analysis.md` | Add citation requirement + evidence format |
| `src/cardiomas/schemas/agent_outputs.py` | Add `evidence: list[str]` to `PaperOutput` |
| `pyproject.toml` | Add `lancedb`, `rank-bm25` to `[rag]` extras |
| `tests/test_rag.py` | **New** — indexer + retriever unit tests |

### New CLI Flag

```bash
cardiomas analyze /data/ptb-xl/ --no-rag   # disable RAG (use old single-shot)
```

RAG is **on by default** once Phase 3 is merged.

---

## Phase 4 — ReAct Verification Loops

**Goal:** Catch agent output errors at the source, before they propagate silently through the pipeline.

### Mechanism

A new `src/cardiomas/agents/verification.py` module provides pure-Python verification functions (no LLM cost unless a correction prompt is triggered).

### 4a — Analysis Agent Field Validation

After LLM produces `AnalysisOutput`, verify:

```python
assert output.id_field in dataset_map.all_fields,
    f"id_field '{output.id_field}' not found. Available: {dataset_map.all_fields}"

if output.label_field:
    assert output.label_field in dataset_map.all_fields,
        f"label_field '{output.label_field}' not found."

assert abs(output.num_records - len(dataset_map.all_record_ids)) < 10,
    f"num_records mismatch: LLM said {output.num_records}, found {len(dataset_map.all_record_ids)}"
```

On failure → send correction prompt with actual field list → retry once → `AgentOutputError` if still failing.

### 4b — Splitter Agent Integrity Check

After generating `proposed_splits`, verify deterministically:

```python
train, val, test = splits["train"], splits["val"], splits["test"]

assert set(train) & set(val)  == set(), "train/val overlap"
assert set(train) & set(test) == set(), "train/test overlap"
assert set(val)   & set(test) == set(), "val/test overlap"
assert len(train) + len(val) + len(test) == len(all_record_ids),
    "split sizes don't sum to total record count"
```

No LLM involved. On failure → raise `SplitIntegrityError` (do not fall back to synthetic IDs).

### 4c — Coder Agent Script Verification

Replace current "check for SPLITS_SHA256= string" with:

```python
result = execute_script("generate_splits.py")

# Step 1: assert script ran successfully
assert result["exit_code"] == 0, f"Script failed: {result['stderr']}"

# Step 2: parse SHA-256 from stdout (hard failure if absent)
match = re.search(r"SPLITS_SHA256=([a-f0-9]{64})", result["stdout"])
assert match, "Script did not output SPLITS_SHA256=<hex>"
script_sha256 = match.group(1)

# Step 3: compare against manifest
assert script_sha256 == manifest_sha256,
    f"SHA-256 mismatch: script={script_sha256}, manifest={manifest_sha256}"
# On mismatch → send script + diff to LLM for correction → regenerate → retry once
```

### Files Added / Changed

| File | Change |
|---|---|
| `src/cardiomas/agents/verification.py` | **New** — `verify_analysis_output()`, `verify_split_integrity()`, `verify_script_sha256()` |
| `src/cardiomas/agents/analysis.py` | Call `verify_analysis_output()` after LLM |
| `src/cardiomas/agents/splitter.py` | Call `verify_split_integrity()` after split generation |
| `src/cardiomas/agents/coder.py` | Call `verify_script_sha256()` after script execution |
| `src/cardiomas/agents/orchestrator.py` | Handle `AgentOutputError` as terminal route |
| `tests/test_verification.py` | **New** — verification unit tests |

---

## 8. Step-by-Step Implementation Checklist

### Setup

- [ ] Create branch `feature/v3-rag-structured-reasoning`
- [ ] Create scaffold directories

### Phase 1 — Structured Output Contracts

- [ ] `src/cardiomas/schemas/agent_outputs.py` — DiscoveryOutput, PaperOutput, AnalysisOutput, NLRequirementOutput
- [ ] `src/cardiomas/agents/base.py` — add `run_structured_agent()` + retry-with-correction
- [ ] `src/cardiomas/agents/discovery.py` — use `run_structured_agent`
- [ ] `src/cardiomas/agents/nl_requirement.py` — use `run_structured_agent`
- [ ] `src/cardiomas/agents/paper.py` — use `run_structured_agent`
- [ ] `src/cardiomas/agents/analysis.py` — use `run_structured_agent`
- [ ] `tests/test_structured_output.py` — schema validation tests
- [ ] All 26 existing tests pass

### Phase 2 — Dataset Semantic Mapper

- [ ] `src/cardiomas/mappers/schemas.py` — DatasetMap, FileInventory
- [ ] `src/cardiomas/mappers/format_readers/csv_reader.py`
- [ ] `src/cardiomas/mappers/format_readers/wfdb_reader.py`
- [ ] `src/cardiomas/mappers/format_readers/hdf5_reader.py`
- [ ] `src/cardiomas/mappers/format_readers/edf_reader.py`
- [ ] `src/cardiomas/mappers/format_readers/numpy_reader.py`
- [ ] `src/cardiomas/mappers/dataset_mapper.py`
- [ ] `src/cardiomas/schemas/state.py` — add `dataset_map` field
- [ ] `src/cardiomas/agents/analysis.py` — integrate DatasetMapper
- [ ] `src/cardiomas/agents/splitter.py` — use real IDs, remove synthetic fallback
- [ ] `src/cardiomas/agents/security.py` — use real patient_record_map
- [ ] `tests/fixtures/` — small synthetic ECG CSV for tests
- [ ] `tests/test_mapper.py` — mapper unit tests
- [ ] All existing + Phase 1 tests pass

### Phase 3 — RAG Paper Agent

- [ ] Add `lancedb`, `rank-bm25` to `pyproject.toml`
- [ ] `src/cardiomas/rag/paper_indexer.py`
- [ ] `src/cardiomas/rag/retriever.py`
- [ ] `src/cardiomas/agents/paper.py` — integrate RAG
- [ ] `src/cardiomas/skills/paper_analysis.md` — add citation requirement
- [ ] `src/cardiomas/schemas/agent_outputs.py` — add `evidence` field to PaperOutput
- [ ] `--no-rag` CLI flag in `cli/main.py`
- [ ] `tests/test_rag.py` — indexer + retriever tests
- [ ] All existing + Phase 1-2 tests pass

### Phase 4 — ReAct Verification Loops

- [ ] `src/cardiomas/agents/verification.py`
- [ ] `src/cardiomas/agents/analysis.py` — call `verify_analysis_output()`
- [ ] `src/cardiomas/agents/splitter.py` — call `verify_split_integrity()`
- [ ] `src/cardiomas/agents/coder.py` — call `verify_script_sha256()`
- [ ] `src/cardiomas/agents/orchestrator.py` — handle `AgentOutputError`
- [ ] `tests/test_verification.py`
- [ ] All existing + Phase 1-3 tests pass

### Finalisation

- [ ] Update `README.md` architecture diagram (V3)
- [ ] Update `CLAUDE.md` — new modules, new CLI flags
- [ ] Update `pyproject.toml` version bump preparation
- [ ] PR against `master`

---

## 9. Expected Improvements by Phase

| Phase | Metric | Before | After |
|---|---|---|---|
| **1** Structured output | Agent JSON parse failure rate | ~15–30% on small models | <2% (schema-enforced + 1 retry) |
| **1** | Incorrect field names in state | Common | Eliminated by type validation |
| **2** Dataset mapper | Records correctly identified | ~60% (CSV-only, one file) | ~95% (multi-format, full scan) |
| **2** | Patient leakage check coverage | 0% (no mapping built) | 100% (real patient→record map) |
| **2** | Reproducibility claim validity | ~40% (often synthetic IDs) | 100% (real IDs or hard stop) |
| **3** RAG paper agent | Split methodology extraction accuracy | ~50% (hallucination-prone) | ~80% (retrieved + cited evidence) |
| **3** | Hallucinated split ratios | Frequent on small models | Rare (evidence-grounded) |
| **4** Verification | Silent errors reaching publisher | Several known paths | Zero (caught at source agent) |
| **4** | Script SHA-256 mismatch undetected | Yes (silently skipped) | No (hard failure + retry) |

---

## 10. What Is Not Changing

| Component | Reason |
|---|---|
| LangGraph hub-and-spoke architecture | Well-designed; no structural change needed |
| SHA-256 deterministic seeding algorithm | Correct and sufficient |
| Checkpoint / resume mechanism | Working correctly |
| Security audit gate | Correct design; Phase 4 makes it more reliable but doesn't restructure |
| CLI / API public surface | No breaking changes; only additive (`--no-rag` flag) |
| Dataset registry YAML | Format unchanged; DatasetMapper adds to it, not replaces it |
| Session logging / recorder | No change |

---

## 11. Branch & Review Protocol

```bash
# Create branch
git checkout -b feature/v3-rag-structured-reasoning

# Commit per phase
git commit -m "feat(phase1): structured output contracts"
git commit -m "feat(phase2): dataset semantic mapper"
git commit -m "feat(phase3): rag-augmented paper agent"
git commit -m "feat(phase4): react verification loops"

# PR against master only after all phases pass CI
gh pr create --base master --title "V3: RAG + structured outputs + semantic mapper + verification"
```

Each phase is independently mergeable if early delivery is needed. **Phase 1 is recommended as the first merge** — highest ROI, lowest risk.

---

*Plan written: 2026-04-15 | Author: CardioMAS team*
