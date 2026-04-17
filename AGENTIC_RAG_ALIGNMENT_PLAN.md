# CardioMAS Fresh Agentic RAG Rebuild Plan

Date: 2026-04-17  
Source concept: `new_paln.md`

## 1. Decision

We will not preserve old implementations that are not working.

This plan assumes a fresh implementation of CardioMAS as a unified Agentic RAG system. Existing code may be read for ideas, but it is not a compatibility target. Backward compatibility with current `analyze`, `organize`, V2, V3, or V4 flows is not a requirement.

## 2. Goal

Rebuild CardioMAS into a clean Agentic RAG platform that matches the concept in `new_paln.md`:

- multiple knowledge sources,
- a reusable retrieval layer,
- a well-defined tool layer,
- an agent planner that decides retrieval and tool use,
- evidence aggregation and grounded answers,
- explicit memory/state policy,
- and strong safety, logging, and evaluation.

The rebuilt system should support:

- dataset understanding,
- evidence-grounded Q&A over project knowledge,
- paper and documentation analysis,
- tool-assisted research tasks,
- and domain-specific ECG workflows as one use case, not the only architecture.

## 3. Clean-Slate Principles

1. Do not wrap broken workflows.
2. Do not preserve legacy module boundaries if they conflict with the target design.
3. Do not carry forward old orchestration logic just because it exists.
4. Reuse code only if it is directly aligned with the new architecture and passes fresh tests.
5. Prefer deleting confusing legacy modules early instead of keeping parallel systems alive.

## 4. What Should Be Considered Legacy

These parts should be treated as replacement candidates, not as foundations that must survive:

- `src/cardiomas/agents/` as the old workflow-specific agent set
- `src/cardiomas/graph/workflow.py` as the old LangGraph routing graph
- `src/cardiomas/organization/` as the current deterministic organization flow
- `src/cardiomas/knowledge_department/`
- `src/cardiomas/coding_department/`
- `src/cardiomas/cardiology_department/`
- `src/cardiomas/testing_department/`
- `src/cardiomas/rag/` in its current narrow paper-centric form
- `src/cardiomas/schemas/state.py` in its current V2/V4-heavy workflow shape
- CLI commands that exist only to support the legacy flow shape

These modules can be mined for small utilities, but the new system should not be constrained by their current contracts.

## 5. Target Architecture

The rebuilt runtime should look like this:

```text
User query / config
    ->
Planner
    ->
Retriever + tool selection
    ->
Evidence collection
    ->
Evidence aggregation
    ->
Grounded response / artifact generation
    ->
Trace log + state + optional approval gate
```

### Core layers

1. Knowledge Sources
   - local files
   - dataset metadata
   - PDFs
   - internal documents
   - web sources
   - APIs

2. Retrieval Layer
   - BM25
   - embeddings
   - hybrid search
   - metadata filters
   - reranking

3. Tool Layer
   - internal retrievers
   - dataset inspection tools
   - web research tools
   - utility tools
   - optional action tools

4. Agent Layer
   - planner/router
   - tool executor
   - evidence aggregator
   - response generator
   - safety/approval controller

5. Memory / State Layer
   - session state
   - tool trace history
   - retrieval history
   - optional short-term memory
   - explicit no-memory mode

## 6. Proposed New Repository Shape

Replace the current workflow-centric structure with a smaller set of shared runtime modules.

```text
src/cardiomas/
  cli/
    main.py
  agentic/
    planner.py
    executor.py
    aggregator.py
    responder.py
    runtime.py
  knowledge/
    loaders.py
    chunking.py
    indexing.py
    corpus.py
    sources.py
  retrieval/
    bm25.py
    dense.py
    hybrid.py
    filters.py
    rerank.py
  tools/
    registry.py
    retrieval_tools.py
    research_tools.py
    dataset_tools.py
    utility_tools.py
    action_tools.py
  memory/
    session.py
    policy.py
  safety/
    policy.py
    approvals.py
    permissions.py
  schemas/
    config.py
    evidence.py
    tools.py
    memory.py
    runtime.py
  evaluation/
    benchmarks.py
    judges.py
    reports.py
```

## 7. Fresh Rebuild Phases

## Phase 0: Legacy Freeze and Removal Plan

Goal:
Stop expanding the old architecture and define the deletion boundary.

Tasks:

- Freeze work on legacy `agents/`, `graph/`, and department modules.
- Mark legacy commands and modules as deprecated in planning docs.
- Decide which files will be deleted immediately and which will remain temporarily during the cutover.
- Remove stale architecture docs that describe the old system as the target.

Deliverables:

- One legacy removal checklist
- One target architecture note
- One list of modules to delete in the first implementation PR

Acceptance criteria:

- The team agrees that the new system is a replacement, not a migration wrapper.

## Phase 1: New Core Skeleton

Goal:
Create the new package layout and shared schemas before implementing logic.

Tasks:

- Add the new `agentic/`, `knowledge/`, `retrieval/`, `memory/`, `safety/`, and `evaluation/` packages.
- Define shared runtime schemas:
  - `RuntimeConfig`
  - `KnowledgeSource`
  - `EvidenceChunk`
  - `Citation`
  - `ToolSpec`
  - `ToolResult`
  - `AgentDecision`
  - `SessionState`
  - `MemoryPolicy`
- Introduce a new config loader that is independent of legacy workflow flags.

Deliverables:

- importable package skeleton
- shared Pydantic schemas
- new YAML/JSON runtime config model

Acceptance criteria:

- The new architecture can be imported and tested without legacy modules.

## Phase 2: Knowledge Ingestion and Corpus Build

Goal:
Build a generic knowledge layer for all supported source types.

Tasks:

- Implement loaders for:
  - local markdown/text files
  - PDFs
  - dataset metadata
  - generated reports
  - web pages
- Normalize source metadata:
  - source path/URL
  - title
  - type
  - trust level
  - visibility
  - freshness
- Implement chunking and corpus persistence.

Deliverables:

- corpus builder
- chunk metadata model
- local artifact storage format

Acceptance criteria:

- The system can build a searchable corpus from mixed sources using one pipeline.

## Phase 3: Retrieval Layer

Goal:
Implement the retriever layer as a first-class subsystem, not a helper.

Tasks:

- Implement BM25 retrieval.
- Implement embedding retrieval.
- Implement hybrid retrieval and reranking.
- Add metadata filtering and source-aware ranking.
- Emit retrieval traces for every query.

Deliverables:

- `retrieve(query, filters, top_k)` service
- retrieval benchmarks
- evidence objects with citations

Acceptance criteria:

- The retriever works across multiple source types and returns citation-ready evidence.

## Phase 4: Tool Layer

Goal:
Build a uniform tool system the planner can reason about.

Tasks:

- Define tool registry and tool contracts.
- Wrap retrievers as tools.
- Add web research, dataset inspection, calculator, and utility tools.
- Add error handling, timeouts, and permission policies.
- Separate read-only tools from action tools.

Deliverables:

- tool registry
- tool metadata and safety boundaries
- standardized tool result schema

Acceptance criteria:

- Any tool can be selected by the planner through one consistent interface.

## Phase 5: Planner and Execution Runtime

Goal:
Implement the actual agentic behavior described in `new_paln.md`.

Tasks:

- Build a planner that decides:
  - whether retrieval is needed,
  - whether tools are needed,
  - which tool sequence to use,
  - whether more evidence is required,
  - when to stop.
- Build an executor that runs the plan step by step.
- Build an aggregator that merges evidence and records conflicts.
- Build a response generator that produces grounded output with citations.

Deliverables:

- planner
- executor
- aggregator
- responder
- runtime trace log

Acceptance criteria:

- A user query can trigger multi-step retrieval and tool use without hardcoded workflow routing.

## Phase 6: Memory and State Policy

Goal:
Add memory only where it improves the system.

Tasks:

- Add session-scoped memory.
- Add task-state memory for long workflows.
- Define no-memory mode as the default safe baseline.
- Record decisions, retrieved entities, and tool traces.

Deliverables:

- memory policy
- session state store
- state serialization rules

Acceptance criteria:

- The system supports multi-step work without forcing persistent memory into every use case.

## Phase 7: Safety, Permissions, and Approval

Goal:
Make the rebuilt system safe before feature expansion.

Tasks:

- Add trust-aware evidence handling.
- Add permission policies for web/API/action tools.
- Add approval gates for:
  - publishing,
  - expensive runs,
  - external actions,
  - low-trust evidence mixtures.
- Require the responder to separate retrieved facts from inference.

Deliverables:

- safety policy
- permission checks
- approval gate module
- refusal and uncertainty behavior

Acceptance criteria:

- Unsafe actions are blocked cleanly and low-confidence answers are surfaced honestly.

## Phase 8: Interface Rebuild

Goal:
Expose the new system through a clean user interface.

Tasks:

- Replace legacy CLI flows with a smaller interface surface.
- Add commands such as:
  - `cardiomas query`
  - `cardiomas build-corpus`
  - `cardiomas inspect-tools`
  - `cardiomas run-plan`
- Keep config-file-driven execution as a first-class path.

Deliverables:

- new CLI
- config-driven runtime
- structured JSON output mode

Acceptance criteria:

- Users can run the new system without knowing any legacy workflow concepts.

## Phase 9: Evaluation and Cutover

Goal:
Validate the rebuild and delete the legacy implementation.

Tasks:

- Create benchmark sets for:
  - direct fact retrieval,
  - multi-hop retrieval,
  - mixed tool queries,
  - failure cases,
  - refusal cases,
  - low-evidence cases.
- Measure:
  - retrieval quality,
  - tool choice quality,
  - faithfulness,
  - latency,
  - robustness.
- Delete legacy modules once the new baseline passes the acceptance suite.

Deliverables:

- evaluation suite
- regression fixtures
- legacy deletion PR

Acceptance criteria:

- The old implementation is removed from the default path and no longer documented as active architecture.

## 8. What To Delete Early

These should be deleted or archived early in the rebuild once the new skeleton is ready:

- old workflow routing in `src/cardiomas/graph/`
- old workflow-specific agent package in `src/cardiomas/agents/`
- deterministic department packages:
  - `knowledge_department/`
  - `coding_department/`
  - `cardiology_department/`
  - `testing_department/`
- old organization dispatcher in `src/cardiomas/organization/`
- old architecture docs that describe those flows as the intended design

## 9. What Can Be Reused Selectively

Re-use is allowed only case by case:

- low-level utility functions
- parsing helpers
- working research/web fetch helpers
- logging utilities
- selected schema ideas

Do not reuse:

- old control flow
- old state shape
- old command surface
- old assumptions about agent boundaries

## 10. Recommended First Implementation Sequence

1. Create the new package skeleton and config/runtime schemas.
2. Build the corpus and retrieval layers.
3. Build the tool registry and tool wrappers.
4. Build the planner, executor, aggregator, and responder.
5. Add safety and approval policies.
6. Replace the CLI with the new command surface.
7. Add evaluation coverage.
8. Delete the legacy architecture.

## 11. Non-Goals

- preserving old command behavior
- preserving old workflow names
- keeping parallel legacy and fresh runtimes for long
- adding persistent user memory in the first build
- production deployment before evaluation exists

## 12. Success Criteria

This rebuild is successful when:

- CardioMAS has one clean Agentic RAG architecture.
- Retrieval, tool use, and response generation are unified.
- The planner can choose tools and retrieval dynamically.
- Outputs are grounded, traceable, and citation-ready.
- Safety and approval gates are explicit.
- Legacy broken flows are removed instead of maintained.

## 13. Immediate Next Step

Start with Phase 0 and Phase 1 only:

- freeze the legacy architecture,
- define the deletion boundary,
- scaffold the fresh package layout,
- and introduce the new shared config and schema layer.

Do not spend time adapting the old workflows first. Build the new core, then cut over.
