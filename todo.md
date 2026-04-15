# CardioMAS: Cardio Multi-Agent System

## Project Overview

A locally-runnable multi-agent system that analyzes ECG datasets, generates reproducible train/validation/test splits, publishes split metadata (record IDs only) to HuggingFace Datasets, and maintains a public GitHub page with corresponding HF links. Delivered as a CLI tool, pip-installable package, and Python library.

**Key Links:**
- **GitHub**: https://github.com/vlbthambawita/CardioMAS
- **PyPI**: `cardiomas` *(name verified available — PyPI returns 404)*
- **HuggingFace Dataset**: `vlbthambawita/ECGBench`
- **Install**: `pip install cardiomas`

---

## Architecture Decisions

- **Agent Framework**: LangGraph (graph-based orchestration of agents)
- **Local LLM Runtime**: Ollama (default), with optional cloud LLM fallback (OpenAI, Anthropic, etc.)
- **Package Name**: `cardiomas` (PyPI) / `CardioMAS` (GitHub)
- **HF Dataset Repo**: `vlbthambawita/ECGBench` (stores only split manifests + analysis reports, never raw data)
- **Agent Communication**: LangGraph message graph with typed state
- **Tool Ecosystem**: LangChain tool interface for all agent tools
- **Reproducibility**: deterministic hashing of dataset + split parameters → same input always yields same output

---

## 1. Project Scaffold

- [ ] Initialize Python project with `pyproject.toml` (PEP 621)
  - Package name: `cardiomas`
  - Entry point: `cardiomas` CLI command
  - Python >=3.10
  - Use dynamic versioning from git tags (see Section 14)
- [ ] Set up directory structure:
  ```
  CardioMAS/
  ├── pyproject.toml
  ├── README.md
  ├── LICENSE
  ├── CHANGELOG.md
  ├── .github/
  │   └── workflows/
  │       ├── ci.yml                   # Tests + lint on PR
  │       └── publish.yml              # Auto-publish to PyPI on tag push
  ├── src/
  │   └── cardiomas/
  │       ├── __init__.py              # __version__ (auto from tag)
  │       ├── cli/                     # CLI interface (typer)
  │       │   ├── __init__.py
  │       │   └── main.py
  │       ├── agents/                  # All agent definitions
  │       │   ├── __init__.py
  │       │   ├── orchestrator.py
  │       │   ├── discovery.py
  │       │   ├── paper.py
  │       │   ├── analysis.py
  │       │   ├── splitter.py
  │       │   ├── security.py
  │       │   └── publisher.py
  │       ├── tools/                   # LangChain-compatible tools
  │       ├── graph/                   # LangGraph orchestration
  │       │   ├── __init__.py
  │       │   ├── state.py
  │       │   └── workflow.py
  │       ├── schemas/                 # Pydantic models for state, configs, splits
  │       ├── datasets/                # Dataset registry, loaders, format handlers
  │       ├── splitters/               # Splitting strategies
  │       ├── publishing/              # HF push, GitHub page update
  │       ├── security/                # Data leak checker
  │       ├── skills/                  # Agent skill .md files
  │       └── config.py                # Global config, paths, defaults
  ├── tests/
  ├── data/                            # Local dataset cache (gitignored)
  └── docs/
  ```
- [ ] Add `.gitignore` (exclude `data/`, model caches, `.env`, `dist/`, `*.egg-info`)
- [ ] Add `LICENSE` (Apache 2.0 or MIT)
- [ ] Add `CHANGELOG.md` (keep-a-changelog format)
- [ ] Set up `pre-commit` hooks (ruff, mypy)

---

## 2. Configuration & LLM Backend

- [ ] Create `config.py` with:
  - `DATA_DIR`: local dataset storage path (default `~/.cardiomas/data/`)
  - `HF_REPO_ID`: `vlbthambawita/ECGBench`
  - `GITHUB_REPO`: `vlbthambawita/CardioMAS`
  - `OLLAMA_MODEL`: default local model (e.g., `llama3.1:8b`, `mistral`, `deepseek-coder`)
  - `CLOUD_LLM_PROVIDER`: optional cloud fallback (`openai`, `anthropic`, `none`)
  - `CLOUD_LLM_MODEL`: cloud model name if enabled
  - `SEED`: global random seed for reproducibility
- [ ] Create `llm_factory.py`:
  - `get_local_llm()` → returns LangChain `ChatOllama` instance
  - `get_cloud_llm()` → returns LangChain `ChatOpenAI` / `ChatAnthropic` if configured
  - `get_llm(prefer_cloud=False)` → router that returns local by default, cloud if requested or if local fails
  - Health check: verify Ollama is running, correct model is pulled
- [ ] Create `.env.example` with all configurable env vars
- [ ] Validate Ollama connectivity at startup; provide clear error if not running

---

## 3. CLI Interface (`cardiomas` command)

- [ ] Choose CLI framework: `typer` (recommended for type hints + auto docs)
- [ ] Implement top-level commands:

### `cardiomas analyze`
```
cardiomas analyze <dataset_source> [OPTIONS]

Arguments:
  dataset_source       URL (HF, PhysioNet, etc.) or local path to dataset

Options:
  --local-path PATH    Explicit local path (skip download)
  --output-dir PATH    Where to save analysis report
  --force-reanalysis   Re-run even if HF already has analysis
  --use-cloud-llm      Use cloud LLM instead of local Ollama
  --seed INT           Override global seed (default: 42)
  --custom-split SPEC  Custom split spec (e.g., "train:0.7,val:0.15,test:0.15")
  --ignore-official    Bypass official splits, generate fresh
  --stratify-by FIELD  Stratify splits by metadata field
  --verbose            Show agent reasoning steps
```

### `cardiomas status`
```
cardiomas status <dataset_name>
```
- Check if dataset already has analysis + splits on HF `vlbthambawita/ECGBench`
- Return summary of existing splits if available

### `cardiomas list`
```
cardiomas list [--remote] [--local]
```
- List datasets in local cache and/or available on HF `vlbthambawita/ECGBench`

### `cardiomas config`
```
cardiomas config [--show] [--set KEY VALUE]
```
- View or update configuration

### `cardiomas contribute`
```
cardiomas contribute <dataset_name> --split-file PATH
```
- Package user-provided splits for pull request to HF `vlbthambawita/ECGBench`

### `cardiomas verify`
```
cardiomas verify <dataset_name> [--seed INT]
```
- Re-run split generation and compare against published splits to confirm reproducibility

### `cardiomas version`
```
cardiomas version
```
- Print installed version (from git tag via `importlib.metadata`)

- [ ] Add progress bars (rich) for downloads and long-running steps
- [ ] Add `--dry-run` flag to preview actions without executing
- [ ] Ensure all CLI output is structured and parseable (JSON mode option with `--json`)

---

## 4. Dataset Registry & Loaders

- [ ] Create `DatasetSource` enum: `PHYSIONET`, `HUGGINGFACE`, `LOCAL`, `URL`, `KAGGLE`
- [ ] Create `DatasetInfo` Pydantic model:
  ```python
  class DatasetInfo(BaseModel):
      name: str
      source_type: DatasetSource
      source_url: str | None
      local_path: Path | None
      description: str | None
      paper_url: str | None
      official_splits: dict | None  # If known
      num_records: int | None
      metadata_fields: list[str]
      ecg_id_field: str  # Field used to uniquely identify ECG records
      sampling_rate: int | None
      num_leads: int | None
  ```
- [ ] Build dataset registry (YAML or JSON) for known ECG datasets:
  - MIMIC-IV-ECG
  - PTB-XL
  - Chapman-Shaoxing (CSN)
  - CODE-15%
  - CPSC 2018
  - Georgia 12-Lead
  - Ningbo
  - Add mechanism to register new datasets dynamically
- [ ] Implement loaders per source type:
  - [ ] `PhysioNetLoader`: download from PhysioNet (handle credentialed access)
  - [ ] `HuggingFaceLoader`: download via `datasets` library
  - [ ] `LocalLoader`: validate and index local directory
  - [ ] `GenericURLLoader`: download + detect format
- [ ] Download manager:
  - Check if dataset exists in `DATA_DIR` before downloading
  - If user provides `--local-path`, skip download, validate path exists
  - Resume interrupted downloads
  - Verify checksums if available
- [ ] Format detection and parsing:
  - WFDB (.hea, .dat)
  - HDF5 / H5
  - CSV/TSV metadata files
  - DICOM-ECG
  - EDF/EDF+
  - Parquet
  - Custom formats (detect via heuristics)

---

## 5. Agent Definitions

All agents operate within a LangGraph state graph. Each agent has a corresponding skill file in `src/cardiomas/skills/`.

### 5.1 Orchestrator Agent
- [ ] Create `orchestrator_agent.py`
- [ ] Create `skills/orchestrator.md` with agent instructions
- [ ] Responsibilities:
  - Receive user request (dataset source + options)
  - Check HF `vlbthambawita/ECGBench` for existing analysis (early exit if found, unless `--force`)
  - Dispatch sub-agents in correct order
  - Collect and validate outputs from all agents
  - Maintain execution log for reproducibility
  - Handle errors and retries gracefully
- [ ] State management: typed `GraphState` Pydantic model tracking:
  - Current phase
  - Dataset info
  - Analysis results
  - Proposed splits
  - Security audit results
  - Publishing status
  - Error log

### 5.2 Dataset Discovery Agent
- [ ] Create `discovery_agent.py`
- [ ] Create `skills/discovery.md`
- [ ] Responsibilities:
  - Parse user-provided URL or path
  - Identify dataset type and source
  - Locate associated papers (DOI, arXiv, PubMed)
  - Find official documentation / README / data description pages
  - Extract dataset metadata (num records, leads, sampling rate, labels)
  - Determine if official train/val/test splits exist
  - **Ground every claim**: cite source URL or file path for all extracted info
- [ ] Tools needed:
  - `web_scraper_tool`: fetch and parse dataset description pages
  - `paper_finder_tool`: search for associated papers via DOI, title, or dataset name
  - `file_inspector_tool`: list and peek at files in dataset directory

### 5.3 Paper Analysis Agent
- [ ] Create `paper_agent.py`
- [ ] Create `skills/paper_analysis.md`
- [ ] Responsibilities:
  - Download and parse dataset papers (PDF → text)
  - Extract split methodology described in the paper
  - Identify stratification criteria used by authors
  - Find patient-level vs record-level split details
  - Detect data exclusion criteria mentioned in papers
  - Extract diagnostic label distributions
  - **Output structured findings** with page/section citations
  - If no paper found, document this explicitly (do not fabricate)
- [ ] Tools needed:
  - `pdf_reader_tool`: extract text from PDF
  - `arxiv_search_tool`: find papers on arXiv
  - `pubmed_search_tool`: find papers on PubMed
  - `citation_extractor_tool`: extract and format citations

### 5.4 Data Analysis Agent
- [ ] Create `analysis_agent.py`
- [ ] Create `skills/data_analysis.md`
- [ ] Responsibilities:
  - Scan dataset files and build record inventory
  - Extract ECG record identifiers (unique IDs)
  - Parse metadata files (demographics, diagnoses, acquisition info)
  - Compute dataset statistics:
    - Total records, unique patients (if patient IDs available)
    - Diagnostic label distribution
    - Demographic distribution (age, sex if available)
    - Recording duration distribution
    - Lead configuration
    - Missing data patterns
  - Identify potential data quality issues
  - Identify grouping keys (patient ID, study ID) to prevent data leakage
  - **Never load raw ECG signal data into memory for analysis** — only metadata and IDs
  - Generate analysis report as structured markdown
- [ ] Tools needed:
  - `csv_reader_tool`: parse metadata CSVs
  - `wfdb_tool`: read WFDB headers (not signals)
  - `hdf5_inspector_tool`: inspect HDF5 structure
  - `statistics_tool`: compute distributions and summary stats

### 5.5 Split Strategy Agent
- [ ] Create `split_agent.py`
- [ ] Create `skills/split_strategy.md`
- [ ] Responsibilities:
  - **Priority 1**: Use official splits if they exist (from paper or dataset docs)
  - **Priority 2**: If user provides `--ignore-official` or `--custom-split`, generate custom splits
  - **Priority 3**: If no official splits, design splits based on analysis
  - Split design considerations:
    - Patient-level splitting (no patient appears in multiple sets) — critical
    - Stratification by diagnosis labels
    - Stratification by demographics if relevant
    - Handling multi-label records
    - Handling class imbalance
    - Minimum samples per class per split
  - Use deterministic algorithm: `seed` + sorted record IDs → hash → split assignment
  - Output: `SplitManifest` with train/val/test record ID lists
  - **Reproducibility guarantee**: same dataset + same seed + same parameters = same splits
- [ ] Implement splitting algorithms:
  - `PatientStratifiedSplit`: group by patient, stratify by label
  - `RecordStratifiedSplit`: stratify by label (when no patient IDs)
  - `OfficialSplit`: load from predefined mapping
  - `CustomSplit`: user-defined ratios and criteria
- [ ] Generate split statistics report:
  - Per-split label distribution
  - Per-split demographic distribution
  - Patient overlap check (must be zero)
  - Comparison to official splits (if available)

### 5.6 Security & Data Leak Agent
- [ ] Create `security_agent.py`
- [ ] Create `skills/security.md`
- [ ] Responsibilities:
  - **Pre-publish audit**: verify no raw ECG data in output
  - Check that split files contain ONLY record identifiers, not signal data
  - Verify no patient-identifiable information (PII) in outputs:
    - No names, dates of birth, medical record numbers
    - No direct identifiers in record IDs (flag if IDs look like MRNs)
  - Verify no data leakage across splits:
    - Same patient must not appear in train AND test
    - Same recording must not appear in multiple splits
  - Check file sizes (flag if suspiciously large — might contain signal data)
  - Validate that published splits reference records that exist in original dataset
  - Generate security audit report
  - **Block publishing** if any check fails
- [ ] Tools needed:
  - `pii_scanner_tool`: regex + heuristic PII detection
  - `file_size_checker_tool`: verify output sizes are reasonable
  - `overlap_checker_tool`: verify set disjointness

### 5.7 Publishing Agent
- [ ] Create `publishing_agent.py`
- [ ] Create `skills/publishing.md`
- [ ] Responsibilities:
  - Push split manifests to HuggingFace `vlbthambawita/ECGBench`
  - File structure on HF:
    ```
    vlbthambawita/ECGBench/
    ├── datasets/
    │   ├── ptb-xl/
    │   │   ├── analysis_report.md
    │   │   ├── splits.json           # {train: [...ids], val: [...ids], test: [...ids]}
    │   │   ├── split_metadata.json   # parameters, seed, strategy, date, cardiomas version
    │   │   └── official_splits.json  # if official splits exist
    │   ├── mimic-iv-ecg/
    │   │   └── ...
    │   └── ...
    ├── README.md                     # Auto-updated dataset index with links
    └── CONTRIBUTING.md               # PR instructions
    ```
  - Update `vlbthambawita/CardioMAS` GitHub README with links to new HF splits
  - Verify upload integrity (re-download and compare hashes)
  - Handle authentication (HF token, GitHub token)
  - Support dry-run mode
- [ ] Tools needed:
  - `hf_upload_tool`: push files to HuggingFace `vlbthambawita/ECGBench`
  - `github_update_tool`: update markdown files in `vlbthambawita/CardioMAS`
  - `hash_tool`: compute and verify file hashes

---

## 6. LangGraph Orchestration

- [ ] Define `GraphState` TypedDict or Pydantic model:
  ```python
  class GraphState(TypedDict):
      dataset_source: str
      dataset_info: DatasetInfo | None
      download_status: str  # pending, downloaded, skipped
      paper_findings: PaperFindings | None
      analysis_report: AnalysisReport | None
      existing_hf_splits: dict | None  # from vlbthambawita/ECGBench check
      proposed_splits: SplitManifest | None
      security_audit: SecurityAudit | None
      publish_status: str  # pending, published, failed, dry_run
      user_options: UserOptions
      execution_log: list[LogEntry]
      errors: list[str]
  ```
- [ ] Build agent graph:
  ```
  START
    │
    ▼
  [Check vlbthambawita/ECGBench for existing] ──(found & !force)──→ [Return existing] → END
    │ (not found or force)
    ▼
  [Discovery Agent] → [Paper Agent] → [Data Analysis Agent]
    │                                        │
    │              ┌─────────────────────────┘
    ▼              ▼
  [Split Strategy Agent]
    │
    ▼
  [Security Agent] ──(fail)──→ [Report issues] → END
    │ (pass)
    ▼
  [Publishing Agent] → END
  ```
- [ ] Add conditional edges:
  - Skip paper agent if no papers found by discovery
  - Skip download if `--local-path` provided
  - Skip publishing if `--dry-run`
  - Retry loop on transient failures (max 3)
- [ ] Add human-in-the-loop checkpoints:
  - After split proposal: show splits, ask user to confirm before publishing
  - After security audit: show any warnings
- [ ] Implement execution logging: every agent action recorded with timestamp, inputs, outputs

---

## 7. Tool Implementations

Build as LangChain `@tool` decorated functions or `BaseTool` subclasses.

### Data Tools
- [ ] `download_dataset_tool(url, dest_path)` → download with progress, resume support
- [ ] `list_dataset_files_tool(path)` → list files with types and sizes
- [ ] `read_csv_metadata_tool(path, nrows)` → read metadata CSV, return schema + sample
- [ ] `read_wfdb_header_tool(record_path)` → parse .hea file, return metadata (no signals)
- [ ] `inspect_hdf5_tool(path)` → return HDF5 structure, keys, shapes
- [ ] `compute_statistics_tool(data, columns)` → distributions, counts, percentiles

### Research Tools
- [ ] `search_arxiv_tool(query, max_results)` → search arXiv, return titles + abstracts + URLs
- [ ] `search_pubmed_tool(query, max_results)` → search PubMed
- [ ] `fetch_webpage_tool(url)` → fetch and extract text from web page
- [ ] `read_pdf_tool(path_or_url)` → extract text from PDF

### Splitting Tools
- [ ] `deterministic_split_tool(record_ids, ratios, seed, group_key)` → reproducible split
- [ ] `stratified_split_tool(record_ids, labels, ratios, seed, group_key)` → stratified split
- [ ] `check_overlap_tool(split_a, split_b)` → verify no overlap
- [ ] `compute_split_stats_tool(splits, metadata)` → per-split statistics

### Publishing Tools
- [ ] `check_hf_repo_tool(repo_id, dataset_name)` → check if dataset already on `vlbthambawita/ECGBench`
- [ ] `push_to_hf_tool(repo_id, files, commit_message)` → upload to `vlbthambawita/ECGBench`
- [ ] `update_github_readme_tool(repo, file_path, content)` → update `vlbthambawita/CardioMAS` files

### Security Tools
- [ ] `scan_for_pii_tool(data)` → scan for personally identifiable information
- [ ] `validate_split_file_tool(path)` → verify file contains only IDs, check size
- [ ] `check_patient_leakage_tool(splits, patient_mapping)` → verify patient-level disjointness

---

## 8. Reproducibility System

- [ ] Create `ReproducibilityConfig` model:
  ```python
  class ReproducibilityConfig(BaseModel):
      cardiomas_version: str
      seed: int
      dataset_name: str
      dataset_source_url: str
      dataset_checksum: str  # hash of dataset file listing
      split_strategy: str
      split_ratios: dict[str, float]
      stratify_by: list[str] | None
      group_by: str | None  # patient ID field
      custom_params: dict | None
      timestamp: datetime
  ```
- [ ] Deterministic pipeline:
  - Sort all record IDs lexicographically before splitting
  - Use SHA-256 hash of (sorted IDs + seed + strategy) as split seed
  - Document exact algorithm version
- [ ] Store `ReproducibilityConfig` alongside every split manifest on HF
- [ ] Verification command: `cardiomas verify <dataset_name>` re-runs split and compares to published

---

## 9. Python Library API

- [ ] Public API (`src/cardiomas/__init__.py`):
  ```python
  from cardiomas import CardioMAS

  mas = CardioMAS(ollama_model="llama3.1:8b", seed=42)

  # Check existing
  status = mas.status("ptb-xl")

  # Run full pipeline
  result = mas.analyze(
      source="https://physionet.org/content/ptb-xl/1.0.3/",
      force=False,
      use_cloud_llm=False,
  )

  # Get splits programmatically
  splits = mas.get_splits("ptb-xl")
  train_ids = splits["train"]
  val_ids = splits["val"]
  test_ids = splits["test"]

  # Custom splits
  custom = mas.analyze(
      source="/local/path/to/data",
      custom_split={"train": 0.7, "val": 0.15, "test": 0.15},
      ignore_official=True,
      stratify_by="diagnosis",
  )

  # Contribute splits
  mas.contribute("ptb-xl", split_file="my_splits.json")
  ```
- [ ] Type hints on all public methods
- [ ] Docstrings with examples on all public methods
- [ ] Support both sync and async usage

---

## 10. Agent Skill Files

Each agent has a `.md` skill file that serves as its system prompt and behavioral contract.

- [ ] `skills/orchestrator.md` — workflow coordination rules, error handling, when to escalate
- [ ] `skills/discovery.md` — how to identify dataset type, where to look for metadata, grounding rules
- [ ] `skills/paper_analysis.md` — how to extract split info from papers, citation requirements, what to do when no paper exists
- [ ] `skills/data_analysis.md` — metadata analysis checklist, statistics to compute, grouping key identification, ECG domain knowledge (leads, diagnoses, recording types)
- [ ] `skills/split_strategy.md` — splitting algorithm selection flowchart, priority order (official > paper-described > auto-generated), stratification best practices for ECG data, patient-level splitting rules
- [ ] `skills/security.md` — PII patterns to scan for, file size thresholds, data leakage detection rules, blocking criteria
- [ ] `skills/publishing.md` — HF repo structure, file naming conventions, commit message format, README update template, PR instructions

---

## 11. Contribution Workflow

- [ ] Write `CONTRIBUTING.md` for the HF `vlbthambawita/ECGBench` repo:
  - How to fork and submit a PR with custom splits
  - Required file format for split submissions
  - Validation steps the CI will run
- [ ] Create split submission schema:
  ```json
  {
    "dataset_name": "ptb-xl",
    "split_version": "community-v1",
    "author": "github_username",
    "description": "Custom splits stratified by superclass only",
    "cardiomas_version": "0.2.0",
    "reproducibility_config": { ... },
    "splits": {
      "train": ["00001", "00002", ...],
      "val": ["10001", ...],
      "test": ["20001", ...]
    }
  }
  ```
- [ ] Build validation script for PR CI:
  - Verify JSON schema
  - Verify no raw data included
  - Verify record IDs exist in original dataset (if accessible)
  - Verify no overlap between splits
  - Verify splits cover all records (or document excluded records)
- [ ] `cardiomas contribute` command packages user splits into correct format

---

## 12. Testing

- [ ] Unit tests:
  - [ ] Splitting algorithms (determinism, no overlap, correct ratios)
  - [ ] Security scanner (PII detection, file size checks)
  - [ ] Dataset loaders (each format)
  - [ ] CLI argument parsing
  - [ ] Reproducibility verification
- [ ] Integration tests:
  - [ ] End-to-end pipeline with a small test dataset (synthetic ECG metadata)
  - [ ] HF push/pull round-trip (to a test repo, not production `vlbthambawita/ECGBench`)
  - [ ] Ollama agent interaction (mock or live)
- [ ] Fixtures:
  - Create small synthetic ECG metadata fixtures for tests
  - Do NOT include real patient data in test fixtures
- [ ] CI/CD:
  - [ ] GitHub Actions `ci.yml` — tests + lint on every PR (see Section 14)
  - [ ] GitHub Actions `publish.yml` — auto-publish to PyPI on tag push (see Section 14)

---

## 13. Documentation

- [ ] `README.md`: project overview, quick start, architecture diagram, badges (PyPI version, CI status, license)
- [ ] `docs/architecture.md`: detailed system design, agent graph, data flow
- [ ] `docs/adding-datasets.md`: how to add support for a new ECG dataset
- [ ] `docs/reproducibility.md`: how the reproducibility system works
- [ ] `docs/contributing.md`: how to contribute splits or code
- [ ] `docs/configuration.md`: all config options explained
- [ ] Auto-generate CLI docs from typer
- [ ] API reference (sphinx or mkdocs with autodoc)

---

## 14. Packaging, Distribution & Automated PyPI Publishing

### pyproject.toml

- [ ] Configure `pyproject.toml` with dynamic versioning from git tags:
  ```toml
  [build-system]
  requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "cardiomas"
  dynamic = ["version"]
  description = "CardioMAS — Cardio Multi-Agent System for reproducible ECG dataset splits"
  readme = "README.md"
  license = {text = "Apache-2.0"}
  requires-python = ">=3.10"
  authors = [
      {name = "Vajira Thambawita", email = "vajira@simula.no"},
  ]
  classifiers = [
      "Development Status :: 3 - Alpha",
      "Intended Audience :: Science/Research",
      "Topic :: Scientific/Engineering :: Medical Science Apps.",
      "License :: OSI Approved :: Apache Software License",
      "Programming Language :: Python :: 3",
      "Programming Language :: Python :: 3.10",
      "Programming Language :: Python :: 3.11",
      "Programming Language :: Python :: 3.12",
  ]
  keywords = ["ecg", "cardiology", "multi-agent", "dataset", "splits", "reproducibility"]

  dependencies = [
      "langgraph>=0.2",
      "langchain>=0.3",
      "langchain-community",
      "typer[all]",
      "rich",
      "pydantic>=2.0",
      "huggingface-hub",
      "datasets",
      "wfdb",
      "pandas",
      "numpy",
      "scikit-learn",
      "h5py",
      "requests",
      "httpx",
      "pyyaml",
  ]

  [project.optional-dependencies]
  cloud = ["langchain-openai", "langchain-anthropic"]
  dev = ["pytest", "pytest-cov", "ruff", "mypy", "pre-commit"]

  [project.scripts]
  cardiomas = "cardiomas.cli.main:app"

  [project.urls]
  Homepage = "https://github.com/vlbthambawita/CardioMAS"
  Documentation = "https://github.com/vlbthambawita/CardioMAS/tree/main/docs"
  Repository = "https://github.com/vlbthambawita/CardioMAS"
  Issues = "https://github.com/vlbthambawita/CardioMAS/issues"
  "HuggingFace Dataset" = "https://huggingface.co/datasets/vlbthambawita/ECGBench"

  [tool.setuptools-scm]
  # Version is derived from git tags: v0.1.0 → 0.1.0
  version_scheme = "guess-next-dev"
  local_scheme = "no-local-version"
  ```

### Version strategy

- [ ] Use `setuptools-scm` — version is auto-derived from the latest git tag
- [ ] Tag format: `v0.1.0`, `v0.2.0`, `v1.0.0` (semver with `v` prefix)
- [ ] `__init__.py` reads version at runtime:
  ```python
  # src/cardiomas/__init__.py
  from importlib.metadata import version, PackageNotFoundError

  try:
      __version__ = version("cardiomas")
  except PackageNotFoundError:
      __version__ = "0.0.0-dev"
  ```

### GitHub Actions — CI (`ci.yml`)

- [ ] Create `.github/workflows/ci.yml`:
  ```yaml
  name: CI

  on:
    push:
      branches: [main]
    pull_request:
      branches: [main]

  jobs:
    test:
      runs-on: ubuntu-latest
      strategy:
        matrix:
          python-version: ["3.10", "3.11", "3.12"]
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: ${{ matrix.python-version }}
        - run: pip install -e ".[dev]"
        - run: ruff check src/ tests/
        - run: ruff format --check src/ tests/
        - run: pytest tests/ -v --cov=cardiomas --cov-report=xml
        - uses: codecov/codecov-action@v4
          if: matrix.python-version == '3.12'
  ```

### GitHub Actions — Auto-publish to PyPI on tag (`publish.yml`)

- [ ] Create `.github/workflows/publish.yml`:
  ```yaml
  name: Publish to PyPI

  on:
    push:
      tags:
        - "v*"   # Triggers on any tag starting with v (v0.1.0, v1.0.0, etc.)

  permissions:
    id-token: write   # Required for trusted publishing (no API token needed)

  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
          with:
            fetch-depth: 0   # Full history needed for setuptools-scm
        - uses: actions/setup-python@v5
          with:
            python-version: "3.12"
        - run: pip install build
        - run: python -m build
        - uses: actions/upload-artifact@v4
          with:
            name: dist
            path: dist/

    publish:
      needs: build
      runs-on: ubuntu-latest
      environment:
        name: pypi
        url: https://pypi.org/p/cardiomas
      steps:
        - uses: actions/download-artifact@v4
          with:
            name: dist
            path: dist/
        - uses: pypa/gh-action-pypi-publish@release/v1
          # Uses trusted publishing — no PYPI_API_TOKEN secret needed
          # Requires one-time setup: link GitHub repo to PyPI project at
          # https://pypi.org/manage/project/cardiomas/settings/publishing/
  ```

### PyPI Trusted Publishing Setup (one-time)

- [ ] Create the `cardiomas` project on PyPI (first manual upload or via pypi.org)
- [ ] Go to https://pypi.org/manage/project/cardiomas/settings/publishing/
- [ ] Add a "trusted publisher":
  - Owner: `vlbthambawita`
  - Repository: `CardioMAS`
  - Workflow name: `publish.yml`
  - Environment name: `pypi`
- [ ] Create a GitHub Environment named `pypi` in repo settings (Settings → Environments → New)
- [ ] Optionally add environment protection rules (require approval for releases)

### Release Workflow (how to publish a new version)

```bash
# 1. Ensure main is up to date
git checkout main && git pull

# 2. Update CHANGELOG.md with new version notes

# 3. Create and push a tag
git tag v0.1.0
git push origin v0.1.0

# 4. GitHub Actions automatically:
#    - Builds sdist + wheel using setuptools-scm (version from tag)
#    - Publishes to PyPI via trusted publishing
#    - Package available as: pip install cardiomas==0.1.0
```

### Additional distribution

- [ ] Build and test locally with `pip install -e .`
- [ ] Docker image for self-contained deployment (includes Ollama)
- [ ] Add PyPI badge to README: `[![PyPI](https://img.shields.io/pypi/v/cardiomas)](https://pypi.org/project/cardiomas/)`

---

## 15. Implementation Order (Recommended Phases)

### Phase 1 — Foundation (Week 1-2)
1. Project scaffold, `pyproject.toml`, directory structure
2. GitHub Actions CI + publish workflows
3. CLI skeleton with `analyze`, `status`, `list`, `verify`, `version` commands
4. `GraphState` and Pydantic schemas
5. LLM factory (Ollama + cloud fallback)
6. Dataset registry for known datasets (PTB-XL, MIMIC-IV-ECG)
7. First tagged release `v0.0.1` to test PyPI publishing pipeline

### Phase 2 — Core Agents (Week 3-4)
8. Discovery agent + web scraping tools
9. Data analysis agent + metadata parsing tools
10. Split strategy agent + splitting algorithms
11. Security agent + audit tools
12. Agent skill .md files

### Phase 3 — Paper Intelligence (Week 5)
13. Paper analysis agent + PDF reader + arXiv/PubMed tools
14. Official split extraction from papers

### Phase 4 — Publishing & Integration (Week 6)
15. Publishing agent + HF upload tools to `vlbthambawita/ECGBench`
16. GitHub `vlbthambawita/CardioMAS` README auto-update with HF links
17. LangGraph orchestration (full pipeline wiring)

### Phase 5 — Polish & Release (Week 7-8)
18. Python library API (`CardioMAS` class)
19. Contribution workflow + PR validation
20. Tests (unit + integration)
21. Documentation
22. Docker image
23. Tag `v0.1.0` — first stable release on PyPI

---

## Notes

- **No hallucinations rule**: Every agent must cite its source (URL, file path, page number) for every factual claim. If information is not found, the agent must say so explicitly rather than guessing.
- **Same input → same output**: The entire pipeline is deterministic given the same dataset, seed, and parameters. This is enforced at the algorithm level (sorted IDs, seeded RNG) and verified by the `cardiomas verify` command.
- **Data never republished**: Only record identifiers (ECG IDs, filenames, index numbers) are stored in `vlbthambawita/ECGBench`. Raw signals, images, or full metadata are never uploaded.
- **Local-first**: The system runs entirely on local hardware with Ollama by default. Cloud LLM usage is opt-in and only for complex reasoning tasks (e.g., paper analysis).
- **Auto-publish**: Pushing a `v*` tag to `vlbthambawita/CardioMAS` automatically builds and publishes to PyPI via trusted publishing. No API tokens needed after initial setup.