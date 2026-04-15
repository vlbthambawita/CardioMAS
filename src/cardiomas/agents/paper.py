from __future__ import annotations

import logging
from typing import Any

from cardiomas.agents.base import AgentOutputError, run_structured_agent
from cardiomas.schemas.agent_outputs import PaperOutput
from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.tools.research_tools import fetch_webpage, read_pdf, search_arxiv
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def paper_agent(state: GraphState) -> GraphState:
    """Find and analyze the dataset paper to extract split methodology."""
    from cardiomas.llm_factory import get_llm_for_agent

    opts = state.user_options
    info = state.dataset_info
    state.execution_log.append(LogEntry(agent="paper", action="start"))
    vprint("paper", f"searching for paper — dataset: {info.name if info else '?'}")

    if info is None:
        state.errors.append("paper_agent: no dataset_info available")
        return state

    # Search arXiv
    vprint("paper", f"searching arXiv: '{info.name} ECG dataset electrocardiogram'")
    arxiv_result = search_arxiv.invoke({
        "query": f"{info.name} ECG dataset electrocardiogram", "max_results": 3
    })
    papers = arxiv_result.get("results", [])
    vprint("paper", f"found {len(papers)} arXiv result(s)")
    for p in papers:
        vprint("paper", f"  → {p['title']} ({p['url']})")

    # Fetch paper text — prefer registry URL, fall back to top arXiv result
    paper_text = ""
    paper_source = "none"

    if info.paper_url:
        vprint("paper", f"downloading PDF from registry URL: {info.paper_url}")
        pdf_result = read_pdf.invoke({"path_or_url": info.paper_url})
        paper_text = pdf_result.get("text", "")
        paper_source = info.paper_url
    elif papers:
        pdf_url = papers[0]["url"].replace("abs", "pdf")
        vprint("paper", f"downloading PDF: {pdf_url}")
        pdf_result = read_pdf.invoke({"path_or_url": pdf_url})
        paper_text = pdf_result.get("text", "")
        paper_source = papers[0]["url"]
        vprint("paper", f"extracted {len(paper_text)} chars from PDF")

    if not paper_text:
        vprint("paper", "no paper text found — recording absence")
        state.paper_findings = {
            "found": False,
            "source": "none",
            "official_splits_described": False,
            "methodology": "No paper found. Will generate custom splits.",
            "evidence": [],
        }
        state.execution_log.append(LogEntry(agent="paper", action="no_paper_found"))
        return state

    # RAG retrieval (Phase 3 hook — uses full text if RAG not available)
    evidence_context = _get_evidence_context(paper_text, paper_source, state)

    llm = get_llm_for_agent(
        "paper",
        prefer_cloud=opts.use_cloud_llm,
        agent_llm_map=opts.agent_llm_map,
    )

    prompt = (
        "Analyze the provided ECG dataset paper evidence and extract split methodology.\n\n"
        "For each claim:\n"
        "- Set 'found' to true if the paper describes splits\n"
        "- Set 'patient_level' based on whether splitting is at patient or recording level\n"
        "- Set 'official_ratios' only if exact numbers are stated\n"
        "- Add exact quoted sentences to 'evidence' for every claim\n"
        "- If information is absent, leave the field null and note it\n\n"
        f"Paper source: {paper_source}\n"
        f"Dataset: {info.name}"
    )

    try:
        output: PaperOutput = run_structured_agent(
            llm, "paper_analysis", prompt, PaperOutput, extra_context=evidence_context
        )
        output = output.model_copy(update={"paper_source_url": paper_source})
    except AgentOutputError as exc:
        logger.error(f"paper_agent: structured output failed — {exc}")
        state.errors.append(f"paper: {exc}")
        output = PaperOutput(
            found=bool(paper_text),
            notes=f"Structured extraction failed: {exc}",
            paper_source_url=paper_source,
        )

    # Store findings in state
    state.paper_findings = {
        "found": output.found,
        "source": paper_source,
        "split_methodology": output.split_methodology,
        "patient_level": output.patient_level,
        "stratify_by": output.stratify_by,
        "official_ratios": output.official_ratios,
        "exclusion_criteria": output.exclusion_criteria,
        "evidence": output.evidence,
        "notes": output.notes,
        "arxiv_results": [{"title": p["title"], "url": p["url"]} for p in papers],
    }

    state.execution_log.append(
        LogEntry(agent="paper", action="complete", detail=paper_source)
    )
    vprint(
        "paper",
        f"complete — found={output.found}"
        f" patient_level={output.patient_level}"
        f" official_ratios={output.official_ratios}"
        f" evidence={len(output.evidence)} quotes",
    )
    return state


def _get_evidence_context(paper_text: str, paper_source: str, state: GraphState) -> str:
    """Return evidence context — uses RAG if available (Phase 3), else raw text excerpt."""
    try:
        from cardiomas.rag.retriever import retrieve_evidence
        dataset_name = state.dataset_info.name if state.dataset_info else "unknown"
        evidence_chunks = retrieve_evidence(
            paper_text=paper_text,
            paper_source=paper_source,
            dataset_name=dataset_name,
            query="train test validation split stratification patient level ratio",
            top_k=5,
        )
        if evidence_chunks:
            blocks = "\n\n".join(
                f"[{i}] {chunk}" for i, chunk in enumerate(evidence_chunks)
            )
            return f"Evidence blocks retrieved from paper:\n\n{blocks}"
    except ImportError:
        pass
    except Exception as exc:
        logger.warning(f"RAG retrieval failed — falling back to raw text: {exc}")

    # Fallback: raw text excerpt (pre-Phase 3 behaviour)
    return paper_text[:6000]
