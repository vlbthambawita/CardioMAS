from __future__ import annotations

import logging
from typing import Any

from cardiomas.agents.base import run_agent
from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.tools.research_tools import search_arxiv, read_pdf, fetch_webpage
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def paper_agent(state: GraphState) -> GraphState:
    """Find and analyze the dataset paper to extract split methodology."""
    from cardiomas.llm_factory import get_llm

    opts = state.user_options
    info = state.dataset_info
    state.execution_log.append(LogEntry(agent="paper", action="start"))
    vprint("paper", f"searching for paper — dataset: {info.name if info else '?'}")

    if info is None:
        state.errors.append("paper_agent: no dataset_info available")
        return state

    # Search for papers
    vprint("paper", f"searching arXiv: '{info.name} ECG dataset electrocardiogram'")
    arxiv_result = search_arxiv.invoke({"query": f"{info.name} ECG dataset electrocardiogram", "max_results": 3})
    papers = arxiv_result.get("results", [])
    vprint("paper", f"found {len(papers)} arXiv result(s)")
    for p in papers:
        vprint("paper", f"  → {p['title']} ({p['url']})")

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
        vprint("paper", "no paper text found — skipping LLM analysis")
        state.paper_findings = {
            "found": False,
            "source": "none",
            "official_splits_described": False,
            "methodology": "No paper found. Will generate custom splits.",
        }
        state.execution_log.append(LogEntry(agent="paper", action="no_paper_found"))
        return state

    llm = get_llm(prefer_cloud=opts.use_cloud_llm, temperature=0.1)
    prompt = (
        "Analyze this ECG dataset paper and extract:\n"
        "1. Are official train/val/test splits defined? (yes/no)\n"
        "2. If yes, how are they defined? (ratio, fold number, explicit mapping)\n"
        "3. Is patient-level or record-level splitting used?\n"
        "4. What stratification criteria are used (diagnosis, demographics)?\n"
        "5. Any data exclusion criteria mentioned?\n"
        "6. Label/diagnosis distribution mentioned?\n\n"
        "Cite the exact page/section for every claim. If information is not in the text, say so explicitly."
    )
    response = run_agent(llm, "paper_analysis", prompt, paper_text[:6000])

    state.paper_findings = {
        "found": True,
        "source": paper_source,
        "analysis": response,
        "arxiv_results": [{"title": p["title"], "url": p["url"]} for p in papers],
    }
    state.execution_log.append(LogEntry(agent="paper", action="complete", detail=paper_source))
    vprint("paper", f"complete — source: {paper_source}")
    return state
