"""
LangGraph nodes — each function receives the full AgentState,
performs one job, and returns a partial state update.
"""

import os
import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState

# Shared LLM — all extract nodes reuse this instance
_llm = ChatOllama(model="gemma4:latest")

EXTRACT_SYSTEM = (
    "You are a precise data extraction agent. "
    "You receive raw scraped text from a website and extract structured information. "
    "Always respond in valid Markdown format as instructed — nothing else."
)


# ── Node 1: Scrape ────────────────────────────────────────────────────────────

def scrape_node(state: AgentState) -> dict:
    """Fetch the URL, strip noise, return clean text + metadata."""
    print(f"\n[ScrapeNode] Fetching: {state['url']}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(state["url"], headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ScrapeNode] ERROR: {e}")
        return {"error": f"Scrape failed: {e}"}

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else "Untitled"

    chunks = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 20:
            prefix = "\n## " if tag.name in ("h1", "h2", "h3", "h4") else ""
            chunks.append(f"{prefix}{text}")

    raw_text = "\n".join(chunks)[:12000]

    links = [
        {"label": a.get_text(strip=True), "url": a["href"]}
        for a in soup.find_all("a", href=True)
        if a["href"].startswith("http") and a.get_text(strip=True)
    ][:30]

    print(f"[ScrapeNode] Done — {len(raw_text)} chars, {len(links)} links")
    return {"title": title, "raw_text": raw_text, "links": links, "error": None}


# ── Node 2: Extract ───────────────────────────────────────────────────────────

def _llm_call(user_prompt: str) -> str:
    messages = [
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
    return _llm.invoke(messages).content.strip()


def extract_node(state: AgentState) -> dict:
    """Use Gemma 4 to extract summary, sections, and key facts."""
    print(f"[ExtractNode] Running Gemma 4 extraction...")

    title = state["title"]
    text = state["raw_text"]

    summary = _llm_call(
        f"Website title: {title}\n\nContent:\n{text[:4000]}\n\n"
        "Write a concise 3-5 sentence summary of this page. Return only the summary."
    )
    print(f"[ExtractNode] Summary done")

    sections = _llm_call(
        f"From the following website content, identify the main sections/topics covered.\n"
        f"Format as a Markdown bullet list with a one-line description for each.\n\n"
        f"Content:\n{text[:6000]}\n\nReturn only the Markdown list."
    )
    print(f"[ExtractNode] Sections done")

    key_facts = _llm_call(
        f"Extract up to 10 key facts or important data points from the following content.\n"
        f"Format as a Markdown bullet list.\n\nContent:\n{text[:6000]}\n\n"
        "Return only the bullet list."
    )
    print(f"[ExtractNode] Key facts done")

    return {"summary": summary, "sections": sections, "key_facts": key_facts}


# ── Node 3: ECG Expert ───────────────────────────────────────────────────────

ECG_EXPERT_SYSTEM = """You are a world-class expert in electrocardiography (ECG/EKG) and cardiac \
electrophysiology with deep knowledge spanning clinical cardiology, biomedical signal processing, \
and ECG dataset research.

Your expertise covers:

SIGNAL CHARACTERISTICS
- Waveform morphology: P wave (atrial depolarization, 80–100 ms, <0.25 mV), \
PR interval (120–200 ms), QRS complex (ventricular depolarization, 60–100 ms, \
axis −30° to +90°), ST segment, T wave (ventricular repolarization), U wave
- Standard acquisition: 12-lead system (I, II, III, aVR, aVL, aVF; V1–V6), \
sampling rates 250–1000 Hz (clinical standard 500 Hz), amplitude resolution 1 µV/LSB, \
bandwidth 0.05–150 Hz, paper speed 25 mm/s, gain 10 mm/mV

CLINICAL CONDITIONS DETECTABLE FROM ECG
- Arrhythmias: atrial fibrillation, flutter, SVT, VT, VF, heart blocks (1°/2°/3°), \
bundle branch blocks (LBBB, RBBB), WPW syndrome
- Ischemia & infarction: STEMI (ST elevation ≥1 mm), NSTEMI, T-wave inversions, \
pathological Q waves, Wellens syndrome
- Structural: LVH (Sokolow-Lyon, Cornell criteria), RVH, chamber enlargement
- Electrolyte & metabolic: hyper/hypokalemia (peaked T / U wave / flattening), \
hyper/hypocalcemia (QT changes), hypothermia (Osborn J wave)
- Drug effects: QTc prolongation, digitalis effect, antiarrhythmic changes

ECG DATASETS & DATA CHARACTERISTICS
- PhysioNet/MIT-BIH Arrhythmia Database: 48 records, 2-lead (MLII + V1/V5), \
360 Hz, 30 min, expert annotations (beat-level)
- PTB Diagnostic ECG Database: 549 records, 15 leads (12 standard + 3 Frank), \
1000 Hz, clinical diagnoses
- PTB-XL: 21 837 12-lead ECGs, 500 Hz/100 Hz, 10 s, SCP-ECG codes, demographics
- CPSC 2018 / CinC Challenge datasets, Georgia 12-Lead ECG Challenge
- Formats: WFDB (.dat/.hea/.atr), SCP-ECG, HL7 aECG, DICOM, CSV/EDF
- Annotation types: beat labels (N/V/A/…), rhythm labels, diagnostic codes (ICD-10, AHA/SCP)

SIGNAL PROCESSING & ML CONTEXT
- Preprocessing: baseline wander removal (high-pass >0.5 Hz), powerline noise \
(notch 50/60 Hz), EMG artifact suppression, Pan-Tompkins R-peak detection
- Feature extraction: RR intervals, HRV metrics (SDNN, RMSSD, pNN50, LF/HF), \
wavelet decomposition, morphological templates
- Deep learning approaches: CNN on raw waveforms, LSTM/GRU for sequences, \
transformer-based models (e.g., ECG-BERT), multi-lead fusion
- Evaluation: sensitivity/specificity per class, F1, AUC-ROC, challenge metrics

Always respond in Markdown. Structure your analysis clearly under appropriate subheadings."""


def ecg_expert_node(state: AgentState) -> dict:
    """Apply ECG domain expertise to the scraped content."""
    print(f"[ECGExpertNode] Analysing content with ECG domain knowledge...")

    text = state["raw_text"]
    title = state["title"]
    summary = state.get("summary", "")

    prompt = f"""You are reviewing content scraped from a website.

Page title: {title}

Summary already extracted:
{summary}

Full content:
{text[:8000]}

Perform a thorough ECG-domain analysis of this content. Structure your response with these sections \
(omit a section only if genuinely not applicable, and say so):

### ECG Relevance
Is this content related to ECG/cardiac signals? Rate relevance: High / Medium / Low / None. \
Explain briefly.

### Signal & Waveform Characteristics Mentioned
List any ECG signal properties, waveforms, intervals, or leads referenced. \
If none are mentioned, note what is absent and what would be expected for this type of resource.

### Clinical Conditions & Diagnoses
Identify any cardiac conditions, arrhythmias, or diagnostic criteria discussed.

### Dataset & Data Format Details
Extract any information about ECG datasets, file formats, sampling rates, lead configurations, \
annotation schemes, or data collection protocols. Flag gaps (e.g., missing sampling rate info).

### Signal Quality & Preprocessing Notes
Note any mention of noise, artifact, filtering, or preprocessing. Suggest what preprocessing \
steps would be standard for this type of data.

### Research & ML Applications
Identify any machine learning, deep learning, or signal processing methods mentioned. \
Relate them to state-of-the-art ECG analysis approaches.

### Expert Observations & Recommendations
Provide 3–5 expert-level insights or recommendations for someone working with ECG data \
from or related to this resource.

Return only the Markdown analysis, no preamble."""

    messages = [
        SystemMessage(content=ECG_EXPERT_SYSTEM),
        HumanMessage(content=prompt),
    ]
    analysis = _llm.invoke(messages).content.strip()
    print(f"[ECGExpertNode] Analysis done")
    return {"ecg_analysis": analysis}


# ── Node 4: Write ─────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:50]


def write_node(state: AgentState) -> dict:
    """Format the extracted data and save to a .md file."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    slug = _slugify(state["title"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{slug}_{timestamp}.md")

    lines = [
        f"# {state['title']}",
        "",
        f"> **Source:** {state['url']}",
        f"> **Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        state["summary"],
        "",
        "## Sections / Topics",
        "",
        state["sections"],
        "",
        "## Key Facts",
        "",
        state["key_facts"],
        "",
        "## ECG Expert Analysis",
        "",
        state.get("ecg_analysis", "_No ECG analysis available._"),
        "",
    ]

    if state.get("links"):
        lines += ["## Links Found", ""]
        for link in state["links"][:20]:
            lines.append(f"- [{link['label']}]({link['url']})")
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[WriteNode] Saved: {filepath}")
    return {"output_path": filepath}
