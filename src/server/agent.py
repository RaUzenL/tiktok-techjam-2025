from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, Literal, List
import os
import re
from datetime import datetime, timezone
from dateutil import tz

import matplotlib.pyplot as plt
import networkx as nx
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from huggingface_hub import InferenceClient

# =========================
# Config
# =========================
# Choose one:
#   "google/gemma-3-12b-it"   or   "Qwen/Qwen3-8B"
MODEL_ID = os.getenv("REVIEW_MODEL_ID", "google/gemma-3-12b-it")
HF_TOKEN = os.getenv("HF_TOKEN")
TEMPERATURE = 0.0
MAX_TOKENS = 256   # Structured JSON; keep it lean.
TIMEOUT = 120

# =========================
# State
# =========================
class ReviewState(TypedDict, total=False):
    review: Dict[str, Any]
    features: Dict[str, Any]
    rule_decision: Optional[Literal["relevant", "not_relevant"]]
    rule_reason: Optional[str]
    llm_vote: Optional[Dict[str, Any]]
    final_decision: Optional[Literal["relevant", "not_relevant"]]
    explanation: Optional[str]
    confidence: Optional[float]

# =========================
# Regex / heuristics
# =========================
URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
PROMO_TERMS = [
    "sale", "discount", "% off", "percent off", "limited time", "limited offer",
    "promo code", "use code", "coupon", "deal", "offer", "book now", "call now",
    "subscribe", "follow us", "visit our website", "click here", "official website",
    "free trial", "sign up", "order now", "link in bio"
]
PROMO_RES = [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in PROMO_TERMS]

VISIT_MARKERS = [
    "i ", " we ", " my ", " our ", "me ", "us ", "i'm", "i’ve", "i was", "we were",
    "ordered", "ate", "drank", "menu", "dish", "coffee", "table", "queue", "waited",
    "staff", "service", "room", "counter", "cashier", "bill", "receipt", "ticket",
    "entrance", "parking", "restroom", "toilet", "seating", "reservation", "check in",
    "checked in", "checkout", "check-out", "walked", "sat", "stood", "line"
]
VISIT_RES = [re.compile(r"\b" + term.strip() + r"\b", re.IGNORECASE) for term in VISIT_MARKERS]

IRRELEVANT_HINTS = [
    "hiring", "vacancy", "job opening", "lost phone", "lost wallet", "crypto",
    "bitcoin", "forex", "giveaway", "telegram", "whatsapp", "contact me",
]
IRRELEVANT_RES = [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in IRRELEVANT_HINTS]

def count_matches(res_list: List[re.Pattern], text: str) -> int:
    return sum(1 for r in res_list if r.search(text))

def extract_pics(review: Dict[str, Any]) -> List[str]:
    pics = review.get("pics") or []
    urls = []
    for p in pics:
        u = p.get("url")
        if isinstance(u, list):
            urls.extend([str(x) for x in u])
        elif isinstance(u, str):
            urls.append(u)
    return urls

def ms_to_date(ms: Optional[int]) -> Optional[str]:
    if ms is None: return None
    try:
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).astimezone(tz.tzlocal())
        return dt.isoformat()
    except Exception:
        return None

# =========================
# Node 1: feature extraction
# =========================
def extract_features(state: ReviewState) -> ReviewState:
    review = state["review"]
    text = (review.get("text") or "").strip()
    rating = review.get("rating")
    time_ms = review.get("time")

    has_url = bool(URL_RE.search(text))
    url_count = len(URL_RE.findall(text))
    promo_hits = [t for t, r in zip(PROMO_TERMS, PROMO_RES) if r.search(text)]
    irrelevant_hits = [t for t, r in zip(IRRELEVANT_HINTS, IRRELEVANT_RES) if r.search(text)]

    pics_urls = extract_pics(review)
    has_pics = len(pics_urls) > 0
    visit_markers_count = count_matches(VISIT_RES, " " + text + " ")
    likely_visited = has_pics or (visit_markers_count >= 2)

    state["features"] = {
        "text_len": len(text),
        "has_url": has_url,
        "url_count": url_count,
        "promo_keywords_found": promo_hits,
        "irrelevant_hints_found": irrelevant_hits,
        "has_pics": has_pics,
        "pics_count": len(pics_urls),
        "visit_markers_count": visit_markers_count,
        "likely_visited": likely_visited,
        "rating": rating,
        "date_iso": ms_to_date(time_ms),
    }
    return state

# =========================
# Node 2: rules (fast negatives for ads)
# =========================
def rule_filter(state: ReviewState) -> ReviewState:
    f = state["features"]
    if f["has_url"] or f["promo_keywords_found"]:
        state["rule_decision"] = "not_relevant"
        reasons = []
        if f["has_url"]:
            reasons.append("contains web link")
        if f["promo_keywords_found"]:
            reasons.append("promotional terms: " + ", ".join(f["promo_keywords_found"]))
        state["rule_reason"] = "Likely advertisement (" + "; ".join(reasons) + ")."
        return state
    state["rule_decision"] = None
    state["rule_reason"] = None
    return state

def rule_filter_next(state: ReviewState) -> Literal["early_exit", "continue"]:
    return "early_exit" if state.get("rule_decision") else "continue"

# =========================
# Node 3: heuristics (fast positives)
# =========================
def heuristics_positive(state: ReviewState) -> ReviewState:
    f = state["features"]
    if f["has_pics"] and not f["has_url"] and not f["promo_keywords_found"]:
        state["rule_decision"] = "relevant"
        state["rule_reason"] = "Has photos and no ad signals."
    else:
        state["rule_decision"] = None
        state["rule_reason"] = None
    return state

def heuristics_next(state: ReviewState) -> Literal["early_exit", "continue"]:
    return "early_exit" if state.get("rule_decision") else "continue"

# =========================
# Node 4: LLM judge (HF InferenceClient with JSON schema)
# =========================
class PolicyJudgement(BaseModel):
    advertisement: bool = Field(..., description="True if promotional or contains links/ads.")
    irrelevant: bool = Field(..., description="True if content is not about the location.")
    rant_without_visit: bool = Field(..., description="True if it's a rant/complaint without evidence of visiting.")
    visited: Literal["yes","probably","unclear","no"] = Field(..., description="Did they likely visit?")
    relevant: bool = Field(..., description="Final judgement: passes all three policies.")
    reasoning: str = Field(..., description="Brief reason.")

LLM_SYSTEM = (
    "You are a precise content-moderation judge for Google location reviews.\n"
    "Policies:\n"
    "1) No Advertisement: reject if promotional or contains links.\n"
    "2) No Irrelevant Content: reject if not about the specific location.\n"
    "3) No Rant Without Visit: complaints require evidence of an actual visit (textual cues or photos).\n"
    "Return a strict JSON object following the provided schema."
)

def _hf_client() -> InferenceClient:
    # InferenceClient exposes OpenAI-compatible chat.completions.
    # It also supports JSON mode / structured outputs via response_format.  (Docs)  # :contentReference[oaicite:3]{index=3}
    return InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=TIMEOUT)

def llm_judge(state: ReviewState) -> ReviewState:
    review = state["review"]
    f = state["features"]

    client = _hf_client()
    # Build schema for structured outputs
    schema = PolicyJudgement.model_json_schema()

    # Compose messages; OpenAI-compatible format is supported in InferenceClient. :contentReference[oaicite:4]{index=4}
    messages = [
        {"role": "system", "content": LLM_SYSTEM},
        {
            "role": "user",
            "content": (
                "Apply the policies exactly. Consider photos as strong evidence of visit.\n\n"
                f"review_record: {review}\n\nextracted_features: {f}"
            ),
        },
    ]

    out = client.chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        # Enforce valid, schema-conformant JSON (Structured Outputs).
        response_format={"type": "json_schema", "json_schema": {"name": "PolicyJudgement", "schema": schema}},
        # Tip: you can pass model/provider-specific flags via extra_body if needed. :contentReference[oaicite:5]{index=5}
    )

    content = out.choices[0].message.content
    # `content` is guaranteed valid JSON matching the schema if the provider honors JSON mode.
    import json
    state["llm_vote"] = json.loads(content)
    return state

# =========================
# Node 5: aggregate
# =========================
def aggregate(state: ReviewState) -> ReviewState:
    if state.get("rule_decision"):
        state["final_decision"] = state["rule_decision"]
        state["explanation"] = state.get("rule_reason") or "Early decision by rules."
        state["confidence"] = 0.9 if state["rule_decision"] == "not_relevant" else 0.8
        return state

    vote = state.get("llm_vote") or {}
    relevant = bool(vote.get("relevant", False))
    state["final_decision"] = "relevant" if relevant else "not_relevant"

    penalties = sum([
        1 if vote.get("advertisement") else 0,
        1 if vote.get("irrelevant") else 0,
        1 if vote.get("rant_without_visit") else 0
    ])
    conf = max(0.5, 0.9 - 0.1 * penalties)
    if vote.get("visited") in ("yes", "probably"):
        conf += 0.05
    state["confidence"] = round(min(conf, 0.95), 2)

    state["explanation"] = vote.get("reasoning", "Combined decision.")
    return state

# =========================
# Build graph
# =========================
def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("extract_features", extract_features)
    g.add_node("rule_filter", rule_filter)
    g.add_node("heuristics_positive", heuristics_positive)
    g.add_node("llm_judge", llm_judge)
    g.add_node("aggregate", aggregate)

    g.add_edge(START, "extract_features")
    g.add_edge("extract_features", "rule_filter")
    g.add_conditional_edges("rule_filter", rule_filter_next,
                            {"early_exit": "aggregate", "continue": "heuristics_positive"})
    g.add_conditional_edges("heuristics_positive", heuristics_next,
                            {"early_exit": "aggregate", "continue": "llm_judge"})
    g.add_edge("llm_judge", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()

def to_networkx(compiled_app) -> nx.DiGraph:
    g = compiled_app.get_graph()
    try:
        g_nx = g  # may already be a networkx graph
        _ = list(g_nx.nodes)  # sanity check
        return g_nx
    except Exception:
        # convert
        out = nx.DiGraph()
        nodes = [getattr(n, "id", str(n)) for n in getattr(g, "nodes", [])]
        out.add_nodes_from(nodes)
        for e in getattr(g, "edges", []):
            s = getattr(e, "source", None)
            t = getattr(e, "target", None)
            sid = getattr(s, "id", s)
            tid = getattr(t, "id", t)
            if sid is not None and tid is not None:
                out.add_edge(sid, tid)
        return out

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    example = {
        "user_id": "112641626927833880743",
        "name": "Little Man",
        "time": 1533121309821,
        "rating": 3,
        "text": "Has nice food choices.",
        "pics": [
            {"url": ["https://lh5.googleusercontent.com/p/AF1QipMDSa1pSffRzM1AqS0phG3a_K2eSssz-vRY-cPf=w150-h150-k-no-p"]},
            {"url": ["https://lh5.googleusercontent.com/p/AF1QipPgm5LcNt7zGDDb24vS8ST5Oe5SoWkjg6bn7vXN=w150-h150-k-no-p"]}
        ],
        "resp": None,
        "gmap_id": "0x89c2605ade02a307:0x798d440705b8d9b3"
    }

    app = build_graph()

    G = to_networkx(app)

    plt.figure(figsize=(8, 6))
    # kamada_kawai gives stable DAG-ish layout without requiring iterability tricks
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, font_size=9, arrows=True)
    plt.tight_layout()
    plt.show()

    out = app.invoke({"review": example})
    print({
        "model": MODEL_ID,
        "final_decision": out["final_decision"],
        "explanation": out["explanation"],
        "confidence": out["confidence"],
        "features": out["features"],
        "llm_vote": out.get("llm_vote"),
    })
