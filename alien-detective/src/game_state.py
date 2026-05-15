"""
game_state.py
Manages all Streamlit session state for the alien signal mystery game.

Centralises initialisation, progress tracking, clue collection,
evidence logging, and mission gating so app.py stays clean.
"""

import streamlit as st
from datetime import datetime


# ── Mission order ──────────────────────────────────────────────────────────────
MISSIONS = [
    "intro",
    "signal_lab",
    "classify",
    "cluster",
    "clue_decoder",
    "case_board",
    "final_deduction",
    "results",
]

MISSION_TITLES = {
    "intro":            "Welcome, Analyst",
    "signal_lab":       "Mission 1 · Signal Lab",
    "classify":         "Mission 2 · Signal Classifier",
    "cluster":          "Mission 3 · Origin Mapping",
    "clue_decoder":     "Mission 4 · Clue Decoder",
    "case_board":       "Mission 5 · Case Board",
    "final_deduction":  "Mission 6 · Final Deduction",
    "results":          "Case Closed",
}

MISSION_ICONS = {
    "intro":            "🛸",
    "signal_lab":       "📡",
    "classify":         "🤖",
    "cluster":          "🗺️",
    "clue_decoder":     "🔍",
    "case_board":       "📋",
    "final_deduction":  "💡",
    "results":          "🏆",
}

# ── Keys ───────────────────────────────────────────────────────────────────────
_KEYS = {
    "player_name":         ("player_name",         ""),
    "current_mission":     ("current_mission",      "intro"),
    "missions_completed":  ("missions_completed",   []),
    "dataset":             ("dataset",              None),
    "df_classified":       ("df_classified",        None),
    "df_clustered":        ("df_clustered",         None),
    "classifier_metrics":  ("classifier_metrics",   None),
    "mystery_score":       ("mystery_score",        0.0),
    "clues_found":         ("clues_found",          []),
    "evidence_log":        ("evidence_log",         []),
    "llm_model":           ("llm_model",            "llama3.2"),
    "ollama_available":    ("ollama_available",      None),  # None = unchecked
    "score":               ("score",                0),
    "start_time":          ("start_time",           None),
    "theory_submitted":    ("theory_submitted",     ""),
    "theory_result":       ("theory_result",        None),
    "briefings":           ("briefings",            {}),
    "hints_used":          ("hints_used",           0),
}


def init_state():
    """Initialise all session state keys with defaults (idempotent)."""
    for key, (skey, default) in _KEYS.items():
        if skey not in st.session_state:
            st.session_state[skey] = default

    if st.session_state.get("start_time") is None and \
       st.session_state.get("current_mission") != "intro":
        st.session_state["start_time"] = datetime.now()


# ── Getters ────────────────────────────────────────────────────────────────────

def get(key: str):
    return st.session_state.get(key)


def set(key: str, value):
    st.session_state[key] = value


# ── Mission management ─────────────────────────────────────────────────────────

def complete_mission(mission: str):
    """Mark a mission as done and advance to the next."""
    completed = st.session_state.get("missions_completed", [])
    if mission not in completed:
        completed.append(mission)
        st.session_state["missions_completed"] = completed
        add_evidence(f"Mission completed: {MISSION_TITLES.get(mission, mission)}")

    # Advance to next mission
    idx = MISSIONS.index(mission) if mission in MISSIONS else -1
    if idx >= 0 and idx + 1 < len(MISSIONS):
        st.session_state["current_mission"] = MISSIONS[idx + 1]


def advance_to(mission: str):
    """Directly navigate to a specific mission (for sidebar nav)."""
    st.session_state["current_mission"] = mission


def mission_is_unlocked(mission: str) -> bool:
    """A mission is unlocked if the previous one is completed."""
    idx = MISSIONS.index(mission) if mission in MISSIONS else -1
    if idx <= 1:          # intro and signal_lab always unlocked
        return True
    prev = MISSIONS[idx - 1]
    return prev in st.session_state.get("missions_completed", [])


def progress_pct() -> float:
    """Return 0-100 completion percentage."""
    completed = st.session_state.get("missions_completed", [])
    total = len(MISSIONS) - 1   # exclude 'results'
    return round(100 * len(completed) / total, 1)


# ── Clue management ────────────────────────────────────────────────────────────

def add_clue(fragment: str, analysis: str = ""):
    """Add a decoded clue to the player's collection."""
    clues = st.session_state.get("clues_found", [])
    entry = {
        "fragment":  fragment,
        "analysis":  analysis,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    if not any(c["fragment"] == fragment for c in clues):
        clues.append(entry)
        st.session_state["clues_found"] = clues
        award_points(15, f"New clue decoded: {fragment[:40]}…")
        add_evidence(f"Clue fragment decoded: {fragment[:60]}")


def get_clues() -> list:
    return st.session_state.get("clues_found", [])


# ── Evidence log ───────────────────────────────────────────────────────────────

def add_evidence(event: str):
    log = st.session_state.get("evidence_log", [])
    log.append({
        "time":  datetime.now().strftime("%H:%M:%S"),
        "event": event,
    })
    st.session_state["evidence_log"] = log


def get_evidence_log() -> list:
    return st.session_state.get("evidence_log", [])


# ── Scoring ────────────────────────────────────────────────────────────────────

def award_points(pts: int, reason: str = ""):
    current = st.session_state.get("score", 0)
    st.session_state["score"] = current + pts
    if reason:
        add_evidence(f"+{pts} pts — {reason}")


def deduct_hint_penalty():
    """Using a hint costs 10 points."""
    hints = st.session_state.get("hints_used", 0) + 1
    st.session_state["hints_used"] = hints
    award_points(-10, "Hint used")


# ── Briefing cache ─────────────────────────────────────────────────────────────

def cache_briefing(mission: str, text: str):
    briefings = st.session_state.get("briefings", {})
    briefings[mission] = text
    st.session_state["briefings"] = briefings


def get_briefing(mission: str) -> str:
    return st.session_state.get("briefings", {}).get(mission, "")


# ── Summary for theory evaluation ─────────────────────────────────────────────

def build_evidence_summary() -> dict:
    df_c = st.session_state.get("df_clustered")
    anomaly_count = 0
    cluster_coords = "unknown"

    if df_c is not None:
        anomaly_count = int((df_c.get("signal_class", df_c.get("predicted_class", [])) == "Anomaly").sum())
        # Try to find anomaly cluster centroid
        if "kmeans_cluster" in df_c.columns:
            anomaly_rows = df_c[df_c.get("signal_class", df_c.get("predicted_class", [])) == "Anomaly"]
            if not anomaly_rows.empty:
                cx = anomaly_rows["sector_x"].mean()
                cy = anomaly_rows["sector_y"].mean()
                cluster_coords = f"({cx:.1f}, {cy:.1f})"

    metrics = st.session_state.get("classifier_metrics") or {}
    return {
        "anomaly_count":          anomaly_count,
        "mystery_score":          st.session_state.get("mystery_score", 0.0),
        "cluster_coords":         cluster_coords,
        "clues_collected":        len(st.session_state.get("clues_found", [])),
        "classification_accuracy": metrics.get("accuracy", "N/A"),
    }


# ── Elapsed time ───────────────────────────────────────────────────────────────

def elapsed_time() -> str:
    start = st.session_state.get("start_time")
    if start is None:
        return "0m 0s"
    delta = datetime.now() - start
    mins, secs = divmod(int(delta.total_seconds()), 60)
    return f"{mins}m {secs}s"


# ── Reset ──────────────────────────────────────────────────────────────────────

def reset_game():
    """Clear all game state to restart from the beginning."""
    keys_to_clear = [
        "player_name", "current_mission", "missions_completed",
        "dataset", "df_classified", "df_clustered", "classifier_metrics",
        "mystery_score", "clues_found", "evidence_log", "score",
        "start_time", "theory_submitted", "theory_result", "briefings",
        "hints_used",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    init_state()
