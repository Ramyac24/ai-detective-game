"""
llm_engine.py
Ollama-based local LLM integration for the alien signal mystery game.

Responsibilities
----------------
- Clue analysis: interpret raw signal metadata fragments as narrative clues
- Theory evaluation: judge the player's deduction against real evidence
- Dynamic narrative: generate mission briefings, hints, and story beats
- Signal report: produce a human-readable summary of ML findings
"""

import ollama
import json
import re
from typing import Optional

# Default model — user can override via .env or the game settings page
DEFAULT_MODEL = "llama3.2"

# Fallback response when Ollama is unavailable
_FALLBACK_PREFIX = "[OFFLINE MODE] Ollama not running. "


# ══════════════════════════════════════════════════════════════════════════════
#  CORE LLM WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def _chat(system_prompt: str, user_prompt: str,
          model: str = DEFAULT_MODEL,
          temperature: float = 0.7,
          max_tokens: int = 512) -> str:
    """
    Single-turn chat with the local Ollama model.
    Returns the response text or a fallback string on failure.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return (
            f"{_FALLBACK_PREFIX}"
            f"Start Ollama with `ollama serve` and pull a model with "
            f"`ollama pull llama3.2`.\n\nError: {e}"
        )


def check_ollama_available(model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    """
    Returns (is_available: bool, message: str).
    Tries a tiny ping request to Ollama.
    """
    try:
        models = ollama.list()
        names = [m.get("name", "") for m in models.get("models", [])]
        if not names:
            return False, "Ollama is running but no models are pulled. Run: `ollama pull llama3.2`"
        if not any(model in n for n in names):
            return False, (
                f"Model '{model}' not found. "
                f"Available: {', '.join(names)}. "
                f"Run: `ollama pull {model}`"
            )
        return True, f"Ollama ready — using model: {model}"
    except Exception as e:
        return False, f"Ollama not reachable. Start it with: `ollama serve`\n({e})"


# ══════════════════════════════════════════════════════════════════════════════
#  GAME FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PERSONA = """
You are ARIA (Alien Research Intelligence Assistant), the AI co-pilot of a deep-space
signal monitoring station in the year 2031. Your job is to help Signal Analysts decode
mysterious transmissions, reason about alien communication patterns, and solve the
mystery of the coordinated anomaly signals detected from Sector 7-Delta.

Keep responses concise (2-4 sentences unless told otherwise), dramatic but scientific,
and always in-character as a station AI. Use terms like "signal burst", "sector grid",
"transmission pattern", "cryptographic anomaly" naturally.
"""


def analyze_clue(clue_fragment: str,
                 signal_metadata: dict,
                 model: str = DEFAULT_MODEL) -> str:
    """
    Interpret a raw signal clue fragment in the context of its signal metadata.
    Returns a narrative analysis string.
    """
    meta_str = "\n".join(f"  {k}: {v}" for k, v in signal_metadata.items()
                         if k != "clue_fragment")
    prompt = f"""
A signal was intercepted with the following metadata:
{meta_str}

Embedded clue fragment recovered from the signal header:
"{clue_fragment}"

Analyze this clue. What does it suggest about the origin of this signal?
Is this an ordinary transmission or something more significant?
Provide a 2-3 sentence intelligence report.
"""
    return _chat(SYSTEM_PERSONA, prompt, model=model, temperature=0.65)


def evaluate_theory(player_theory: str,
                    evidence_summary: dict,
                    model: str = DEFAULT_MODEL) -> dict:
    """
    Evaluate the player's final theory against the collected evidence.

    Parameters
    ----------
    player_theory : str
        The player's written deduction.
    evidence_summary : dict
        Keys: anomaly_count, mystery_score, cluster_coords,
              clues_collected, classification_accuracy

    Returns
    -------
    dict with keys: verdict (str), score (0-100), explanation (str), reveal (str)
    """
    ev = evidence_summary
    evidence_text = f"""
Evidence on file:
- Anomaly signals detected: {ev.get('anomaly_count', '?')}
- Anomaly concentration score: {ev.get('mystery_score', '?')} / 1.0
- Suspected cluster coordinates: {ev.get('cluster_coords', 'unknown')}
- Clues decoded: {ev.get('clues_collected', 0)}
- Classifier accuracy: {ev.get('classification_accuracy', '?')}
"""
    prompt = f"""
{evidence_text}

The analyst has submitted this theory:
"{player_theory}"

Evaluate the theory. Respond ONLY with valid JSON in this exact format:
{{
  "verdict": "CORRECT" | "PARTIALLY CORRECT" | "INCORRECT",
  "score": <integer 0-100>,
  "explanation": "<2-3 sentences evaluating the theory>",
  "reveal": "<1-2 sentences revealing what was actually happening with the anomaly signals>"
}}
"""
    raw = _chat(SYSTEM_PERSONA, prompt, model=model, temperature=0.3, max_tokens=400)

    # Parse JSON safely
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback if JSON parse fails
    return {
        "verdict":     "PARTIALLY CORRECT",
        "score":       55,
        "explanation": raw[:300] if raw else "Unable to evaluate theory at this time.",
        "reveal":      "The anomaly signals appear to be a coordinated beacon from an automated probe originating beyond the outer heliosphere.",
    }


def generate_mission_briefing(mission_name: str,
                               context: dict,
                               model: str = DEFAULT_MODEL) -> str:
    """
    Generate a dramatic mission briefing for each game phase.

    Parameters
    ----------
    mission_name : str
        One of: 'signal_lab', 'classify', 'cluster', 'clue_decoder', 'final_deduction'
    context : dict
        Relevant stats/info to weave into the briefing.
    """
    briefing_hints = {
        "signal_lab":       "Introduce the analyst to the signal database. Mention {n_signals} intercepted signals. Set an ominous tone.",
        "classify":         "Brief the analyst on using ML classification. {accuracy}% accuracy expected. Stress that Anomaly signals are rare but critical.",
        "cluster":          "Explain that clustering the signals by origin sector may reveal a hidden pattern. Hint at a tight cluster in sector {sector}.",
        "clue_decoder":     "The analyst has found {n_clues} embedded clue fragments. Urge analysis with AI assistance.",
        "final_deduction":  "The analyst has gathered all evidence. Mystery score: {mystery_score}. Prompt them to write their final theory.",
    }

    hint = briefing_hints.get(mission_name, "Brief the analyst on their current mission.")
    for k, v in context.items():
        hint = hint.replace("{" + k + "}", str(v))

    prompt = f"""
Generate a mission briefing for mission: "{mission_name}".
Instructions: {hint}
Keep it to 3 sentences. Be dramatic and scientific.
"""
    return _chat(SYSTEM_PERSONA, prompt, model=model, temperature=0.75)


def summarize_ml_results(metrics: dict, model: str = DEFAULT_MODEL) -> str:
    """
    Produce a plain-English narrative summary of the ML classification results.
    """
    acc  = metrics.get("accuracy", 0)
    f1   = metrics.get("f1_weighted", 0)
    cv   = metrics.get("cv_mean", 0)
    top_feature = max(
        metrics.get("feature_importance", {}).items(),
        key=lambda x: x[1],
        default=("unknown", 0)
    )

    prompt = f"""
Summarize these ML classification results in 2 sentences for a signal analyst:
- Accuracy: {acc:.1%}
- F1 Score (weighted): {f1:.4f}
- Cross-validation mean: {cv:.1%}
- Most informative feature: {top_feature[0]} (importance: {top_feature[1]:.3f})

Make it sound like a technical intelligence report. Mention what the accuracy implies
about our ability to detect Anomaly signals.
"""
    return _chat(SYSTEM_PERSONA, prompt, model=model, temperature=0.5)


def get_hint(current_mission: str,
             clues_found: list,
             model: str = DEFAULT_MODEL) -> str:
    """
    Return a contextual hint for the player without giving away the answer.
    """
    clue_str = "; ".join(clues_found[-3:]) if clues_found else "none yet"
    prompt = f"""
The analyst is stuck on mission: "{current_mission}".
Recent clues decoded: {clue_str}

Give a subtle, 1-2 sentence hint that nudges them in the right direction
without revealing the final answer. Be cryptic but helpful.
"""
    return _chat(SYSTEM_PERSONA, prompt, model=model, temperature=0.8)
