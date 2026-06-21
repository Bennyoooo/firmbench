"""
FirmBench — scoring layer (judge-as-translator).

Converts NL artifacts (ad copy, feature specs) into sim parameters:
  - score_ad_copy  → {craft: float, target_pains: set[int]}
  - score_feature_spec → {feature_id: int, quality: float}

Targeting extraction is rule-based (keyword matching, zero cost).
Craft/quality scoring uses a cheap LLM (configurable, default Fireworks).
fast_mode=True skips all LLM calls and returns defaults (craft=1.0, quality=1.0).
"""

import os
import re
import logging
from dataclasses import dataclass

log = logging.getLogger("firmbench.scorer")


@dataclass
class ScoreConfig:
    fast_mode: bool = False
    model: str = ""
    api_key: str = ""
    base_url: str = "https://api.fireworks.ai/inference/v1"

    def __post_init__(self):
        if not self.model:
            self.model = os.environ.get(
                "FIREWORKS_MODEL",
                "accounts/fireworks/models/llama-v3p2-3b-instruct")
        if not self.api_key:
            self.api_key = os.environ.get("FIREWORKS_API_KEY", "")


# ----------------------------- targeting (rule-based) ----------------

def extract_target_pains(ad_copy: str, pain_keywords: dict) -> set:
    """Match ad copy against pain keyword dictionaries. Zero cost, deterministic."""
    if not ad_copy:
        return set()
    lowered = ad_copy.lower()
    matched = set()
    for pain_id, keywords in pain_keywords.items():
        if any(kw.lower() in lowered for kw in keywords):
            matched.add(pain_id)
    return matched


def identify_feature(spec_text: str, feature_keywords: dict) -> int:
    """Match spec text against feature keyword dictionaries. Returns best-match ID."""
    if not spec_text:
        return 0
    lowered = spec_text.lower()
    best_id = 0
    best_count = 0
    for feat_id, keywords in feature_keywords.items():
        count = sum(1 for kw in keywords if kw.lower() in lowered)
        if count > best_count:
            best_count = count
            best_id = feat_id
    return best_id


# ----------------------------- craft/quality scoring (LLM) ----------

_CRAFT_PROMPT = """Rate this advertising copy on a scale from 0.0 to 1.0.
Score based on: clarity (is the message clear?), persuasiveness (would it convince
someone to try the product?), and specificity (does it name concrete benefits?).

Ad copy:
\"\"\"
{ad_copy}
\"\"\"

Reply with ONLY a single decimal number between 0.0 and 1.0. Nothing else."""

_QUALITY_PROMPT = """Rate this product feature specification on a scale from 0.0 to 1.0.
Score based on: completeness (does it describe what the feature does?), coherence
(is it logically organized?), and usefulness (would a developer know what to build?).

Feature spec:
\"\"\"
{spec_text}
\"\"\"

Reply with ONLY a single decimal number between 0.0 and 1.0. Nothing else."""


def _parse_score(text: str, default: float = 0.5) -> float:
    """Extract a float from LLM response. Robust to chatty models."""
    if not text:
        return default
    matches = re.findall(r'(\d+\.?\d*)', text.strip())
    if matches:
        val = float(matches[0])
        return max(0.0, min(1.0, val))
    return default


def _llm_score(prompt: str, cfg: ScoreConfig, default: float = 0.5) -> float:
    """Call a cheap LLM for a 0-1 score. Returns default on any failure."""
    if not cfg.api_key:
        log.warning("No API key for scoring; returning default %.1f", default)
        return default
    try:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10,
        )
        text = resp.choices[0].message.content or ""
        return _parse_score(text, default)
    except Exception as e:
        log.warning("LLM scoring failed (%s); returning default %.1f", e, default)
        return default


# ----------------------------- public API ---------------------------

def score_ad_copy(ad_copy: str, world, cfg: ScoreConfig = None) -> dict:
    """Score ad copy → {craft: float, target_pains: set[int]}.

    craft: 0-1 quality score (LLM-based, or 1.0 in fast_mode).
    target_pains: set of pain IDs the ad addresses (keyword-based, always runs).
    """
    cfg = cfg or ScoreConfig(fast_mode=True)
    target_pains = extract_target_pains(ad_copy, world.pain_keywords or {})

    if cfg.fast_mode:
        craft = 1.0
    else:
        prompt = _CRAFT_PROMPT.format(ad_copy=ad_copy[:2000])
        raw = _llm_score(prompt, cfg, default=0.5)
        # stretch from [0,1] to [0.3, 1.0] so bad copy still has some effect
        craft = 0.3 + 0.7 * raw

    return {"craft": round(craft, 3), "target_pains": target_pains}


def score_feature_spec(spec_text: str, world, cfg: ScoreConfig = None) -> dict:
    """Score feature spec → {feature_id: int, quality: float}.

    feature_id: best-match feature ID (keyword-based, always runs).
    quality: 0-1 quality score (LLM-based, or 1.0 in fast_mode).
    """
    cfg = cfg or ScoreConfig(fast_mode=True)
    feature_id = identify_feature(spec_text, world.feature_keywords or {})

    if cfg.fast_mode:
        quality = 1.0
    else:
        prompt = _QUALITY_PROMPT.format(spec_text=spec_text[:2000])
        raw = _llm_score(prompt, cfg, default=0.5)
        quality = 0.3 + 0.7 * raw

    return {"feature_id": feature_id, "quality": round(quality, 3)}
