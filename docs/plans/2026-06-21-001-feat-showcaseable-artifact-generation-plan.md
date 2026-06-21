# feat: Showcaseable artifact generation for FirmBench campaigns and features

**Status:** active
**Created:** 2026-06-21
**Origin:** User request — make the simulation visually compelling by having agents produce real artifacts (ad copy with visuals, mini webpages for features) instead of just passing numeric IDs.

---

## Summary

Upgrade FirmBench's action space so agents produce **real artifacts** — marketing ad cards with copy and visuals for campaigns, and mini HTML product pages for built features. Artifacts feed the sim via the existing judge-as-translator pattern (`craft` and `implementation_quality` multipliers) and are saved to disk for a replay viewer that makes the demo visually compelling. Total marginal cost: ~$2–5 per 5,000 episodes.

---

## Problem Frame

FirmBench currently works but looks like a spreadsheet — agents pass `target_pains: [0, 3]` and `build: 5`, and the output is JSON numbers. For the hackathon demo, judges need to *see* the business running: ad creatives, product pages, campaign performance dashboards. The artifacts also serve a technical purpose: the `craft` and `implementation_quality` multipliers in the funnel (currently hardcoded to 1.0) become meaningful, adding a genuine skill dimension to the environment.

---

## Requirements

- R1. Agents write **ad copy text** for each campaign (headline, body, CTA)
- R2. Ad copy is rendered into a **visual ad card** (HTML template, saved to disk)
- R3. Agents write **feature page HTML/content** when building a feature
- R4. Feature pages are saved as **renderable HTML files**
- R5. A **craft score** (0–1) is computed from ad copy quality and multiplies `p_try` in the funnel
- R6. An **implementation quality** score (0–1) is computed from feature spec quality and stored in `built[f]`
- R7. **Targeting extraction** from ad copy determines which pains the ad addresses (keyword-based, not LLM)
- R8. All artifacts are **stored per-episode/per-round** on disk for replay
- R9. A **replay viewer** (static HTML) lets humans step through an episode and see the artifacts alongside business metrics
- R10. The existing structured-action path (no NL, `craft=1.0`) remains as a **fast mode** for training

---

## Scope Boundaries

### In scope
- Ad copy generation (agent writes text, we render cards)
- Feature page generation (agent writes HTML/structured content, we save it)
- Scoring layer (cheap LLM for craft, rule-based for targeting)
- Artifact storage (filesystem, JSON + HTML)
- Replay viewer (static HTML, one file)
- Updated MCP tool interfaces (backward-compatible optional params)

### Deferred to follow-up work
- AI-generated images via Flux/DALL-E (use HTML/CSS cards instead — zero cost)
- Playwright screenshots (save HTML now, screenshot in a future pass)
- Video replay / animated walkthrough
- Multi-agent artifact collaboration

### Out of scope
- Real ad platform integration
- Production-quality web design
- Image generation during the RL training loop

---

## Key Technical Decisions

**KTD1. HTML/CSS ad cards, not AI-generated images.**
Rationale: AI image generation costs $0.001–0.005/image and adds 1–2s latency per call — prohibitive in an RL training loop. HTML/CSS cards rendered via Jinja2 are free, instant, deterministic, and still look good in the replay viewer. A post-hoc Flux pass on winning episodes is a future stretch.

**KTD2. Rule-based targeting extraction, LLM only for craft scoring.**
Rationale: Targeting (which pains does the ad address?) is extractable via keyword matching from the world's pain descriptions — zero cost, zero latency, deterministic. Only `craft` (ad quality, 0–1) uses an LLM, and the cheapest viable option (Llama 3.2 3B on Fireworks at $0.10/1M tokens) costs ~$2 per 100K calls. Total NL layer cost: ~$2–5 per 5K episodes.

**KTD3. Optional NL parameters — backward-compatible tools.**
Rationale: The MCP tools gain optional `ad_copy` and `spec_html` parameters. When omitted, the structured phase runs as before (`craft=1.0`, `quality=1.0`). When provided, the translator scores them. This keeps `python3 sim.py` and `python3 run.py` working without any LLM dependency.

**KTD4. Artifacts stored as flat files, not a database.**
Rationale: No infrastructure to manage. One directory per episode seed, subdirectories per round, JSON for metadata + HTML for artifacts. A `manifest.json` enables the replay viewer. Matches the existing `rft_out/` pattern.

---

## Implementation Units

### U1. Scoring layer (`scorer.py`)

**Goal:** A standalone module that converts NL artifacts to sim parameters.

**Requirements:** R5, R6, R7

**Dependencies:** None

**Files:**
- `scorer.py` (new)
- `tests/test_scorer.py` (new)

**Approach:**
- `score_ad_copy(text, world) -> {craft: float, target_pains: set[int]}`:
  - **Targeting extraction** (rule-based): `generate_world` already produces pain descriptions implicitly via the population. Add a `pain_keywords: dict[int, list[str]]` field to `World` (generated per seed). Match ad copy tokens against keywords. `target_pains = {p for p in pains if any(kw in ad_copy.lower() for kw in pain_keywords[p])}`.
  - **Craft scoring** (LLM): Call a cheap model (configurable, default Fireworks Llama 3.2 3B) with a fixed rubric prompt: "Rate this ad copy 0.0–1.0 on clarity, persuasiveness, and specificity. Reply with just the number." Parse the float. On failure, default to 0.5.
- `score_feature_spec(text, world) -> {feature_id: int, quality: float}`:
  - **Feature identification** (keyword-based): match spec text against feature descriptions (also generated per seed in `World`).
  - **Quality scoring** (LLM): same cheap model, "Rate this product spec 0.0–1.0 on completeness and coherence."
- `ScoreConfig` dataclass: model name, API key, base URL, fast_mode flag (skip LLM, return defaults).
- All scoring is sync (blocking). Async wrapper optional.

**Patterns to follow:** Same style as `sim.py` — dataclasses, pure functions, no framework deps. The LLM call pattern follows `agent.py`'s `FireworksAgent` (OpenAI client, lazy import).

**Test scenarios:**
- Craft scoring returns float in [0, 1] for reasonable ad copy
- Craft scoring defaults to 0.5 on LLM failure / timeout
- Targeting extraction finds correct pains from keyword-laden copy
- Targeting extraction returns empty set for gibberish copy
- Feature identification picks the closest feature from spec text
- Quality scoring returns float in [0, 1] for reasonable spec
- fast_mode=True skips LLM calls and returns (craft=1.0, quality=1.0)

---

### U2. World metadata for keyword matching

**Goal:** Add pain/feature descriptions and keyword dictionaries to `World` so the scoring layer can do rule-based extraction.

**Requirements:** R7

**Dependencies:** None (can run in parallel with U1)

**Files:**
- `sim.py` (modify `World` dataclass, `generate_world`)

**Approach:**
- Add to `World`: `pain_names: list[str]`, `pain_keywords: dict[int, list[str]]`, `feature_names: list[str]`, `feature_keywords: dict[int, list[str]]`.
- In `generate_world`: use the RNG to sample from pools of realistic pain/feature names (e.g., "slow onboarding", "billing errors", "missing integrations" for pains; "quick-start wizard", "payment dashboard", "API connector" for features). Generate 3–5 keywords per pain/feature from the name.
- These are cosmetic — the sim mechanics are unchanged. But they give the NL layer something meaningful to match against, and they make the replay viewer more readable.

**Patterns to follow:** Same seeded RNG pattern as the existing demography generation.

**Test scenarios:**
- `generate_world` produces `pain_names` and `feature_names` of the right lengths
- `pain_keywords` has entries for all pain IDs with 3+ keywords each
- Different seeds produce different names/keywords (domain randomization)
- Existing sim behavior (funnel, oracle, scripted) is unchanged after the addition

---

### U3. Ad card renderer (`renderer.py`)

**Goal:** Render agent-written ad copy into a visual HTML card and save to disk.

**Requirements:** R1, R2, R8

**Dependencies:** U2 (for pain names in the card)

**Files:**
- `renderer.py` (new)
- `templates/ad_card.html` (new — Jinja2 template)
- `templates/feature_page.html` (new — Jinja2 template)

**Approach:**
- `render_ad_card(headline, body, cta, target_pains, pain_names, output_path) -> Path`:
  - Fill a Jinja2 HTML template with the ad content. Template has a clean card layout (gradient header, headline, body, CTA button, targeted-pains tags).
  - Save to `output_path` (e.g., `artifacts/{seed}/round_{N}/campaign_{i}/ad_card.html`).
- `render_feature_page(feature_name, description, benefits, output_path) -> Path`:
  - Fill a Jinja2 HTML template with a mini product landing page (hero, description, benefits list, CTA).
  - Save to `output_path`.
- Both functions are pure (no LLM, no network). Jinja2 is the only new dependency.
- Template CSS is inline (self-contained HTML files, no external assets).

**Patterns to follow:** Jinja2 `Environment(loader=FileSystemLoader("templates"))`. Templates use modern CSS (flexbox, gradients, rounded corners).

**Test scenarios:**
- `render_ad_card` produces valid HTML containing the headline and body text
- `render_feature_page` produces valid HTML containing the feature name and description
- Output files are written to the specified paths
- Templates handle empty/missing fields gracefully (no crash, shows placeholder)

---

### U4. Updated MCP tools (env.py)

**Goal:** Add optional NL parameters to the MCP tools. When provided, artifacts are scored and rendered; when omitted, the structured fast path runs.

**Requirements:** R1, R3, R5, R6, R10

**Dependencies:** U1, U2, U3

**Files:**
- `env.py` (modify `probe_market`, `build_feature`, `run_campaign`)
- `tasks.py` (update system prompt to explain the NL action format)

**Approach:**
- `probe_market` and `run_campaign` gain an optional `ad_copy: str = None` param. When provided:
  1. Call `score_ad_copy(ad_copy, world)` → get `craft`, `target_pains`.
  2. Call `render_ad_card(...)` → save HTML to artifacts dir.
  3. Use scored `craft` and extracted `target_pains` in the campaign (override the structured `target_pains` if NL mode).
  4. Log `ad_copy`, `craft`, rendered path in `_CURRENT_ROUND_ACTION`.
- `build_feature` gains an optional `spec: str = None` param. When provided:
  1. Call `score_feature_spec(spec, world)` → get `feature_id`, `quality`.
  2. Call `render_feature_page(...)` → save HTML to artifacts dir.
  3. Use scored `feature_id` and `quality` (override the structured `feature_id` if NL mode).
  4. Log `spec`, `quality`, rendered path in `_CURRENT_ROUND_ACTION`.
- When NL params are absent, behavior is identical to current (structured, `craft=1.0`, `quality=1.0`).
- Artifacts saved to `artifacts/{seed}/round_{N}/`.
- Verifier replay in `_replay_on_holdout` reads `craft` and `quality` from the action log instead of hardcoding 1.0.

**Patterns to follow:** Existing tool signature pattern (type-annotated async functions with docstrings). Backward compatibility via optional params with `None` defaults.

**Test scenarios:**
- `probe_market` with `ad_copy=None` behaves identically to current (craft=1.0)
- `probe_market` with `ad_copy="great ad about onboarding"` returns craft < 1.0 and extracts target pains
- `build_feature` with `spec=None` behaves identically to current (quality=1.0)
- `build_feature` with `spec="A dashboard for tracking payments"` returns quality < 1.0
- Artifact HTML files are created in the artifacts directory
- Action log entries include the NL fields when provided
- Verifier replay uses logged craft/quality values, not hardcoded 1.0

---

### U5. Artifact storage and manifest

**Goal:** Save all artifacts per-episode for the replay viewer.

**Requirements:** R8

**Dependencies:** U3, U4

**Files:**
- `env.py` (modify `_grade_episode` to write manifest)

**Approach:**
- Artifacts dir: `artifacts/{seed}/` with subdirs `round_{01..10}/campaigns/`, `round_{01..10}/features/`.
- At episode end (`_grade_episode`), write `artifacts/{seed}/manifest.json`:
  ```json
  {
    "seed": 42,
    "rounds": 10,
    "final_reward": 0.044,
    "holdout_profit": 1320.5,
    "actions": [... the full action log with artifact paths ...]
  }
  ```
- Artifacts dir is added to `.gitignore`.
- Renderer functions (U3) already write to these paths; this unit wires the manifest.

**Test scenarios:**
- After an episode with NL artifacts, `manifest.json` exists and is valid JSON
- Manifest contains the correct seed, round count, and reward
- Manifest action entries reference existing artifact HTML files
- Artifacts directory structure matches the expected layout

---

### U6. Replay viewer (`replay.html`)

**Goal:** A static HTML page that lets a human step through an episode and see ads, feature pages, and metrics side by side.

**Requirements:** R9

**Dependencies:** U5

**Files:**
- `replay.html` (new)

**Approach:**
- Single self-contained HTML file with vanilla JS.
- User opens it, picks an episode seed from a dropdown (reads manifest files via fetch from `python3 -m http.server`).
- For each round: shows the ad card HTML (inline via `<iframe srcdoc="...">`), feature page if built, campaign results table (audience/impressions/tries/purchases/revenue), running cash/profit chart.
- Styling: dark theme, card-based layout, responsive. No build step.
- "Play" button auto-advances rounds (1 round/second) for demo mode.

**Patterns to follow:** Standard vanilla JS SPA pattern. `fetch("artifacts/{seed}/manifest.json")` for data.

**Test scenarios:**
- Opens without errors when served via `python3 -m http.server`
- Dropdown lists available episode seeds from the artifacts directory
- Stepping through rounds shows the correct ad cards and feature pages
- Campaign results table matches the manifest data
- Running profit chart updates correctly each round

---

### U7. Updated system prompt and task definitions

**Goal:** Update the HUD task prompt to explain the NL action format and coach the agent to write ad copy and feature specs.

**Requirements:** R1, R3

**Dependencies:** U4

**Files:**
- `tasks.py` (modify `SYSTEM_PROMPT`)

**Approach:**
- Add to the tool descriptions: `probe_market` and `run_campaign` now accept optional `ad_copy` (a string with headline, body, CTA). `build_feature` now accepts optional `spec` (HTML or structured description of the feature).
- Add examples of good ad copy and feature specs to the strategy section.
- Keep the existing structured-path instructions as a fallback ("if you don't provide ad_copy, default targeting and craft=1.0 apply").

**Test scenarios:**
- Updated prompt mentions `ad_copy` and `spec` parameters
- HUD eval still works with the updated prompt (no syntax errors)
- An agent can successfully call `probe_market` with and without `ad_copy`

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM craft scoring adds latency to the RL loop | Use the cheapest/fastest model (Llama 3.2 3B, ~100ms). `fast_mode` flag bypasses LLM entirely for training. |
| Craft scores cluster around 0.5–0.7 (non-discriminative) | Calibrate the rubric prompt. Stretch craft to [0.3, 1.0] range via `craft = 0.3 + 0.7 * raw_score`. |
| Keyword-based targeting extraction misses nuanced copy | Acceptable — targeting is a secondary signal. The craft score captures overall quality. |
| Artifact storage grows large for many episodes | Only save artifacts in NL mode. Training in fast mode generates no artifacts. |
| Jinja2 is a new dependency | Minimal, well-established, already used in many HUD envs. Add to `pyproject.toml`. |

---

## Sources & Research

- Repo research: `sim.py` line 186 (`craft=1.0` hardcoded), line 210 (`quality=1.0`), `env.py` MCP tool pattern
- Cost research: Fireworks Llama 3.2 3B at $0.10/1M tokens (~$2/100K scoring calls); Flux Schnell on Replicate $0.001–0.003/image (deferred); OpenAI embeddings $0.02/1M tokens
- PLAN.md Phase 3 design: judge-as-translator pattern, NL layer deferred after numeric core validation
