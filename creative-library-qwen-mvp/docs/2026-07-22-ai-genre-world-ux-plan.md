# AI Genre, World Extraction, and Creative Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make genre, content-driven emotion curves, characters, factions, and world elements model-derived and human-editable while simplifying the left content navigation.

**Architecture:** Keep the existing FastAPI and single-file frontend. Extend the analysis JSON with a focused `world` payload produced by a second model request, normalize it independently, aggregate uploaded episode entities in the browser, and preserve compatibility with old local JSON files.

**Tech Stack:** Python 3.11, FastAPI, Pydantic 2, OpenAI-compatible Qwen API, vanilla HTML/CSS/JavaScript, pytest.

## Global Constraints

- Existing analysis and upload files in `runtime-data` must not be deleted or replaced.
- Whole-series analysis may only use successfully parsed uploaded videos.
- Audience emotion curves are qualitative content predictions, not measured audience data.
- Unknown character or world attributes must be labelled as incomplete instead of invented.
- Human editing remains available without adding a database.

---

### Task 1: Analysis Contract and Prompt Semantics

**Files:**
- Modify: `app/schema.py`
- Modify: `app/qwen.py`
- Modify: `app/main.py`
- Test: `tests/test_schema.py`
- Test: `tests/test_qwen_upload.py`

**Interfaces:**
- Produces: `WorldAnalysis`, `CharacterEntity`, `RelationshipEntity`, `FactionEntity`, `WorldElementEntity` and `ModelAnalysis.world`.
- Produces: `build_world_prompt(...)` and `QwenClient.analyze_world(...)`.

- [ ] Write tests proving genre is inferred rather than forced, emotion curve wording excludes comments, and world entities accept incomplete values.
- [ ] Run the focused tests and confirm they fail for the missing contract.
- [ ] Add the Pydantic models and two-stage Qwen prompt/request implementation.
- [ ] Normalize AI genre and world entities without overwriting them with upload hints.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Focused World Reanalysis API

**Files:**
- Modify: `app/main.py`
- Modify: `app/storage.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `POST /api/jobs/{job_id}/reanalyze-world`.
- Consumes: persisted job video path and current analysis JSON.

- [ ] Write an API test proving an existing completed job can request focused world extraction without re-uploading.
- [ ] Run the test and confirm the endpoint is absent.
- [ ] Implement a background retry stage that updates only `world` and leaves core fields intact.
- [ ] Run the API tests and confirm old analysis files remain compatible.

### Task 3: Genre and World Editing UI

**Files:**
- Modify: `web/index.html`
- Test: `tests/test_frontend_modal.py`

**Interfaces:**
- Consumes: `analysis.world` and the reanalysis endpoint.
- Produces: separate editable character, faction, relationship, and world-element records in frontend state.

- [ ] Write frontend contract tests for the expanded genre datalist, AI-auto default, add/delete entity actions, and incomplete-background notice.
- [ ] Run the tests and confirm the required UI strings/actions are missing.
- [ ] Apply real analysis world entities to video state and aggregate them by actual uploaded episodes.
- [ ] Add edit, add, delete, retry, and uncertain-character merge controls.
- [ ] Run frontend tests and confirm the controls are present.

### Task 4: Content-Driven Curve and Left Navigation

**Files:**
- Modify: `web/index.html`
- Test: `tests/test_frontend_modal.py`

**Interfaces:**
- Produces: one-series content navigator and `renderSectionDirectory()`.
- Consumes: the current six-section definitions and `jump-section` dispatch action.

- [ ] Write frontend tests proving only the selected series is rendered, episode lists have vertical-only scrolling, and six directory anchors are present.
- [ ] Run the tests and confirm they fail.
- [ ] Render a single selected series, compact episode rows, and dynamic six-section directory.
- [ ] Update the curve title, explanatory copy, y-axis labels, and hover reasons to state it is a content-based qualitative prediction.
- [ ] Run the frontend tests and confirm they pass.

### Task 5: End-to-End Verification

**Files:**
- Verify: `app/*.py`
- Verify: `web/index.html`
- Verify: `tests/*.py`

**Interfaces:**
- Verifies all interfaces from Tasks 1–4.

- [ ] Run `.venv/bin/pytest -q` and require all tests to pass.
- [ ] Start the local app in mock mode on an unused port.
- [ ] Verify upload, AI genre, world extraction retry, entity editing, one-series navigation, anchors, and curve hover at desktop width.
- [ ] Verify the same page has no overlap or horizontal episode scrolling at a narrow viewport.
- [ ] Stop only the verification server started for this task.
