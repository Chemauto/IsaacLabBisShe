# EnvTest Status Panel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a continuously refreshed terminal status panel to the EnvTest player that shows runtime commands plus robot, goal, obstacle, and box geometry with `None` for missing values.

**Architecture:** Keep scene-state collection and terminal rendering in small pure helper functions so the main simulation loop only gathers runtime values and calls a single render function. Reuse existing EnvTest scene constants and geometry helpers instead of duplicating obstacle metadata.

**Tech Stack:** Python, torch, pytest-style targeted unit tests, ANSI terminal control

---

### Task 1: Add failing tests for panel formatting

**Files:**
- Create: `tests/test_envtest_status_panel.py`
- Modify: `NewTools/envtest_model_use_player.py`

**Step 1: Write the failing test**

```python
def test_format_status_lines_renders_none_for_missing_fields():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_envtest_status_panel.py -q`
Expected: FAIL because helper functions do not exist yet.

**Step 3: Write minimal implementation**

Add pure helpers for:
- vector formatting
- asset formatting
- panel line rendering

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_envtest_status_panel.py -q`
Expected: PASS

### Task 2: Collect live EnvTest state

**Files:**
- Modify: `NewTools/envtest_model_use_player.py`

**Step 1: Write the failing test**

```python
def test_snapshot_keeps_platform_slots_stable():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_envtest_status_panel.py -q`
Expected: FAIL because snapshot assembly is missing.

**Step 3: Write minimal implementation**

Add helpers that:
- map `model_use` to skill name
- select `platform_1` and `platform_2`
- use `None` for absent assets
- carry runtime `start`, `vel_command`, and `pose_command`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_envtest_status_panel.py -q`
Expected: PASS

### Task 3: Integrate the terminal overlay into the player loop

**Files:**
- Modify: `NewTools/envtest_model_use_player.py`

**Step 1: Implement minimal loop integration**

- Add CLI options for enabling and throttling the panel if needed.
- Update the loop to gather current state and refresh the panel periodically.
- Keep existing logs intact.

**Step 2: Run targeted verification**

Run: `pytest tests/test_envtest_status_panel.py -q`
Expected: PASS

**Step 3: Run syntax verification**

Run: `python -m py_compile NewTools/envtest_model_use_player.py tests/test_envtest_status_panel.py`
Expected: PASS
