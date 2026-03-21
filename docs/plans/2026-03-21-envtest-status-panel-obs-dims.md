# EnvTest Status Panel Observation Dimensions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Show both the unified EnvTest observation dimension and the current policy input dimension in the live terminal status panel.

**Architecture:** Keep the change minimal by extending the existing status snapshot dataclass and formatting function, then pass the dimensions from the player loop without changing control flow.

**Tech Stack:** Python, pytest-style targeted tests

---

### Task 1: Add failing panel-format tests

**Files:**
- Modify: `tests/test_envtest_status_panel.py`
- Modify: `NewTools/envtest_status_panel.py`

**Step 1: Write the failing test**

```python
def test_build_status_lines_renders_observation_dimensions():
    ...
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_status_panel.py -q`
Expected: FAIL because the new fields do not exist yet.

**Step 3: Write minimal implementation**

- add `unified_obs_dim`
- add `policy_obs_dim`
- render them in the panel

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_status_panel.py -q`
Expected: PASS

### Task 2: Thread dimensions from the player loop

**Files:**
- Modify: `NewTools/envtest_model_use_player.py`

**Step 1: Implement minimal integration**

- compute the unified observation dimension from the observation manager
- set current `policy_obs_dim` only when a policy observation is produced
- pass both values into the status snapshot

**Step 2: Run targeted verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_status_panel.py -q`
Expected: PASS

### Task 3: Run syntax verification

**Files:**
- Modify: `NewTools/envtest_status_panel.py`
- Modify: `NewTools/envtest_model_use_player.py`

**Step 1: Run syntax verification**

Run: `python -m py_compile NewTools/envtest_status_panel.py NewTools/envtest_model_use_player.py tests/test_envtest_status_panel.py`
Expected: PASS
