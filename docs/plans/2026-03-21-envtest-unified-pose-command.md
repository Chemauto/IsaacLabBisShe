# EnvTest Unified Pose Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the navigation `pose_command` into the EnvTest unified observation so all model branches slice from one shared observation vector.

**Architecture:** Add a small pure observation-schema module that defines the unified observation dimensions and term groupings, then thread a new runtime `pose_command` buffer through EnvTest and switch the navigation branch to slice its `197`-D input from the shared observation.

**Tech Stack:** Python, torch, pytest-style targeted tests

---

### Task 1: Add failing tests for the unified observation schema

**Files:**
- Create: `tests/test_envtest_unified_obs_schema.py`
- Create: `source/MyProject/MyProject/tasks/manager_based/EnvTest/observation_schema.py`

**Step 1: Write the failing test**

```python
def test_unified_policy_dim_includes_pose_command():
    ...
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_unified_obs_schema.py -q`
Expected: FAIL because the schema module does not exist yet.

**Step 3: Write minimal implementation**

- define unified term dims
- define total unified dimension `256`
- define navigation slice terms and dimension `197`

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_unified_obs_schema.py -q`
Expected: PASS

### Task 2: Wire pose_command into EnvTest runtime observations

**Files:**
- Modify: `source/MyProject/MyProject/tasks/manager_based/EnvTest/mdp/observations.py`
- Modify: `source/MyProject/MyProject/tasks/manager_based/EnvTest/env_test_env.py`
- Modify: `source/MyProject/MyProject/tasks/manager_based/EnvTest/env_test_env_cfg.py`

**Step 1: Implement the minimal runtime buffer path**

- add a runtime `pose_command` buffer
- expose it through a new observation term
- add it to the unified `policy` observation group

**Step 2: Keep the layout and docs aligned**

- update the unified observation comments from `252` to `256`

### Task 3: Switch navigation slicing to the unified observation

**Files:**
- Modify: `NewTools/envtest_model_use_player.py`

**Step 1: Implement the minimal player change**

- compute the base-frame navigation `pose_command`
- write it into the runtime observation buffers
- slice navigation inputs from the unified observation instead of custom concatenation

**Step 2: Run targeted verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_unified_obs_schema.py tests/test_envtest_navigation_bridge.py tests/test_envtest_status_panel.py tests/test_envtest_socket_reset.py -q`
Expected: PASS

### Task 4: Run syntax verification

**Files:**
- Modify: `source/MyProject/MyProject/tasks/manager_based/EnvTest/mdp/observations.py`
- Modify: `source/MyProject/MyProject/tasks/manager_based/EnvTest/env_test_env.py`
- Modify: `source/MyProject/MyProject/tasks/manager_based/EnvTest/env_test_env_cfg.py`
- Modify: `NewTools/envtest_model_use_player.py`

**Step 1: Run syntax verification**

Run: `python -m py_compile source/MyProject/MyProject/tasks/manager_based/EnvTest/observation_schema.py source/MyProject/MyProject/tasks/manager_based/EnvTest/mdp/observations.py source/MyProject/MyProject/tasks/manager_based/EnvTest/env_test_env.py source/MyProject/MyProject/tasks/manager_based/EnvTest/env_test_env_cfg.py NewTools/envtest_model_use_player.py tests/test_envtest_unified_obs_schema.py`
Expected: PASS
