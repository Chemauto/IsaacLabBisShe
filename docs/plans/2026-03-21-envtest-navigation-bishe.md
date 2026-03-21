# EnvTest Navigation BiShe Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the `NavigationBishe.pt` policy as `model_use=4` in the EnvTest player and allow the socket path to select it.

**Architecture:** Build a small pure helper module that converts world-frame goals into NavigationTest base-frame `pose_command` tensors, then use a new high-level navigation branch in the player that reuses the existing rough-walk low-level policy. Extend the socket/client protocol only where necessary so `model_use=4` can be selected externally.

**Tech Stack:** Python, torch, argparse, UDP socket control, pytest-style targeted tests

---

### Task 1: Add failing tests for navigation bridge math and protocol acceptance

**Files:**
- Create: `tests/test_envtest_navigation_bridge.py`
- Modify: `Socket/envtest_socket_server.py`

**Step 1: Write the failing test**

```python
def test_build_navigation_pose_command_rotates_world_goal_into_body_frame():
    ...
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_navigation_bridge.py -q`
Expected: FAIL because the helper module and model id support do not exist yet.

**Step 3: Write minimal implementation**

- add a pure helper for world-goal -> base-frame pose command
- extend server-side model id validation to include `4`

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_navigation_bridge.py -q`
Expected: PASS

### Task 2: Add the new player branch

**Files:**
- Create: `NewTools/envtest_navigation_bridge.py`
- Modify: `NewTools/envtest_model_use_player.py`

**Step 1: Implement the minimal integration**

- register `model_use=4`
- load `NavigationBishe.pt` with the correct actor shape
- build 197-D navigation observations from EnvTest state
- hold high-level navigation actions for 10 EnvTest steps
- feed the processed 3-D output into the existing rough-walk low-level policy

**Step 2: Keep external control consistent**

- accept `model_use=4` in `Socket/envtest_socket_client.py`
- keep status panel skill naming aligned

**Step 3: Run targeted verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_navigation_bridge.py tests/test_envtest_socket_reset.py tests/test_envtest_status_panel.py -q`
Expected: PASS

### Task 3: Run syntax verification

**Files:**
- Modify: `NewTools/envtest_model_use_player.py`
- Modify: `Socket/envtest_socket_client.py`
- Modify: `Socket/envtest_socket_server.py`
- Create: `NewTools/envtest_navigation_bridge.py`

**Step 1: Run syntax verification**

Run: `python -m py_compile NewTools/envtest_navigation_bridge.py NewTools/envtest_model_use_player.py Socket/envtest_socket_client.py Socket/envtest_socket_server.py tests/test_envtest_navigation_bridge.py`
Expected: PASS
