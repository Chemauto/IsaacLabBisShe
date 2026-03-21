# EnvTest Reset Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a one-shot `reset` UDP command that resets the environment while preserving current control settings.

**Architecture:** Reuse the existing client -> server -> control-file -> player pattern by adding a dedicated reset flag file. Keep reset handling edge-triggered in the player by clearing the flag immediately after `env.reset()`.

**Tech Stack:** Python, argparse, UDP sockets, file-based control flags, pytest-style targeted tests

---

### Task 1: Add failing tests for reset protocol

**Files:**
- Create: `tests/test_envtest_socket_reset.py`
- Modify: `Socket/envtest_socket_client.py`
- Modify: `Socket/envtest_socket_server.py`

**Step 1: Write the failing test**

```python
def test_build_message_emits_reset_field():
    ...
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_socket_reset.py -q`
Expected: FAIL because reset support is not implemented yet.

**Step 3: Write minimal implementation**

- add `--reset` in the client
- add reset parsing and reset file output in the server

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_socket_reset.py -q`
Expected: PASS

### Task 2: Wire one-shot reset into the player

**Files:**
- Modify: `NewTools/envtest_model_use_player.py`

**Step 1: Write the failing test**

```python
def test_consume_reset_request_rewrites_flag_to_zero():
    ...
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_socket_reset.py -q`
Expected: FAIL because player reset helpers do not exist yet.

**Step 3: Write minimal implementation**

- add `--reset_file`
- read and consume reset requests
- perform `env.reset()` and clear push-state buffers

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_socket_reset.py -q`
Expected: PASS

### Task 3: Run verification

**Files:**
- Modify: `Socket/envtest_socket_client.py`
- Modify: `Socket/envtest_socket_server.py`
- Modify: `NewTools/envtest_model_use_player.py`
- Test: `tests/test_envtest_socket_reset.py`

**Step 1: Run targeted tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_envtest_status_panel.py tests/test_envtest_socket_reset.py -q`
Expected: PASS

**Step 2: Run syntax verification**

Run: `python -m py_compile Socket/envtest_socket_client.py Socket/envtest_socket_server.py NewTools/envtest_model_use_player.py tests/test_envtest_socket_reset.py`
Expected: PASS
