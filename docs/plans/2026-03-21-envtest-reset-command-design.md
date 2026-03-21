# EnvTest Reset Command Design

**Goal:** Add a dedicated `reset` control path so the UDP client can trigger a one-shot `env.reset()` without changing the current `model_use`, `start`, `goal`, or `velocity` settings.

## Scope

- Extend the UDP client to send a reset command.
- Extend the UDP server to parse reset requests and write a separate reset flag file.
- Extend the player to poll the reset flag file and execute exactly one environment reset per request.
- Keep the existing status panel unchanged; it should reflect post-reset state naturally.

## Protocol

- Client sends `reset=1` when `--reset` is provided.
- Server accepts:
  - `reset=1`
  - `reset=true`
  - plain `reset`
- Server writes the reset flag into a dedicated file such as `/tmp/envtest_reset.txt`.

## Player Behavior

- The player checks the reset flag every step.
- When a reset request is observed:
  - call `env.reset()`
  - clear push-related runtime buffers and counters
  - write the reset flag back to `0`
- The player must not change current control files or inferred commands.

## Error Handling

- Missing reset file means no reset request.
- Invalid reset values are ignored.
- Reset handling must be edge-triggered by clearing the file after the reset is consumed.
