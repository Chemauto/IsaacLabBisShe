# EnvTest Status Panel Observation Dimensions Design

**Goal:** Extend the live terminal status panel so it also shows the EnvTest unified observation dimension and the currently active policy input dimension.

## Scope

- Add `unified_obs_dim` to the panel.
- Add `policy_obs_dim` to the panel.
- Keep existing panel fields and ordering stable except for inserting the new dimension lines.

## Behavior

- `unified_obs_dim` is always read from the EnvTest `policy` observation group dimension.
- `policy_obs_dim` is the actual input width of the policy used on the current step.
- When the player is idle or has not built a policy observation for the current step, `policy_obs_dim` displays `None`.

## Integration

- Extend `StatusSnapshot` and the formatting helpers in `NewTools/envtest_status_panel.py`.
- Track the current `policy_obs` width inside `envtest_model_use_player.py` and forward it into the status snapshot builder.

## Error Handling

- Missing `policy_obs_dim` is expected in idle paths and should not be treated as an error.
