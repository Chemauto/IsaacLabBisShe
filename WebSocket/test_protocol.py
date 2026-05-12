from WebSocket.protocol import push_goal, stop_command_payload, walk_velocity


def test_walk_velocity_uses_default_speed_when_v_is_missing():
    assert walk_velocity({"direction": "back"}) == (-0.5, 0.0, 0.0)


def test_walk_velocity_keeps_explicit_speed():
    assert walk_velocity({"direction": "left", "v": 0.25}) == (0.0, 0.25, 0.0)


def test_push_goal_uses_box_height_when_z_is_missing_or_zero():
    box_world = {"x": 0.78, "y": 0.0, "z": 0.16}
    assert push_goal({"x": 1.1, "y": 0.0}, box_world) == [1.1, 0.0, 0.16]
    assert push_goal({"x": 1.1, "y": 0.0, "z": 0.0}, box_world) == [1.1, 0.0, 0.16]


def test_push_goal_falls_back_to_deploy_default_height():
    assert push_goal({"x": 1.1, "y": 0.0}, {}) == [1.1, 0.0, 0.12]


def test_stop_command_payload_is_idle_not_zero_velocity_walk():
    assert stop_command_payload() == {
        "model_use": 0,
        "skill": "idle",
        "start": False,
        "velocity": [0.0, 0.0, 0.0],
    }
