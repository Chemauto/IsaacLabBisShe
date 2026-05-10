from WebSocket.protocol import walk_velocity


def test_walk_velocity_uses_default_speed_when_v_is_missing():
    assert walk_velocity({"direction": "back"}) == (-0.5, 0.0, 0.0)


def test_walk_velocity_keeps_explicit_speed():
    assert walk_velocity({"direction": "left", "v": 0.25}) == (0.0, 0.25, 0.0)
