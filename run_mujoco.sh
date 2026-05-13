#!/bin/bash
# Launch MuJoCo simulation + DDS→ROS2 bridge + robot_service
# Usage: bash run_mujoco.sh

BASE="/home/xcj/work/IsaacLab/IsaacLabBisShe"
MUJOCO_DIR="$BASE/Mujoco/simulate_python"

# Free port 8765 if occupied
lsof -ti:8765 | xargs kill -9 2>/dev/null || true

cleanup() {
    echo ""
    echo "[run_mujoco] stopping..."
    kill -9 $PID_SIM $PID_DDS $PID_BRIDGE $PID_SERVICE 2>/dev/null
    wait $PID_SIM $PID_DDS $PID_BRIDGE $PID_SERVICE 2>/dev/null
    ros2 daemon stop 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate ros2_env

# Must match legged_ws RMW (CycloneDDS) for topic discovery
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_HOME=/home/xcj/cyclonedds/install
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="lo" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'

# unitree_mujoco: must run from simulate_python dir (relative paths in config)
cd "$MUJOCO_DIR" && python unitree_mujoco.py &
PID_SIM=$!
echo "[run_mujoco] started unitree_mujoco (pid=$PID_SIM)"
sleep 2

# mujoco_dds_state
python "$MUJOCO_DIR/mujoco_dds_state.py" &
PID_DDS=$!
echo "[run_mujoco] started mujoco_dds_state (pid=$PID_DDS)"
sleep 1

# mujoco_ros2_bridge
python "$MUJOCO_DIR/mujoco_ros2_bridge.py" &
PID_BRIDGE=$!
echo "[run_mujoco] started mujoco_ros2_bridge (pid=$PID_BRIDGE)"
sleep 1

# robot_service
python "$BASE/WebSocket/robot_service.py" &
PID_SERVICE=$!
echo "[run_mujoco] started robot_service (pid=$PID_SERVICE)"

echo "[run_mujoco] all processes running. Ctrl+C to stop."
wait
