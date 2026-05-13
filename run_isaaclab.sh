#!/bin/bash
# Launch IsaacLab simulation + ROS2 bridge + robot_service
# Usage: bash run_isaaclab.sh

BASE="/home/xcj/work/IsaacLab/IsaacLabBisShe"

# Free port 8765 if occupied
lsof -ti:8765 | xargs kill -9 2>/dev/null || true

cleanup() {
    echo ""
    echo "[run_isaaclab] stopping..."
    kill -9 $PID_SIM $PID_BRIDGE $PID_SERVICE 2>/dev/null
    wait $PID_SIM $PID_BRIDGE $PID_SERVICE 2>/dev/null
    ros2 daemon stop 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

eval "$(conda shell.bash hook 2>/dev/null)"

# Must match legged_ws RMW (CycloneDDS) for topic discovery
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_HOME=/home/xcj/cyclonedds/install
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="lo" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'

# FinalSim: env_isaaclab conda env (subshell to isolate env)
(conda activate env_isaaclab && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && export CYCLONEDDS_HOME=/home/xcj/cyclonedds/install && export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>' && python "$BASE/Isaaclab/FinalSim.py" --scene_id 4 --enable_front_camera) &
PID_SIM=$!
echo "[run_isaaclab] started FinalSim (pid=$PID_SIM)"
sleep 2

# ros2_bridge + robot_service: ros2_env
conda activate ros2_env

python "$BASE/Isaaclab/ros2_bridge.py" &
PID_BRIDGE=$!
echo "[run_isaaclab] started ros2_bridge (pid=$PID_BRIDGE)"
sleep 1

python "$BASE/WebSocket/robot_service.py" &
PID_SERVICE=$!
echo "[run_isaaclab] started robot_service (pid=$PID_SERVICE)"

echo "[run_isaaclab] all processes running. Ctrl+C to stop."
wait
