import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# =============================
# Editable defaults (model_trans_rough.py style)
# =============================
# TASK_NAME = "Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0"
# CHECKPOINT_PATH = "/home/xcj/work/IsaacLab/IsaacLabBisShe/ModelBackup/BiShePolicy/BiSheClimbPolicy.pt"
# VIDEO_OUTPUT_DIR = "videos/play"
# FPS = 50  # 行走任务的默认帧率

TASK_NAME = "Template-Push-Box-Go2-Play-v0"
CHECKPOINT_PATH = "/home/xcj/work/IsaacLab/modeltest/model_7800.pt"
VIDEO_OUTPUT_DIR = "videos/play"
FPS = 50  # Play 配置：decimation=8，录制帧率 = 1/(8*0.005) = 25 fps，用 25 fps 播放是正常速度且流畅

# =============================
# 1. args (optional overrides)
# =============================
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default=TASK_NAME)
parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")

# video
parser.add_argument("--video", action="store_true", default=True)
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Stop after this many steps if > 0 and --no-stop_on_episode. 0 means no step cap.",
)
parser.add_argument("--video_output_dir", type=str, default=VIDEO_OUTPUT_DIR)
parser.add_argument("--video_name", type=str, default="")
parser.add_argument("--fps", type=int, default=FPS)
parser.add_argument(
    "--stop_on_episode",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Stop after first episode termination (recommended for one complete clip).",
)

# camera follow (keep current defaults unchanged)
parser.add_argument("--follow_camera", action="store_true", default=True)
parser.add_argument(
    "--camera_eye_offset",
    type=float,
    nargs=3,
    default=[0.0, 3.2, 1.1],
    help="Camera eye offset relative to robot base position.",
)
parser.add_argument(
    "--camera_target_offset",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.45],
    help="Camera target offset relative to robot base position.",
)

# append AppLauncher args (headless/device/enable_cameras/etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# recording needs camera rendering
if args_cli.video:
    args_cli.enable_cameras = True

if not os.path.isfile(args_cli.checkpoint):
    raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")

print("[INFO] Using config:")
print(f"  task: {args_cli.task}")
print(f"  checkpoint: {args_cli.checkpoint}")
print(f"  num_envs: {args_cli.num_envs}")
print(f"  device: {args_cli.device}")
print(f"  video_output_dir: {args_cli.video_output_dir}")
print(f"  stop_on_episode: {args_cli.stop_on_episode}")
print(f"  video_length: {args_cli.video_length}")
print(f"  fps: {args_cli.fps}")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# register standard and custom tasks
import isaaclab_tasks  # noqa: F401
import MyProject.tasks  # noqa: F401

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# =============================
# 2. environment
# =============================
env_cfg = parse_env_cfg(
    args_cli.task,
    device=args_cli.device,
    num_envs=args_cli.num_envs,
)

gym_env = gym.make(
    args_cli.task,
    cfg=env_cfg,
    render_mode="rgb_array" if args_cli.video else None,
)
base_env = gym_env.unwrapped

# =============================
# 3. camera helper
# =============================
def update_follow_camera() -> None:
    if not args_cli.follow_camera:
        return
    robot = base_env.scene["robot"]
    root_pos = robot.data.root_pos_w[0, :3].detach().cpu().numpy()
    eye = (root_pos + np.asarray(args_cli.camera_eye_offset, dtype=np.float32)).tolist()
    target = (root_pos + np.asarray(args_cli.camera_target_offset, dtype=np.float32)).tolist()
    base_env.sim.set_camera_view(eye=eye, target=target)


def should_stop(done: bool, timestep: int) -> bool:
    if args_cli.stop_on_episode and done:
        return True
    if (not args_cli.stop_on_episode) and args_cli.video_length > 0 and timestep >= args_cli.video_length:
        return True
    return False


# =============================
# 4. manual video writer (single complete mp4)
# =============================
video_writer = None
video_path = ""
frames_written = 0
warned_no_frame = False
if args_cli.video:
    import imageio.v2 as imageio

    os.makedirs(args_cli.video_output_dir, exist_ok=True)
    name = args_cli.video_name or f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    video_path = os.path.join(args_cli.video_output_dir, f"{name}.mp4")
    video_writer = imageio.get_writer(video_path, fps=args_cli.fps)
    print(f"[INFO] Video writer opened: {video_path}")


def write_frame() -> None:
    global frames_written, warned_no_frame
    if video_writer is None:
        return
    frame = gym_env.render()
    if frame is None:
        if not warned_no_frame:
            print("[WARN] render() returned None; no frame captured yet.")
            warned_no_frame = True
        return
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    video_writer.append_data(frame)
    frames_written += 1


# =============================
# 5. rollout
# =============================
checkpoint = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
vec_env = None

try:
    # Case A: callable policy checkpoint
    if (isinstance(checkpoint, dict) and "policy" in checkpoint) or hasattr(checkpoint, "eval"):
        print("[INFO] Detected callable policy checkpoint.")
        policy = checkpoint["policy"] if isinstance(checkpoint, dict) else checkpoint
        policy.eval()

        obs, _ = gym_env.reset()
        write_frame()
        timestep = 0

        while simulation_app.is_running():
            update_follow_camera()

            with torch.inference_mode():
                policy_obs = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
                action = policy(policy_obs)

            obs, _, terminated, truncated, _ = gym_env.step(action)
            write_frame()
            timestep += 1

            done = bool(np.any(np.asarray(terminated)) or np.any(np.asarray(truncated)))
            if should_stop(done, timestep):
                print(f"[INFO] Stop at step={timestep}, done={done}")
                break

    # Case B: RSL-RL training checkpoint
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("[INFO] Detected RSL-RL training checkpoint. Loading through runner.")

        agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
        agent_cfg.device = args_cli.device

        vec_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        obs = vec_env.get_observations()
        write_frame()
        timestep = 0

        while simulation_app.is_running():
            update_follow_camera()

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = vec_env.step(actions)
                policy_nn.reset(dones)

            write_frame()
            timestep += 1

            done = bool(torch.any(dones).item())
            if should_stop(done, timestep):
                print(f"[INFO] Stop at step={timestep}, done={done}")
                break

    else:
        raise RuntimeError(
            "Unsupported checkpoint format. Expected either:\n"
            "1) exported policy checkpoint (contains key 'policy' or callable module), or\n"
            "2) RSL-RL training checkpoint (contains key 'model_state_dict')."
        )

finally:
    if vec_env is not None:
        vec_env.close()
    else:
        gym_env.close()

    if video_writer is not None:
        video_writer.close()
        print(f"[INFO] Video saved: {video_path}")
        print(f"[INFO] Frames written: {frames_written}")

    simulation_app.close()
