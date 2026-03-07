import argparse
import os

import gymnasium as gym
import numpy as np
import torch

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# =============================
# Editable defaults (model_trans_rough.py style)
# =============================
TASK_NAME = "Template-Velocity-Go2-Walk-BiShe-Pit-Play-v0"
CHECKPOINT_PATH = "/home/robot/work/IsaacLabBisShe/ModelBackup/BiShePolicy/BiSheClimbPitPolicy.pt"
VIDEO_OUTPUT_DIR = "videos/play"

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
    help="Recorded clip length in steps. Use 0 to record until episode termination.",
)
parser.add_argument("--video_output_dir", type=str, default=VIDEO_OUTPUT_DIR)
parser.add_argument(
    "--stop_on_episode",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Stop after first episode termination (recommended for one complete clip).",
)

# camera follow
parser.add_argument("--follow_camera", action="store_true", default=True)
parser.add_argument(
    "--camera_eye_offset",
    type=float,
    nargs=3,
    default=[0.0, 3.0, 1.4],
    help="Camera eye offset relative to robot base position.",
)
parser.add_argument(
    "--camera_target_offset",
    type=float,
    nargs=3,
    default=[0.6, 0.0, 0.5],
    help="Camera target offset relative to robot base position (lower z => look more downward).",
)
# eye_offset = [-3.0, 0, 1.4]
# target_offset = [0.6, 0, 0.5]
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
# 2. environment and wrappers
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

if args_cli.video:
    from gymnasium.wrappers import RecordVideo

    os.makedirs(args_cli.video_output_dir, exist_ok=True)
    gym_env = RecordVideo(
        gym_env,
        video_folder=args_cli.video_output_dir,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )


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
    if (not args_cli.stop_on_episode) and args_cli.video and args_cli.video_length > 0 and timestep >= args_cli.video_length:
        return True
    return False


# =============================
# 4. load checkpoint
# =============================
checkpoint = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)

# Case A: exported policy checkpoint (callable module directly)
if (isinstance(checkpoint, dict) and "policy" in checkpoint) or hasattr(checkpoint, "eval"):
    print("[INFO] Detected callable policy checkpoint.")
    policy = checkpoint["policy"] if isinstance(checkpoint, dict) else checkpoint
    policy.eval()

    obs, _ = gym_env.reset()
    timestep = 0
    while simulation_app.is_running():
        update_follow_camera()

        with torch.inference_mode():
            policy_obs = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
            action = policy(policy_obs)

        obs, _, terminated, truncated, _ = gym_env.step(action)
        timestep += 1

        done = bool(np.any(np.asarray(terminated)) or np.any(np.asarray(truncated)))
        if should_stop(done, timestep):
            print(f"[INFO] Stop at step={timestep}, done={done}")
            break

# Case B: RSL-RL training checkpoint (model_state_dict / optimizer_state_dict / iter / infos)
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

    # keep recurrent policy states clean when episodes reset
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    obs = vec_env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        update_follow_camera()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = vec_env.step(actions)
            policy_nn.reset(dones)

        timestep += 1
        done = bool(torch.any(dones).item())
        if should_stop(done, timestep):
            print(f"[INFO] Stop at step={timestep}, done={done}")
            break

    vec_env.close()

else:
    raise RuntimeError(
        "Unsupported checkpoint format. Expected either:\n"
        "1) exported policy checkpoint (contains key 'policy' or callable module), or\n"
        "2) RSL-RL training checkpoint (contains key 'model_state_dict')."
    )

# =============================
# 5. close
# =============================
gym_env.close()
simulation_app.close()
