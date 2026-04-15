# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import json
from datetime import datetime
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=0,
    help="If > 0, run evaluation until this many completed episodes are collected and save metrics to disk.",
)
parser.add_argument(
    "--eval_output_dir",
    type=str,
    default=None,
    help="Optional directory to store evaluation outputs. Defaults to <checkpoint_dir>/eval/<timestamp>.",
)
parser.add_argument(
    "--eval_tensorboard",
    action="store_true",
    default=False,
    help="Write evaluation metrics to TensorBoard under <eval_output_dir>/tensorboard.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# 默认开启相机渲染，避免带相机的环境因缺少 --enable_cameras 报错。
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import MyProject.tasks  # noqa: F401


def _to_float_if_scalar(value) -> float | None:
    """Convert a scalar-like value into a float."""
    if isinstance(value, (int, float, bool)):
        return float(value)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.item())
    return None


def _extract_eval_scalars(log_data: dict) -> dict[str, float]:
    """Keep only scalar values from Isaac Lab episode logs."""
    scalars = {}
    for key, value in log_data.items():
        scalar_value = _to_float_if_scalar(value)
        if scalar_value is not None:
            scalars[key] = scalar_value
    return scalars


def _summarize_eval_records(records: list[dict], requested_episodes: int) -> dict[str, object]:
    """Compute weighted means over reset events."""
    total_completed = 0
    weighted_sums: dict[str, float] = {}

    for record in records:
        remaining = requested_episodes - total_completed if requested_episodes > 0 else record["num_resets"]
        if remaining <= 0:
            break
        weight = min(int(record["num_resets"]), remaining)
        total_completed += weight
        for key, value in record.items():
            if key in {"step", "num_resets"}:
                continue
            weighted_sums[key] = weighted_sums.get(key, 0.0) + float(value) * weight

    metric_means = {}
    if total_completed > 0:
        metric_means = {key: value / total_completed for key, value in weighted_sums.items()}

    return {
        "requested_episodes": requested_episodes,
        "completed_episodes": total_completed,
        "num_reset_events": len(records),
        "metric_means": metric_means,
    }


def _write_eval_outputs(
    output_dir: str,
    records: list[dict],
    summary: dict[str, object],
    task_name: str,
    checkpoint_path: str,
    num_envs: int,
) -> None:
    """Write evaluation files to disk."""
    os.makedirs(output_dir, exist_ok=True)

    summary_payload = {
        "task": task_name,
        "checkpoint": checkpoint_path,
        "num_envs": num_envs,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **summary,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False)

    records_path = os.path.join(output_dir, "episodes.csv")
    fieldnames = ["step", "num_resets"]
    extra_fields = sorted({key for record in records for key in record.keys()} - set(fieldnames))
    fieldnames.extend(extra_fields)
    with open(records_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[INFO] Evaluation summary written to: {summary_path}")
    print(f"[INFO] Evaluation episode logs written to: {records_path}")


def _write_eval_tensorboard_scalars(writer, record: dict[str, float], completed_episodes: int) -> None:
    """Write one reset event into TensorBoard."""
    writer.add_scalar("eval/num_resets", record["num_resets"], completed_episodes)
    for key, value in record.items():
        if key in {"step", "num_resets"}:
            continue
        writer.add_scalar(f"eval/{key}", value, completed_episodes)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    eval_mode = args_cli.eval_episodes > 0
    eval_records: list[dict] = []
    eval_output_dir = None
    eval_tb_writer = None
    if eval_mode:
        eval_output_dir = args_cli.eval_output_dir or os.path.join(
            log_dir, "eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        print(f"[INFO] Evaluation mode enabled for {args_cli.eval_episodes} completed episodes.")
        print(f"[INFO] Evaluation outputs will be saved to: {eval_output_dir}")
        if args_cli.eval_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(eval_output_dir, "tensorboard")
            eval_tb_writer = SummaryWriter(log_dir=tb_dir)
            print(f"[INFO] Evaluation TensorBoard logs will be written to: {tb_dir}")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    completed_eval_episodes = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, infos = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if eval_mode:
            done_count = int((dones > 0).sum().item())
            log_data = infos.get("log")
            if done_count > 0:
                record = {"step": timestep, "num_resets": done_count}
                if isinstance(log_data, dict):
                    record.update(_extract_eval_scalars(log_data))
                else:
                    print("[WARNING] Completed episodes were detected, but Isaac Lab did not return extras['log'].")
                eval_records.append(record)
                completed_eval_episodes += done_count
                if eval_tb_writer is not None:
                    _write_eval_tensorboard_scalars(eval_tb_writer, record, completed_eval_episodes)
                print(
                    f"[INFO] Evaluation progress: {completed_eval_episodes}/{args_cli.eval_episodes} completed episodes."
                )

        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length and not eval_mode:
                break

        if eval_mode and completed_eval_episodes >= args_cli.eval_episodes:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if eval_mode:
        summary = _summarize_eval_records(eval_records, args_cli.eval_episodes)
        _write_eval_outputs(
            output_dir=eval_output_dir,
            records=eval_records,
            summary=summary,
            task_name=args_cli.task,
            checkpoint_path=resume_path,
            num_envs=env.unwrapped.num_envs,
        )

        metric_means = summary["metric_means"]
        if eval_tb_writer is not None:
            for key, value in metric_means.items():
                eval_tb_writer.add_scalar(f"eval_summary/{key}", value, summary["completed_episodes"])
            eval_tb_writer.add_scalar(
                "eval_summary/completed_episodes", summary["completed_episodes"], summary["completed_episodes"]
            )
            eval_tb_writer.flush()
            eval_tb_writer.close()

        tracked_keys = [
            "Metrics/base_velocity/error_vel_xy",
            "Metrics/base_velocity/error_vel_yaw",
            "Episode_Termination/time_out",
            "Episode_Termination/base_contact",
        ]
        print("[INFO] Evaluation metrics summary:")
        for key in tracked_keys:
            if key in metric_means:
                print(f"  - {key}: {metric_means[key]:.6f}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
