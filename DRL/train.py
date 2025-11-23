from __future__ import annotations

import argparse
import json
import time
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from checkpoint import CheckpointManager, capture_rng_state, restore_rng_state
from env import GomokuEnv
from model import PolicyValueNet
from ppo import PPOAgent, PPOConfig, RolloutBuffer
from utils import ensure_dir, get_device, load_config, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for Gomoku")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Force resume from latest checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Start from scratch even if checkpoint exists")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint path to load")
    return parser.parse_args()


def build_agent(board_size: int, training_cfg: Dict, device: torch.device) -> PPOAgent:
    model = PolicyValueNet(board_size)
    ppo_cfg = PPOConfig(
        learning_rate=training_cfg["learning_rate"],
        gamma=training_cfg["gamma"],
        gae_lambda=training_cfg["gae_lambda"],
        clip_range=training_cfg["clip_range"],
        entropy_coef=training_cfg["entropy_coef"],
        value_coef=training_cfg["value_coef"],
        max_grad_norm=training_cfg["max_grad_norm"],
        ppo_epochs=training_cfg["ppo_epochs"],
        minibatch_size=training_cfg["minibatch_size"],
    )
    return PPOAgent(model, ppo_cfg, device)


def collect_self_play(env: GomokuEnv, agent: PPOAgent, target_games: int, show_progress: bool = False) -> Dict:
    buffer = RolloutBuffer(agent.cfg.gamma, agent.cfg.gae_lambda)
    stats = {"wins": 0, "losses": 0, "draws": 0}
    games_completed = 0
    iterator = range(target_games)
    if show_progress and target_games > 0:
        iterator = tqdm(iterator, desc="Self-play games", leave=False)

    for _ in iterator:
        obs = env.reset()
        mask = env.legal_actions_mask()
        done = False
        final_info: Dict = {}
        while not done:
            action, log_prob, value = agent.act(obs, mask)
            result = env.step(action)
            buffer.add(obs, mask, action, log_prob, value, result.reward, result.done)
            obs = result.observation
            mask = env.legal_actions_mask()
            done = result.done
            final_info = result.info
        winner = final_info.get("winner")
        if winner == 0:
            stats["draws"] += 1
        elif winner == 1:
            stats["wins"] += 1
        elif winner == -1:
            stats["losses"] += 1
        games_completed += 1
    return {"buffer": buffer, "stats": stats}


def evaluate_vs_random(env: GomokuEnv, agent: PPOAgent, games: int, show_progress: bool = False) -> Dict[str, float]:
    wins = losses = draws = 0
    iterator = range(games)
    if show_progress and games > 0:
        iterator = tqdm(iterator, desc="Eval vs Random", leave=False)
    for _ in iterator:
        obs = env.reset()
        done = False
        final_info: Dict = {}
        while not done:
            mask = env.legal_actions_mask()
            if env.current_player == 1:
                action, _, _ = agent.act(obs, mask)
            else:
                legal_indices = np.flatnonzero(mask)
                action = int(np.random.choice(legal_indices))
            result = env.step(action)
            obs = result.observation
            done = result.done
            final_info = result.info
        winner = final_info.get("winner")
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1
    total = max(games, 1)
    return {"win_rate": wins / total, "loss_rate": losses / total, "draw_rate": draws / total}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    board_size = int(config["board_size"])
    win_length = int(config["win_length"])
    training_cfg = config["training"]

    if training_cfg.get("use_seed", False):
        set_global_seed(int(training_cfg["seed"]))

    device = get_device(
        training_cfg.get("device"),
        training_cfg.get("max_vram_gb"),
    )
    env = GomokuEnv(board_size, win_length)
    agent = build_agent(board_size, training_cfg, device)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    ensure_dir(checkpoint_dir)
    manager = CheckpointManager(checkpoint_dir, training_cfg.get("max_checkpoints", 5))

    start_iter = 0
    best_score = float("-inf")

    ckpt_path = args.checkpoint
    auto_resume = not args.fresh
    if ckpt_path is None and auto_resume:
        ckpt_path = manager.latest_checkpoint()
    if ckpt_path is not None:
        print(f"Resuming from checkpoint: {ckpt_path}")
        state = manager.load(str(ckpt_path))
        agent.load_state_dict(state["model_state"])
        agent.optimizer.load_state_dict(state["optimizer_state"])
        restore_rng_state(state.get("rng_state"))
        start_iter = state.get("iteration", 0)
        best_score = state.get("best_score", best_score)
    elif args.resume:
        raise FileNotFoundError("--resume requested but no checkpoint found")

    total_iterations = int(training_cfg["total_iterations"])
    games_per_batch = int(training_cfg["games_per_batch"])
    checkpoint_interval = int(training_cfg["checkpoint_interval"])
    eval_games = int(training_cfg.get("eval_random_games", 10))
    log_interval = int(config.get("log_interval", 10))
    show_self_play_progress = bool(training_cfg.get("show_self_play_progress", False))
    show_eval_progress = bool(training_cfg.get("show_eval_progress", True))

    iteration_range = range(start_iter + 1, total_iterations + 1)
    progress_bar = tqdm(
        iteration_range,
        desc="Training",
        initial=start_iter,
        total=total_iterations,
        dynamic_ncols=True,
    )
    for iteration in progress_bar:
        progress_bar.set_description(f"Iter {iteration}/{total_iterations} · Self-play")
        rollout_data = collect_self_play(env, agent, games_per_batch, show_progress=show_self_play_progress)

        progress_bar.set_description(f"Iter {iteration}/{total_iterations} · PPO Update")
        metrics = agent.update(rollout_data["buffer"])

        if iteration % log_interval == 0:
            stats = rollout_data["stats"]
            payload = {
                "iter": iteration,
                "policy_loss": metrics.get("policy_loss", 0.0),
                "value_loss": metrics.get("value_loss", 0.0),
                "entropy": metrics.get("entropy", 0.0),
                "wins": stats["wins"],
                "losses": stats["losses"],
                "draws": stats["draws"],
            }
            tqdm.write(json.dumps(payload))
            progress_bar.set_postfix(
                loss=f"{payload['policy_loss']:.3f}",
                val=f"{payload['value_loss']:.3f}",
                ent=f"{payload['entropy']:.3f}",
                wins=payload["wins"],
                draws=payload["draws"],
            )

        if iteration % checkpoint_interval == 0 or iteration == total_iterations:
            progress_bar.set_description(f"Iter {iteration}/{total_iterations} · Eval & CKPT")
            eval_scores = evaluate_vs_random(env, agent, eval_games, show_progress=show_eval_progress)
            score = eval_scores["win_rate"]
            best_score = max(best_score, score)
            state = {
                "model_state": agent.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
                "config": config,
                "iteration": iteration,
                "score": score,
                "best_score": best_score,
                "rng_state": capture_rng_state(),
                "timestamp": time.time(),
            }
            path = manager.save(state, iteration=iteration, score=score)
            tqdm.write(f"Saved checkpoint to {path}")
            if score >= training_cfg.get("early_stop_win_rate", 1.1):
                tqdm.write("Early stopping criteria met.")
                break
        progress_bar.set_description(f"Iter {iteration}/{total_iterations} · Ready")


if __name__ == "__main__":
    main()
