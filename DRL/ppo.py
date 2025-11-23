from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    ppo_epochs: int
    minibatch_size: int


class RolloutBuffer:
    def __init__(self, gamma: float, gae_lambda: float) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self) -> None:
        self.observations: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.observations.append(obs)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def size(self) -> int:
        return len(self.actions)

    def compute_advantages(self) -> None:
        advantages = []
        returns = []
        next_value = 0.0
        next_advantage = 0.0
        for idx in reversed(range(self.size())):
            done = self.dones[idx]
            reward = self.rewards[idx]
            value = self.values[idx]
            mask = 0.0 if done else 1.0
            delta = reward + self.gamma * next_value * mask - value
            next_advantage = delta + self.gamma * self.gae_lambda * mask * next_advantage
            advantages.insert(0, next_advantage)
            returns.insert(0, next_advantage + value)
            next_value = value
            if done:
                next_value = 0.0
                next_advantage = 0.0
        self.advantages = advantages
        self.returns = returns

    def to_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        obs = torch.tensor(np.array(self.observations), dtype=torch.float32, device=device)
        masks = torch.tensor(np.array(self.masks), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        return {
            "observations": obs,
            "masks": masks,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "returns": returns,
            "advantages": advantages,
            "values": values,
        }


class PPOAgent:
    def __init__(self, model: nn.Module, config: PPOConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.cfg = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def act(self, observation: np.ndarray, mask: np.ndarray) -> Tuple[int, float, float]:
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(obs_tensor)
        logits = logits.masked_fill(mask_tensor == 0, -1e9)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.squeeze().item())

    def evaluate(self, obs: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.model(obs)
        logits = logits.masked_fill(masks == 0, -1e9)
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, entropy, values

    def update(self, rollout: RolloutBuffer) -> Dict[str, float]:
        if rollout.size() == 0:
            return {"loss": 0.0}
        rollout.compute_advantages()
        batch = rollout.to_tensors(self.device)
        num_samples = rollout.size()
        minibatch_size = min(self.cfg.minibatch_size, num_samples)
        indices = np.arange(num_samples)
        metrics_accum: Dict[str, float] = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        total_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                obs_mb = batch["observations"][mb_idx]
                mask_mb = batch["masks"][mb_idx]
                actions_mb = batch["actions"][mb_idx]
                old_log_probs_mb = batch["old_log_probs"][mb_idx]
                returns_mb = batch["returns"][mb_idx]
                advantages_mb = batch["advantages"][mb_idx]

                log_probs, entropy, values = self.evaluate(obs_mb, mask_mb, actions_mb)
                value_preds = values.reshape(-1)
                ratios = torch.exp(log_probs - old_log_probs_mb)
                surr1 = ratios * advantages_mb
                surr2 = torch.clamp(ratios, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value_preds, returns_mb)
                entropy_loss = entropy.mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                metrics_accum["policy_loss"] += float(policy_loss.item())
                metrics_accum["value_loss"] += float(value_loss.item())
                metrics_accum["entropy"] += float(entropy_loss.item())
                total_updates += 1

        if total_updates:
            for key in metrics_accum:
                metrics_accum[key] /= total_updates
        return metrics_accum

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)
