from typing import Callable, Dict, List, Any, Union, Iterable
from tensordict import TensorDict
import torch
from torchrl.data import TensorSpec, DiscreteTensorSpec
from omegaconf import DictConfig, OmegaConf
import logging
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate
from .base import Policy
from .common import (
    make_dataset_naive,
    make_ppo_ac,
    get_optimizer,
    make_critic,
    make_ppo_actor,
)
from gomoku_rl.utils.module import (
    count_parameters,
)
from gomoku_rl.utils.threat_detection import compute_threat_boost

logger = logging.getLogger(__name__)


class PPO(Policy):
    def __init__(
        self,
        cfg: DictConfig,
        action_spec: DiscreteTensorSpec,
        observation_spec: TensorSpec,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        super().__init__(cfg, action_spec, observation_spec, device)
        self.cfg: DictConfig = cfg
        self.device: Union[str, torch.device] = device

        self.clip_param: float = cfg.clip_param
        self.ppo_epoch: int = int(cfg.ppo_epochs)

        self.entropy_coef: float = cfg.entropy_coef
        self.gae_gamma: float = cfg.gamma
        self.gae_lambda: float = cfg.gae_lambda
        self.average_gae: float = cfg.average_gae
        self.batch_size: int = int(cfg.batch_size)

        self.max_grad_norm: float = cfg.max_grad_norm
        
        # Strategic evaluation parameters
        # Disable by default during training for speed - enable only for evaluation
        self.use_threat_detection: bool = cfg.get("use_threat_detection", False)
        self.use_threat_detection_during_training: bool = cfg.get("use_threat_detection_during_training", False)
        self.threat_boost_temperature: float = cfg.get("threat_boost_temperature", 1.0)
        
        # Threat parameters
        self.double_three_boost: float = cfg.get("double_three_boost", 2.0)
        self.double_open_three_boost: float = cfg.get("double_open_three_boost", 3.0)
        self.open_four_boost: float = cfg.get("open_four_boost", 5.0)
        
        # Blocking parameters
        self.block_fork_boost: float = cfg.get("block_fork_boost", 4.0)
        self.block_open_four_boost: float = cfg.get("block_open_four_boost", 6.0)
        
        # Positional parameters
        self.center_boost: float = cfg.get("center_boost", 0.5)
        self.corner_boost: float = cfg.get("corner_boost", 0.3)
        self.edge_boost: float = cfg.get("edge_boost", 0.2)
        
        # Performance optimization
        self.max_actions_to_check: int = cfg.get("max_actions_to_check", 30)  # Optimized default
        
        # Debug settings
        self.debug_threat_detection: bool = cfg.get("debug_threat_detection", False)
        self.debug_step_interval: int = cfg.get("debug_step_interval", 100)  # Log every N steps
        self._step_count = 0
        
        if self.cfg.get("share_network"):
            actor_value_operator = make_ppo_ac(
                cfg, action_spec=action_spec, device=self.device
            )
            self.actor = actor_value_operator.get_policy_operator()
            self.critic = actor_value_operator.get_value_head()
        else:
            self.actor = make_ppo_actor(
                cfg=cfg, action_spec=action_spec, device=self.device
            )
            self.critic = make_critic(cfg=cfg, device=self.device)

        fake_input = observation_spec.zero()
        fake_input["action_mask"] = ~fake_input["action_mask"]
        with torch.no_grad():
            self.actor(fake_input)
            self.critic(fake_input)
        # print(f"actor params:{count_parameters(self.actor)}")
        # print(f"critic params:{count_parameters(self.critic)}")

        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type="smooth_l1",
        )
        # Set the correct tensor key for log probability
        self.loss_module.set_keys(sample_log_prob="action_log_prob")

        self.optim = get_optimizer(self.cfg.optimizer, self.loss_module.parameters())

    def __call__(self, tensordict: TensorDict):
        self._step_count += 1
        should_log = self.debug_threat_detection and (self._step_count % self.debug_step_interval == 0)
        
        # Only move to device if not already there
        if tensordict.device != self.device:
            tensordict = tensordict.to(self.device)
        actor_input = tensordict.select("observation", "action_mask", strict=False)
        
        # Get actor output (probabilities)
        actor_output: TensorDict = self.actor(actor_input)
        
        # Apply threat detection boost if enabled
        # Only run during evaluation unless explicitly enabled for training
        should_use_threat_detection = (
            self.use_threat_detection and 
            (self.use_threat_detection_during_training or not self.actor.training)
        )
        
        if should_log:
            logger.debug(f"[Step {self._step_count}] Threat detection enabled: {should_use_threat_detection}, "
                        f"Training mode: {self.actor.training}, "
                        f"use_threat_detection: {self.use_threat_detection}, "
                        f"use_threat_detection_during_training: {self.use_threat_detection_during_training}")
        
        if should_use_threat_detection:
            # Extract board state from observation
            observation = tensordict["observation"]  # (E, 3, B, B)
            action_mask = tensordict["action_mask"]  # (E, B*B)
            
            board_size = observation.shape[-1]
            num_envs = observation.shape[0]
            
            # Reconstruct board state: layer 0 = current player (1), layer 1 = opponent (-1)
            current_player_board = observation[:, 0, :, :]  # (E, B, B)
            opponent_board = observation[:, 1, :, :]  # (E, B, B)
            
            # Create full board: 1 for current player, -1 for opponent, 0 for empty
            board = torch.zeros(num_envs, board_size, board_size, device=self.device, dtype=torch.long)
            board = torch.where(current_player_board > 0.5, 1, board)
            board = torch.where(opponent_board > 0.5, -1, board)
            
            # Current player is always 1 (black) in the observation encoding
            # The observation always shows current player as layer 0
            player_piece = 1  # Current player piece
            
            if should_log:
                num_valid_actions = action_mask.sum(dim=1).float().mean().item()
                logger.debug(f"[Step {self._step_count}] Board state - "
                            f"Num envs: {num_envs}, Board size: {board_size}, "
                            f"Avg valid actions: {num_valid_actions:.1f}")
            
            # Compute comprehensive strategic boost
            strategic_boost = compute_threat_boost(
                board=board,
                action_mask=action_mask,
                player_piece=player_piece,
                board_size=board_size,
                device=self.device,
                double_three_boost=self.double_three_boost,
                double_open_three_boost=self.double_open_three_boost,
                open_four_boost=self.open_four_boost,
                block_fork_boost=self.block_fork_boost,
                block_open_four_boost=self.block_open_four_boost,
                center_boost=self.center_boost,
                corner_boost=self.corner_boost,
                edge_boost=self.edge_boost,
                max_actions_to_check=self.max_actions_to_check,
                debug=should_log,
                step_count=self._step_count,
            )
            
            if should_log:
                # Analyze boost statistics
                boost_stats = {
                    'max': strategic_boost.max().item(),
                    'mean': strategic_boost[action_mask].mean().item() if action_mask.any() else 0.0,
                    'non_zero': (strategic_boost > 0).sum().item(),
                    'total_actions': action_mask.sum().item(),
                }
                logger.debug(f"[Step {self._step_count}] Strategic boost stats - "
                            f"Max: {boost_stats['max']:.2f}, "
                            f"Mean (valid): {boost_stats['mean']:.4f}, "
                            f"Non-zero boosts: {boost_stats['non_zero']}/{boost_stats['total_actions']}")
            
            # Modify probabilities by converting to logits, applying boost, and converting back
            if "probs" in actor_output.keys():
                probs = actor_output["probs"]  # (E, B*B)
                
                if should_log:
                    # Log top actions before boost
                    top_k = 5
                    for env_idx in range(min(2, num_envs)):  # Log first 2 environments
                        env_probs = probs[env_idx]
                        env_mask = action_mask[env_idx]
                        valid_probs = env_probs[env_mask]
                        valid_indices = torch.where(env_mask)[0]
                        top_probs, top_indices = torch.topk(valid_probs, min(top_k, len(valid_probs)))
                        top_actions = valid_indices[top_indices]
                        logger.debug(f"[Step {self._step_count}] Env {env_idx} - Top {len(top_probs)} actions BEFORE boost:")
                        for i, (action, prob) in enumerate(zip(top_actions, top_probs)):
                            r, c = action.item() // board_size, action.item() % board_size
                            boost_val = strategic_boost[env_idx, action.item()].item()
                            logger.debug(f"  {i+1}. Action ({r},{c}) - Prob: {prob.item():.4f}, Boost: {boost_val:.2f}")
                
                # Convert to logits (add small epsilon to avoid log(0))
                logits = torch.log(probs + 1e-8)
                # Apply strategic boost (divide by temperature to control strength)
                logits = logits + strategic_boost / self.threat_boost_temperature
                # Re-normalize with action mask (invalid actions should have -inf)
                logits = torch.where(action_mask, logits, torch.tensor(-float('inf'), device=self.device))
                # Convert back to probabilities
                probs_boosted = torch.softmax(logits, dim=-1)
                
                if should_log:
                    # Log top actions after boost
                    for env_idx in range(min(2, num_envs)):  # Log first 2 environments
                        env_probs = probs_boosted[env_idx]
                        env_mask = action_mask[env_idx]
                        valid_probs = env_probs[env_mask]
                        valid_indices = torch.where(env_mask)[0]
                        top_probs, top_indices = torch.topk(valid_probs, min(top_k, len(valid_probs)))
                        top_actions = valid_indices[top_indices]
                        logger.debug(f"[Step {self._step_count}] Env {env_idx} - Top {len(top_probs)} actions AFTER boost:")
                        for i, (action, prob) in enumerate(zip(top_actions, top_probs)):
                            r, c = action.item() // board_size, action.item() % board_size
                            boost_val = strategic_boost[env_idx, action.item()].item()
                            prob_before = probs[env_idx, action.item()].item()
                            logger.debug(f"  {i+1}. Action ({r},{c}) - Prob: {prob_before:.4f} -> {prob.item():.4f}, Boost: {boost_val:.2f}")
                
                # Update the actor output
                actor_output["probs"] = probs_boosted
        
        actor_output = actor_output.exclude("probs")
        tensordict.update(actor_output)

        # share_network=True, use `hidden` as input
        # share_network=False, use `observation` as input
        critic_input = tensordict.select("hidden", "observation", strict=False)
        critic_output = self.critic(critic_input)
        tensordict.update(critic_output)

        return tensordict

    def learn(self, data: TensorDict):
        should_log = self.debug_threat_detection and (self._step_count % self.debug_step_interval == 0)
        
        if should_log:
            logger.debug(f"[Step {self._step_count}] Learning - Data shape: {data.shape}, "
                        f"Device: {data.device}, Policy device: {self.device}")
        
        # Move entire data tensor to device once (if not already there)
        if data.device != self.device:
            data = data.to(self.device)
        
        # Extract tensors (already on device now)
        value = data["state_value"]
        next_value = data["next", "state_value"]
        done = data["next", "done"].unsqueeze(-1)
        reward = data["next", "reward"]
        
        with torch.no_grad():
            adv, value_target = vec_generalized_advantage_estimate(
                self.gae_gamma,
                self.gae_lambda,
                value,
                next_value,
                reward,
                done=done,
                terminated=done,
                time_dim=data.ndim - 1,
            )
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-4)
            if self.average_gae:
                adv = adv - loc
                adv = adv / scale

            data.set("advantage", adv)
            data.set("value_target", value_target)

        # filter out invalid white transitions
        invalid = data.get("invalid", None)
        if invalid is not None:
            num_invalid = invalid.sum().item()
            if should_log:
                logger.debug(f"[Step {self._step_count}] Learning - Filtered {num_invalid} invalid transitions")
            data = data[~invalid]

        data = data.reshape(-1)
        
        if should_log:
            logger.debug(f"[Step {self._step_count}] Learning - After filtering: {data.shape[0]} transitions, "
                        f"PPO epochs: {self.ppo_epoch}, Batch size: {self.batch_size}")

        self.train()
        loss_objectives = []
        loss_critics = []
        loss_entropies = []
        losses = []
        grad_norms = []
        for epoch in range(self.ppo_epoch):
            if should_log and epoch == 0:
                logger.debug(f"[Step {self._step_count}] Learning - Starting PPO epoch {epoch+1}/{self.ppo_epoch}")
            for minibatch in make_dataset_naive(
                data, batch_size=self.batch_size
            ):
                # minibatch is already on device (it's a view/slice of data)
                loss_vals = self.loss_module(minibatch)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                # Store values without unnecessary cloning during training
                loss_objectives.append(loss_vals["loss_objective"].detach())
                loss_critics.append(loss_vals["loss_critic"].detach())
                loss_entropies.append(loss_vals["loss_entropy"].detach())
                losses.append(loss_value.detach())
                # Optimization: backward, grad clipping and optim step
                loss_value.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(), self.max_grad_norm
                )
                grad_norms.append(grad_norm.detach())
                self.optim.step()
                self.optim.zero_grad()

        self.eval()
        # Batch all .item() calls together to minimize CPU-GPU synchronization overhead
        # Stack tensors first, then compute mean, then item() - more efficient
        grad_norm_mean = torch.stack(grad_norms).mean()
        loss_mean = torch.stack(losses).mean()
        loss_obj_mean = torch.stack(loss_objectives).mean()
        loss_crit_mean = torch.stack(loss_critics).mean()
        loss_ent_mean = torch.stack(loss_entropies).mean()
        
        result = {
            "advantage_meam": loc.item(),
            "advantage_std": scale.item(),
            "grad_norm": grad_norm_mean.item(),
            "loss": loss_mean.item(),
            "loss_objective": loss_obj_mean.item(),
            "loss_critic": loss_crit_mean.item(),
            "loss_entropy": loss_ent_mean.item(),
        }
        
        if should_log:
            logger.debug(f"[Step {self._step_count}] Learning - Completed. "
                        f"Loss: {result['loss']:.4f} (obj: {result['loss_objective']:.4f}, "
                        f"crit: {result['loss_critic']:.4f}, ent: {result['loss_entropy']:.4f}), "
                        f"Grad norm: {result['grad_norm']:.4f}, "
                        f"Advantage mean: {result['advantage_meam']:.4f}, std: {result['advantage_std']:.4f}")
        
        return result

    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        self.critic.load_state_dict(state_dict["critic"], strict=False)
        self.actor.load_state_dict(state_dict["actor"])

        self.loss_module = ClipPPOLoss(
            actor=self.actor,
            critic=self.critic,
            clip_epsilon=self.clip_param,
            entropy_bonus=bool(self.entropy_coef),
            entropy_coef=self.entropy_coef,
            normalize_advantage=self.cfg.get("normalize_advantage", True),
            loss_critic_type="smooth_l1",
        )
        # Set the correct tensor key for log probability
        self.loss_module.set_keys(sample_log_prob="action_log_prob")

        self.optim = get_optimizer(self.cfg.optimizer, self.loss_module.parameters())

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
