"""PPO checkpoint support for tilted-world training."""

import math
import os
import re

import torch

from algo.ppo_onestep.ppo import PPO


class TiltedPPO(PPO):
    """Add full resume checkpoints and legacy 0-degree weight migration."""

    CHECKPOINT_VERSION = 1
    ACTION_HEAD_KEYS = ("log_std", "actor_mean.8.weight", "actor_mean.8.bias")

    def _expanded_input_weight(self, source, target, added_dims=3):
        if source.ndim != 2 or target.ndim != 2:
            return None
        if source.shape[0] != target.shape[0]:
            return None
        if target.shape[1] != source.shape[1] + added_dims:
            return None

        # ActorCritic replaces the trailing point cloud with a PointNet
        # embedding. worldtilt is inserted immediately before that embedding.
        pc_emb_dim = int(getattr(self.actor_critic, "pc_emb_dim", 0))
        split = source.shape[1] - pc_emb_dim
        expanded = target.clone()
        expanded[:, :split] = source[:, :split]
        expanded[:, split:split + added_dims] = 0.0
        expanded[:, split + added_dims:] = source[:, split:]
        return expanded

    def _load_model_state(
        self,
        state_dict,
        allow_observation_expansion,
        allow_action_head_mismatch=False,
    ):
        current = self.actor_critic.state_dict()
        migrated = {}
        skipped = []
        skipped_action_head = []

        for key, source in state_dict.items():
            if key not in current:
                skipped.append(key)
                continue
            target = current[key]
            if source.shape == target.shape:
                migrated[key] = source
                continue
            if allow_observation_expansion and key in (
                "actor_mean.0.weight",
                "critic.0.weight",
            ):
                expanded = self._expanded_input_weight(source, target)
                if expanded is not None:
                    migrated[key] = expanded
                    print(
                        f"Expanded {key}: {tuple(source.shape)} -> "
                        f"{tuple(target.shape)} with zero world-tilt columns"
                    )
                    continue
            if allow_action_head_mismatch and key in self.ACTION_HEAD_KEYS:
                skipped_action_head.append(
                    f"{key}: checkpoint {tuple(source.shape)}, "
                    f"current {tuple(target.shape)}"
                )
                continue
            skipped.append(
                f"{key}: checkpoint {tuple(source.shape)}, "
                f"current {tuple(target.shape)}"
            )

        missing, unexpected = self.actor_critic.load_state_dict(
            migrated, strict=False
        )
        if skipped:
            print("Skipped checkpoint entries:", skipped)
        if skipped_action_head:
            print(
                "Reinitialized action head entries for current hand:",
                skipped_action_head,
            )
        if unexpected:
            print("Unexpected checkpoint entries:", unexpected)
        allowed_missing = {"actor_mean.0.weight", "critic.0.weight"}
        if allow_action_head_mismatch:
            allowed_missing.update(self.ACTION_HEAD_KEYS)
        disallowed_missing = [
            key for key in missing
            if key not in allowed_missing
        ]
        if disallowed_missing:
            raise RuntimeError(
                "Checkpoint is incompatible; missing parameters: "
                + ", ".join(disallowed_missing)
            )

    def load_pretrained(self, path, is_testing=False, action_std=None):
        """Load model weights, allowing legacy observations without worldtilt."""
        payload = torch.load(path, map_location=self.device)
        state_dict = (
            payload["model_state_dict"]
            if isinstance(payload, dict) and "model_state_dict" in payload
            else payload
        )
        self._load_model_state(
            state_dict,
            allow_observation_expansion=True,
            allow_action_head_mismatch=not is_testing,
        )
        self.current_learning_iteration = 0
        if action_std is not None:
            action_std = float(action_std)
            if action_std <= 0:
                raise ValueError("pretrained_action_std must be positive")
            with torch.no_grad():
                self.actor_critic.log_std.fill_(math.log(action_std))
            print(f"Set pretrained policy action std to {action_std:g}")
        if is_testing:
            self.actor_critic.eval()
        else:
            self.actor_critic.train()
        print(f"Loaded pretrained model weights from {path}")

    def configure_pretrained_finetuning(
        self, learning_rate=3e-5, epochs=2, freeze_backbone=True
    ):
        """Protect a strong legacy policy from destructive first updates."""
        learning_rate = float(learning_rate)
        epochs = int(epochs)
        if learning_rate <= 0:
            raise ValueError("pretrained_learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("pretrained_epochs must be positive")

        self.step_size = learning_rate
        self.num_learning_epochs = epochs
        # The inherited adaptive KL uses squashed means as Gaussian means, so
        # it is not a reliable learning-rate controller for this policy.
        self.schedule = "fixed"
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        self._pretrained_backbone_frozen = bool(
            freeze_backbone and self.actor_critic.backbone is not None
        )
        if self._pretrained_backbone_frozen:
            for parameter in self.actor_critic.backbone.parameters():
                parameter.requires_grad_(False)
            # Keep BatchNorm running statistics fixed as well.
            self.actor_critic.backbone.eval()

        print(
            "Pretrained finetuning: "
            f"lr={learning_rate:g}, epochs={epochs}, "
            f"freeze_backbone={self._pretrained_backbone_frozen}, "
            "schedule=fixed"
        )

    def update(self):
        if getattr(self, "_pretrained_backbone_frozen", False):
            self.actor_critic.backbone.eval()
        return super().update()

    def log(self, locs, width=80, pad=35):
        metrics = self.vec_env.latest_angle_metrics
        successes = [
            values["success"] for values in metrics.values()
        ]
        hits = [values["hit"] for values in metrics.values()]
        if successes:
            # The inherited fields otherwise describe only the final angle
            # collected in this update.
            locs["current_success_rate"] = sum(successes) / len(successes)
            locs["current_hit_table_rate"] = sum(hits) / len(hits)
            # The original run loop's reward list is cumulative and its
            # fixed-size deque retains only the final angle of this update.
            locs["rewbuffer"].clear()
            locs["rewbuffer"].append(float(locs["mean_reward"]))
            locs["lenbuffer"].clear()
            locs["lenbuffer"].append(float(locs["mean_trajectory_length"]))

        super().log(locs, width=width, pad=pad)
        # Prevent the inherited temporary lists from growing for 20k updates.
        locs["reward_sum"].clear()
        locs["episode_length"].clear()
        if not metrics:
            return

        rows = []
        ordered_successes = []
        for angle in self.vec_env.world_tilt_angles:
            angle_metrics = metrics.get(float(angle))
            if angle_metrics is None:
                continue
            ordered_successes.append(angle_metrics["success"])
            rows.append(
                f"{angle:g}deg={angle_metrics['success']:.3f}"
                f" (hit={angle_metrics['hit']:.3f}, "
                f"clear={angle_metrics['min_clearance']:.4f}m)"
            )
            self.writer.add_scalar(
                f"Tilt/success_{angle:g}deg",
                angle_metrics["success"],
                locs["it"],
            )
            self.writer.add_scalar(
                f"Tilt/hit_{angle:g}deg",
                angle_metrics["hit"],
                locs["it"],
            )
        if ordered_successes:
            balanced_success = (
                sum(ordered_successes) / len(ordered_successes)
            )
            self.writer.add_scalar(
                "Tilt/balanced_success", balanced_success, locs["it"]
            )
            print(
                f"Balanced tilt success: {balanced_success:.3f} | "
                + " | ".join(rows)
            )

    def evaluate_pretrained(self, num_rounds=1):
        """Report deterministic and sampled success before PPO updates."""
        was_training = self.actor_critic.training
        self.actor_critic.eval()
        env_ids = torch.arange(
            self.vec_env.num_envs, device=self.vec_env.device
        )

        def evaluate(sample_actions):
            rates = []
            for round_idx in range(int(num_rounds)):
                current_obs = self.vec_env.reset_idx(env_ids)["obs"]
                current_states = self.vec_env.get_state()
                if sample_actions:
                    actions = self.actor_critic(
                        current_obs, current_states
                    )[0]
                else:
                    actions = self.actor_critic(
                        current_obs, current_states, inference=True
                    )
                self.vec_env.generate_reaching_plan_idx(
                    env_ids, actions=actions
                )
                success = None
                for step in range(self.vec_env.max_episode_length):
                    env_action = self.vec_env.compute_reference_actions()
                    self.vec_env.step(env_action)
                    if step == self.vec_env.max_episode_length - 2:
                        success = self.vec_env.successes.clone()
                if success is None:
                    success = self.vec_env.successes
                rate = success.float().mean().item()
                rates.append(rate)
                mode = "sampled" if sample_actions else "deterministic"
                print(
                    f"Pretrained {mode} eval {round_idx + 1}/"
                    f"{num_rounds}: success={rate:.3f}, "
                    "trajectory-rot=mixed"
                )
            return sum(rates) / len(rates)

        with torch.no_grad():
            deterministic_mean = evaluate(sample_actions=False)
            sampled_mean = evaluate(sample_actions=True)
        if was_training:
            self.actor_critic.train()
            if getattr(self, "_pretrained_backbone_frozen", False):
                self.actor_critic.backbone.eval()
        print(
            "Pretrained eval summary: "
            f"deterministic={deterministic_mean:.3f}, "
            f"sampled={sampled_mean:.3f}, "
            f"action_std={self.actor_critic.log_std.exp().mean().item():.3f}"
        )

    def load_resume(self, path):
        """Restore model, optimizer, counters, and the next iteration."""
        payload = torch.load(path, map_location=self.device)
        if not (
            isinstance(payload, dict)
            and "model_state_dict" in payload
            and "optimizer_state_dict" in payload
        ):
            raise ValueError(
                "Full resume requires a checkpoint produced by "
                "residual_tilt_grasp/tilted_ppo.py. Legacy model_*.pt files contain "
                "weights only; use +resume=False for those."
            )

        self.actor_critic.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.current_learning_iteration = int(payload["iteration"])
        self.tot_timesteps = int(payload.get("tot_timesteps", 0))
        self.tot_time = float(payload.get("tot_time", 0.0))
        self.step_size = float(payload.get("step_size", self.step_size))
        for group in self.optimizer.param_groups:
            group["lr"] = self.step_size
        self.actor_critic.train()
        print(
            f"Resumed full training state from {path} at iteration "
            f"{self.current_learning_iteration}"
        )

    def save(self, path):
        match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
        saved_iteration = (
            min(int(match.group(1)) + 1, self.num_learning_iterations)
            if match
            else self.current_learning_iteration
        )
        torch.save(
            {
                "checkpoint_version": self.CHECKPOINT_VERSION,
                "model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iteration": saved_iteration,
                "tot_timesteps": self.tot_timesteps,
                "tot_time": self.tot_time,
                "step_size": self.step_size,
            },
            path,
        )
