"""PPO runner for a frozen baseline policy plus a residual policy."""

import hashlib
import os
import statistics
import time
from collections import deque

import torch

from algo.ppo_onestep.module import ActorCritic
from algo.ppo_onestep.storage import RolloutStorage
from residual_tilt_grasp.tilted_ppo import TiltedPPO


class ResidualRolloutStorage(RolloutStorage):
    def __init__(
        self,
        *args,
        per_angle_advantage=True,
        advantage_eps=1e-8,
        advantage_min_samples=32,
        inactive_angle_ids=(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.angle_ids = torch.zeros(
            self.num_transitions_per_env,
            self.num_envs,
            1,
            dtype=torch.long,
            device=self.device,
        )
        self.per_angle_advantage = per_angle_advantage
        self.advantage_eps = float(advantage_eps)
        self.advantage_min_samples = int(advantage_min_samples)
        self.inactive_angle_ids = {
            int(angle_id) for angle_id in inactive_angle_ids
        }

    def add_residual_transition(
        self,
        observations,
        states,
        actions,
        rewards,
        dones,
        values,
        actions_log_prob,
        mu,
        sigma,
        angle_id,
    ):
        step = self.step
        super().add_transitions(
            observations,
            states,
            actions,
            rewards,
            dones,
            values,
            actions_log_prob,
            mu,
            sigma,
        )
        if torch.is_tensor(angle_id):
            angle_tensor = angle_id.to(
                device=self.device, dtype=torch.long
            ).view(self.num_envs, 1)
            self.angle_ids[step].copy_(angle_tensor)
        else:
            self.angle_ids[step].fill_(int(angle_id))

    def compute_returns(self, last_values=None, gamma=None, lam=None):
        self.returns.copy_(self.rewards)
        raw_advantages = self.rewards - self.values
        if not self.per_angle_advantage:
            self.advantages.copy_(
                (raw_advantages - raw_advantages.mean())
                / (raw_advantages.std(unbiased=False) + self.advantage_eps)
            )
            return

        self.advantages.zero_()
        active_advantages = self.advantages[:self.step]
        for angle_id in torch.unique(self.angle_ids[:self.step]):
            mask = self.angle_ids[:self.step] == angle_id
            values = raw_advantages[:self.step][mask]
            if int(angle_id.item()) in self.inactive_angle_ids:
                normalized = torch.zeros_like(values)
            elif values.numel() < self.advantage_min_samples:
                normalized = values
            else:
                normalized = (
                    values - values.mean()
                ) / (values.std(unbiased=False) + self.advantage_eps)
            active_advantages[mask] = normalized


class ResidualPPO(TiltedPPO):
    """Train only residual actions while a legacy hand policy stays frozen."""

    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        *args,
        baseline_checkpoint,
        residual_metadata,
        per_angle_advantage=True,
        advantage_eps=1e-8,
        advantage_min_samples=32,
        zero_inactive_angle_advantage=True,
        freeze_residual_backbone=True,
        initialize_residual_features=True,
        baseline_checkpoint_hash_override=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.baseline_checkpoint = os.path.abspath(baseline_checkpoint)
        self.residual_metadata = dict(residual_metadata)
        self.baseline_checkpoint_hash = (
            str(baseline_checkpoint_hash_override)
            if baseline_checkpoint_hash_override
            else self._sha256(self.baseline_checkpoint)
        )
        baseline_payload = torch.load(
            self.baseline_checkpoint, map_location=self.device
        )
        baseline_state_dict = self._extract_model_state_dict(
            baseline_payload, "Baseline"
        )

        legacy_baseline_obs_dim = (
            self.observation_space.shape[0]
            - (3 if "worldtilt" in self.vec_env.obs_type.split("+") else 0)
        )
        baseline_obs_dim = legacy_baseline_obs_dim
        first_weight = baseline_state_dict.get("actor_mean.0.weight")
        if torch.is_tensor(first_weight):
            baseline_state_obs_dim = int(first_weight.shape[1])
            if self.is_vision:
                pc_dim = int(
                    self.model_cfg["pc_shape"][0]
                    * self.model_cfg["pc_shape"][1]
                )
                pc_emb_dim = int(self.model_cfg["pc_emb_dim"])
                baseline_obs_dim = (
                    baseline_state_obs_dim - pc_emb_dim + pc_dim
                )
            else:
                baseline_obs_dim = baseline_state_obs_dim
        self.baseline_uses_residual_observation = (
            baseline_obs_dim == self.observation_space.shape[0]
        )
        self.vec_env.baseline_uses_residual_observation = (
            self.baseline_uses_residual_observation
        )
        baseline_action_dim = (
            getattr(self.vec_env, "policy_action_dim", None)
            or self.vec_env.num_active_hand_dofs
        )
        baseline_action_shape = (baseline_action_dim,)
        self.baseline_actor = ActorCritic(
            (baseline_obs_dim,),
            self.state_space.shape,
            baseline_action_shape,
            self.init_noise_std,
            self.model_cfg,
            asymmetric=self.asymmetric,
            use_pcl=self.is_vision,
        ).to(self.device)
        self._load_baseline_state_dict(baseline_state_dict)
        if initialize_residual_features:
            self._initialize_residual_features_from_baseline()
        self._residual_backbone_frozen = bool(
            freeze_residual_backbone
            and self.actor_critic.backbone is not None
            and self.baseline_actor.backbone is not None
        )
        if self._residual_backbone_frozen:
            self.actor_critic.backbone.load_state_dict(
                self.baseline_actor.backbone.state_dict()
            )
            for parameter in self.actor_critic.backbone.parameters():
                parameter.requires_grad_(False)
            self.actor_critic.backbone.eval()
            print(
                "Initialized residual PointNet from the baseline and froze it"
            )
        self._print_initial_exploration_scale()

        self.angle_to_id = {
            float(angle): idx
            for idx, angle in enumerate(self.vec_env.world_tilt_angles)
        }
        self.storage = ResidualRolloutStorage(
            self.vec_env.num_envs,
            self.num_transitions_per_env,
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            self.device,
            self.sampler,
            per_angle_advantage=per_angle_advantage,
            advantage_eps=advantage_eps,
            advantage_min_samples=advantage_min_samples,
            inactive_angle_ids=(
                [
                    angle_id
                    for angle, angle_id in self.angle_to_id.items()
                    if not self.vec_env.residual_is_active_for_angle(angle)
                ]
                if zero_inactive_angle_advantage
                else []
            ),
        )

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _extract_model_state_dict(payload, checkpoint_kind):
        if not isinstance(payload, dict):
            raise ValueError(
                f"{checkpoint_kind} checkpoint format is invalid"
            )
        for key in ("model_state_dict", "state_dict"):
            state_dict = payload.get(key)
            if isinstance(state_dict, dict):
                return state_dict
        if payload and all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in payload.items()
        ):
            return payload
        raise ValueError(
            f"{checkpoint_kind} checkpoint does not contain model weights"
        )

    def _load_baseline_state_dict(self, state_dict):
        target_state = self.baseline_actor.state_dict()
        migrated = {}
        skipped = []
        policy_indices = getattr(
            self.vec_env, "policy_hand_action_indices", None
        )
        full_action_dim = getattr(self.vec_env, "num_active_hand_dofs", None)
        policy_action_dim = (
            getattr(self.vec_env, "policy_action_dim", None)
            or full_action_dim
        )

        for key, source in state_dict.items():
            target = target_state.get(key)
            if target is None:
                skipped.append(key)
                continue
            if source.shape == target.shape:
                migrated[key] = source
                continue
            if (
                policy_indices is not None
                and full_action_dim is not None
                and source.shape
                and source.shape[0] == full_action_dim
                and target.shape
                and target.shape[0] == policy_action_dim
            ):
                indices = torch.tensor(
                    policy_indices,
                    dtype=torch.long,
                    device=source.device,
                )
                migrated[key] = source.index_select(0, indices)
                print(
                    f"Removed structural wrist entries from baseline {key}: "
                    f"{tuple(source.shape)} -> {tuple(target.shape)}"
                )
                continue
            skipped.append(
                f"{key}: checkpoint {tuple(source.shape)}, "
                f"current {tuple(target.shape)}"
            )

        missing, unexpected = self.baseline_actor.load_state_dict(
            migrated, strict=False
        )
        if skipped:
            print("Skipped baseline checkpoint entries:", skipped)
        if unexpected:
            print("Unexpected baseline checkpoint entries:", unexpected)
        if missing:
            raise RuntimeError(
                "Baseline checkpoint is incompatible; missing parameters: "
                + ", ".join(missing)
            )
        self.baseline_actor.eval()
        for parameter in self.baseline_actor.parameters():
            parameter.requires_grad_(False)
        print(
            f"Loaded frozen baseline policy from {self.baseline_checkpoint} "
            f"(sha256={self.baseline_checkpoint_hash[:12]}, "
            f"obs_dim={self.baseline_actor.actor_mean[0].in_features})"
        )

    def _expand_input_weight(self, source, target):
        if (
            source.ndim != 2
            or target.ndim != 2
            or source.shape[0] != target.shape[0]
            or target.shape[1] != source.shape[1] + 3
        ):
            return None
        split = source.shape[1] - int(self.actor_critic.pc_emb_dim)
        expanded = target.detach().clone()
        expanded[:, :split].copy_(source[:, :split])
        expanded[:, split:split + 3].zero_()
        expanded[:, split + 3:].copy_(source[:, split:])
        return expanded

    def _initialize_residual_features_from_baseline(self):
        """Reuse the pretrained representation while keeping zero output."""
        baseline = self.baseline_actor.state_dict()
        residual = self.actor_critic.state_dict()
        migrated = {}

        for key, target in residual.items():
            source = baseline.get(key)
            if source is None:
                continue
            if key == "log_std":
                # Residual exploration has its own configured scale.
                continue
            if source.shape == target.shape:
                migrated[key] = source
                continue
            if key in ("actor_mean.0.weight", "critic.0.weight"):
                expanded = self._expand_input_weight(source, target)
                if expanded is not None:
                    migrated[key] = expanded

        self.actor_critic.load_state_dict(migrated, strict=False)
        output_layer = self.actor_critic.actor_mean[-1]
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.zero_()
        print(
            "Initialized residual actor features and critic from the "
            "baseline; kept the residual output head at zero"
        )

    def _print_initial_exploration_scale(self):
        std = self.actor_critic.log_std.exp().detach()
        if self.vec_env.residual_mode in ("wrist", "hybrid"):
            wrist_std = std[:6]
            translation = (
                wrist_std[:3]
                * self.vec_env.residual_translation_scale
            )
            rotation_deg = torch.rad2deg(
                wrist_std[3:6]
                * self.vec_env.residual_rotation_scale
            )
            print(
                "Initial wrist exploration (1 std, before tilt modulation): "
                f"translation={translation.tolist()}m, "
                f"rotation={rotation_deg.tolist()}deg"
            )
        if self.vec_env.residual_mode in ("finger", "hybrid"):
            raw_finger_std = (
                std
                if self.vec_env.residual_mode == "finger"
                else std[6:]
            )
            expanded_finger_std = self.vec_env._expand_policy_hand_actions(
                raw_finger_std.unsqueeze(0)
            ).squeeze(0)
            finger_std = expanded_finger_std * self.vec_env.finger_residual_scale
            print(
                "Initial finger exploration (1 std, before tilt modulation): "
                f"mean={finger_std.mean().item():.4f}rad, "
                f"min={finger_std.min().item():.4f}rad, "
                f"max={finger_std.max().item():.4f}rad"
            )

    def _baseline_actions(self, residual_obs, states):
        baseline_obs = self.vec_env.get_baseline_observation(residual_obs)
        with torch.no_grad():
            return self.baseline_actor(
                baseline_obs, states, inference=True
            )

    def run(self):
        if self.is_testing:
            return self._run_test()

        rewbuffer = deque(maxlen=max(self.vec_env.num_envs, 1))
        lenbuffer = deque(maxlen=max(self.vec_env.num_envs, 1))
        cur_reward_sum = torch.zeros(
            self.vec_env.num_envs, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.vec_env.num_envs, device=self.device
        )
        env_ids = torch.arange(self.vec_env.num_envs, device=self.device)

        for it in range(
            self.current_learning_iteration, self.num_learning_iterations
        ):
            collection_start = time.time()
            reward_sum = []
            episode_length = []
            ep_infos = []
            residual_action_magnitudes = []
            residual_mean_magnitudes = []
            wrist_translation_magnitudes = []
            finger_delta_magnitudes = []

            for _ in range(self.num_transitions_per_env):
                current_obs = self.vec_env.reset_idx(env_ids)["obs"]
                current_states = self.vec_env.get_state()
                baseline_actions = self._baseline_actions(
                    current_obs, current_states
                )
                # PPO recomputes the graph from stored observations in update().
                # Avoid retaining unnecessary rollout-time activation buffers.
                with torch.no_grad():
                    (
                        residual_actions,
                        actions_log_prob,
                        values,
                        mu,
                        sigma,
                    ) = self.actor_critic(current_obs, current_states)
                self.vec_env.generate_residual_reaching_plan_idx(
                    env_ids,
                    baseline_actions=baseline_actions,
                    residual_actions=residual_actions,
                )
                residual_action_magnitudes.append(
                    residual_actions.abs().mean().item()
                )
                residual_mean_magnitudes.append(mu.abs().mean().item())
                wrist_translation_magnitudes.append(
                    self.vec_env.last_wrist_translation.norm(
                        dim=-1
                    ).mean().item()
                )
                finger_delta_magnitudes.append(
                    self.vec_env.last_finger_delta.abs().mean().item()
                )

                dones = torch.ones(
                    self.vec_env.num_envs, device=self.device
                )
                for step in range(self.vec_env.max_episode_length):
                    env_action = self.vec_env.compute_reference_actions()
                    _, _, _, extras = self.vec_env.step(env_action)
                    if step == self.vec_env.max_episode_length - 2:
                        rewards = self.vec_env.successes.clone().to(
                            self.device
                        )
                        rewards = torch.where(
                            self.vec_env.has_hit_table,
                            torch.zeros_like(rewards),
                            rewards,
                        )
                        break

                angle_ids = self.vec_env.env_angle_ids.view(
                    self.vec_env.num_envs, 1
                )
                self.storage.add_residual_transition(
                    current_obs,
                    current_states,
                    residual_actions,
                    rewards,
                    dones,
                    values,
                    actions_log_prob,
                    mu,
                    sigma,
                    angle_ids,
                )

                cur_reward_sum += rewards
                cur_episode_length += 1
                reward_sum.extend(cur_reward_sum.cpu().tolist())
                episode_length.extend(cur_episode_length.cpu().tolist())
                cur_reward_sum.zero_()
                cur_episode_length.zero_()

            rewbuffer.clear()
            lenbuffer.clear()
            rewbuffer.extend(reward_sum)
            lenbuffer.extend(episode_length)
            collection_time = time.time() - collection_start

            mean_trajectory_length, mean_reward = (
                self.storage.get_statistics()
            )
            current_success_rate = mean_reward.item()
            angle_metrics = self.vec_env.latest_angle_metrics.values()
            current_hit_table_rate = (
                statistics.mean(metric["hit"] for metric in angle_metrics)
                if self.vec_env.latest_angle_metrics
                else self.vec_env.has_hit_table.float().mean().item()
            )
            mean_residual_action = statistics.mean(
                residual_action_magnitudes
            )
            mean_residual_policy_mean = statistics.mean(
                residual_mean_magnitudes
            )
            mean_wrist_translation = statistics.mean(
                wrist_translation_magnitudes
            )
            mean_finger_delta = statistics.mean(finger_delta_magnitudes)

            learning_start = time.time()
            self.storage.compute_returns()
            mean_value_loss, mean_surrogate_loss = self.update()
            residual_head_norm = (
                self.actor_critic.actor_mean[-1].weight.norm().item()
            )
            self.storage.clear()
            learn_time = time.time() - learning_start
            self.current_learning_iteration = it + 1

            if self.print_log:
                num_learning_iterations = self.num_learning_iterations
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(
                    os.path.join(self.log_dir, f"model_{it}.pt")
                )

        self.save(
            os.path.join(
                self.log_dir, f"model_{self.num_learning_iterations}.pt"
            )
        )

    def log(self, locs, width=80, pad=35):
        super().log(locs, width=width, pad=pad)
        self.writer.add_scalar(
            "Residual/raw_action_abs_mean",
            locs["mean_residual_action"],
            locs["it"],
        )
        self.writer.add_scalar(
            "Residual/policy_mean_abs",
            locs["mean_residual_policy_mean"],
            locs["it"],
        )
        self.writer.add_scalar(
            "Residual/output_head_weight_norm",
            locs["residual_head_norm"],
            locs["it"],
        )
        self.writer.add_scalar(
            "Residual/wrist_translation_norm_m",
            locs["mean_wrist_translation"],
            locs["it"],
        )
        self.writer.add_scalar(
            "Residual/finger_delta_abs_rad",
            locs["mean_finger_delta"],
            locs["it"],
        )
        print(
            "Residual magnitude: "
            f"sampled={locs['mean_residual_action']:.4f}, "
            f"mean={locs['mean_residual_policy_mean']:.4f}, "
            f"head_norm={locs['residual_head_norm']:.4f}, "
            f"wrist={locs['mean_wrist_translation']:.4f}m, "
            f"finger={locs['mean_finger_delta']:.4f}rad"
        )

    def update(self):
        """Run every configured epoch with a fresh minibatch iterator."""
        if self._residual_backbone_frozen:
            self.actor_critic.backbone.eval()
        configured_epochs = self.num_learning_epochs
        self.num_learning_epochs = 1
        value_loss = 0.0
        surrogate_loss = 0.0
        try:
            for _ in range(configured_epochs):
                epoch_value_loss, epoch_surrogate_loss = super().update()
                value_loss += epoch_value_loss
                surrogate_loss += epoch_surrogate_loss
        finally:
            self.num_learning_epochs = configured_epochs
        return (
            value_loss / configured_epochs,
            surrogate_loss / configured_epochs,
        )

    def _run_test(self):
        env_ids = torch.arange(self.vec_env.num_envs, device=self.device)
        num_rounds = 10
        success_rates = []
        angle_success = {}
        for round_idx in range(num_rounds):
            obs = self.vec_env.reset_idx(env_ids)["obs"]
            states = self.vec_env.get_state()
            baseline_actions = self._baseline_actions(obs, states)
            residual_actions = self.actor_critic(
                obs, states, inference=True
            )
            self.vec_env.generate_residual_reaching_plan_idx(
                env_ids, baseline_actions, residual_actions
            )
            for step in range(self.vec_env.max_episode_length):
                env_action = self.vec_env.compute_reference_actions()
                self.vec_env.step(env_action)
                if step == self.vec_env.max_episode_length - 2:
                    effective_success = torch.where(
                        self.vec_env.has_hit_table,
                        torch.zeros_like(self.vec_env.successes),
                        self.vec_env.successes,
                    )
                    if hasattr(self.vec_env, "record_pregrasp_trial_success"):
                        self.vec_env.record_pregrasp_trial_success(
                            env_ids, effective_success
                        )
                    success = effective_success.float().mean().item()
                    success_rates.append(success)
                    any_success = None
                    if hasattr(self.vec_env, "pregrasp_any_success_rate"):
                        any_success = self.vec_env.pregrasp_any_success_rate()
                    angle_rows = []
                    if hasattr(self.vec_env, "env_angle_ids"):
                        for angle_id in torch.unique(
                            self.vec_env.env_angle_ids
                        ):
                            mask = self.vec_env.env_angle_ids == angle_id
                            angle_index = int(angle_id.item())
                            angle = float(
                                self.vec_env.world_tilt_angles[angle_index]
                            )
                            if hasattr(self.vec_env, "format_angle_id"):
                                angle_label = self.vec_env.format_angle_id(
                                    angle_index
                                )
                            else:
                                angle_label = f"{angle:g}deg"
                            angle_rate = (
                                effective_success[mask].float().mean().item()
                            )
                            angle_fraction = (
                                mask.float().mean().item()
                            )
                            angle_success.setdefault(angle_label, []).append(
                                angle_rate
                            )
                            angle_rows.append(
                                f"{angle_label}={angle_rate:.3f} "
                                f"({angle_fraction * 100.0:.1f}%)"
                            )
                    print(
                        f"Residual test {round_idx + 1}/{num_rounds}: "
                        "trajectory-rot=mixed, "
                        f"success={success:.3f}"
                        + (
                            f", any_success={any_success:.3f}"
                            if any_success is not None
                            else ""
                        )
                        + (
                            " | " + ", ".join(angle_rows)
                            if angle_rows
                            else ""
                        )
                    )
                    break
        if success_rates:
            values = torch.tensor(success_rates, device=self.device)
            print(
                "Residual test summary: "
                f"rounds={len(success_rates)}, "
                f"mean={values.mean().item():.3f}, "
                f"std={values.std(unbiased=False).item():.3f}, "
                f"min={values.min().item():.3f}, "
                f"max={values.max().item():.3f}"
            )
            rows = []
            for angle_label, rates in sorted(angle_success.items()):
                if not rates:
                    continue
                angle_values = torch.tensor(rates, device=self.device)
                rows.append(
                    f"{angle_label}={angle_values.mean().item():.3f}"
                )
            if rows:
                print("Residual test per-angle mean: " + ", ".join(rows))

    def evaluate_initial_policy(self, rounds=1):
        """Evaluate deterministic zero-mean residuals before optimization."""
        env_ids = torch.arange(self.vec_env.num_envs, device=self.device)
        evaluations = max(int(rounds), 0)
        if evaluations == 0:
            return

        was_training = self.actor_critic.training
        self.actor_critic.eval()
        with torch.no_grad():
            for eval_idx in range(evaluations):
                obs = self.vec_env.reset_idx(env_ids)["obs"]
                states = self.vec_env.get_state()
                baseline_actions = self._baseline_actions(obs, states)
                residual_actions = self.actor_critic(
                    obs, states, inference=True
                )
                self.vec_env.generate_residual_reaching_plan_idx(
                    env_ids, baseline_actions, residual_actions
                )
                for step in range(self.vec_env.max_episode_length):
                    env_action = self.vec_env.compute_reference_actions()
                    self.vec_env.step(env_action)
                    if step == self.vec_env.max_episode_length - 2:
                        effective_success = torch.where(
                            self.vec_env.has_hit_table,
                            torch.zeros_like(self.vec_env.successes),
                            self.vec_env.successes,
                        )
                        print(
                            "Initial residual eval "
                            f"{eval_idx + 1}/{evaluations}: "
                            "trajectory-rot=mixed, "
                            "residual_mean="
                            f"{residual_actions.abs().mean().item():.6f}, "
                            "success="
                            f"{effective_success.float().mean().item():.3f}"
                        )
                        break
        if was_training:
            self.actor_critic.train()

    def load_residual(self, path, resume=False):
        payload = torch.load(path, map_location=self.device)
        state_dict = self._extract_model_state_dict(
            payload, "Residual"
        )
        checkpoint_data = payload if state_dict is not payload else {}
        saved_hash = checkpoint_data.get("baseline_checkpoint_hash")
        if saved_hash and saved_hash != self.baseline_checkpoint_hash:
            raise ValueError(
                "Residual checkpoint was trained with a different baseline"
            )
        saved_metadata = checkpoint_data.get("residual_metadata", {})
        compatibility_keys = (
            "mode",
            "action_dim",
            "frame",
            "scope",
            "pose_composition",
            "translation_scale",
            "rotation_scale_deg",
            "finger_alpha",
            "finger_min_ratio",
            "finger_max_ratio",
            "tilt_modulation",
            "tilt_modulation_max_angle",
            "wrist_tilt_modulation",
            "finger_tilt_modulation",
            "tilt_angles",
            "pointcloud_frame",
        )
        for key in compatibility_keys:
            saved_value = saved_metadata.get(key)
            current_value = self.residual_metadata.get(key)
            if (
                key == "tilt_angles"
                and saved_value is not None
                and current_value is not None
                and not resume
            ):
                saved_angles = {float(angle) for angle in saved_value}
                current_angles = {float(angle) for angle in current_value}
                if current_angles.issubset(saved_angles):
                    if current_angles != saved_angles:
                        print(
                            "Testing residual checkpoint on tilt-angle "
                            f"subset {current_value}; trained angles were "
                            f"{saved_value}"
                        )
                else:
                    print(
                        "Testing residual checkpoint on extrapolated "
                        f"tilt angles {current_value}; trained angles were "
                        f"{saved_value}. Success is not guaranteed."
                    )
                continue
            if (
                key == "tilt_modulation_max_angle"
                and saved_value is not None
                and current_value is not None
                and not resume
            ):
                if float(saved_value) != float(current_value):
                    print(
                        "Testing residual checkpoint with "
                        f"tilt_modulation_max_angle={current_value}; "
                        f"checkpoint was trained with {saved_value}."
                    )
                continue
            if saved_value is not None and saved_value != current_value:
                raise ValueError(
                    f"Residual checkpoint {key}={saved_value!r}, "
                    f"current configuration uses {current_value!r}"
                )
        self.actor_critic.load_state_dict(state_dict)
        if resume:
            if not (
                "optimizer_state_dict" in checkpoint_data
                and "iteration" in checkpoint_data
            ):
                raise ValueError(
                    "resume=True requires a full residual checkpoint with "
                    "optimizer_state_dict and iteration"
                )
            self.optimizer.load_state_dict(
                checkpoint_data["optimizer_state_dict"]
            )
            self.current_learning_iteration = int(
                checkpoint_data["iteration"]
            )
            self.tot_timesteps = int(
                checkpoint_data.get("tot_timesteps", 0)
            )
            self.tot_time = float(checkpoint_data.get("tot_time", 0.0))
        else:
            self.current_learning_iteration = 0
        print(f"Loaded residual checkpoint from {path}")

    def save(self, path):
        torch.save(
            {
                "checkpoint_version": self.CHECKPOINT_VERSION,
                "model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iteration": self.current_learning_iteration,
                "tot_timesteps": self.tot_timesteps,
                "tot_time": self.tot_time,
                "baseline_checkpoint": self.baseline_checkpoint,
                "baseline_checkpoint_hash": self.baseline_checkpoint_hash,
                "baseline_model_state_dict": self.baseline_actor.state_dict(),
                "residual_metadata": self.residual_metadata,
            },
            path,
        )
