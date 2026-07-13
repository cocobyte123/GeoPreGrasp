"""Collect random object-level M/T/P pregrasp rollouts with SE residual PPO.

This collector is intentionally independent from the removed DGA bridge
pipeline.  It samples object-frame PregraspPrior bins during normal DemoGrasp
random object resets, executes the SE residual policy, and writes PPO rollout
``batch_*.npz`` files that can be consumed by
``build_Pregrasp_data.aggregate.build_field_store_from_rollouts``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ISAACGYM_ROOT = PROJECT_ROOT / "thirdparty" / "isaacgym" / "python"
ISAACGYM_ENVS_ROOT = PROJECT_ROOT / "thirdparty" / "IsaacGymEnvs"
DEFAULT_PREGRASP_ROOT = WORKSPACE_ROOT / "PregraspPrior"
DEFAULT_ROLLOUT_ROOT = DEFAULT_PREGRASP_ROOT / "data" / "rollouts" / "ppo_random_mtp"
DEFAULT_SUBDIVISION = 3
DEFAULT_TILT_PITCH = [0.0, 15.0, 30.0]

for path in (PROJECT_ROOT, ISAACGYM_ROOT, ISAACGYM_ENVS_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from isaacgymenvs.utils.torch_jit_utils import (  # noqa: E402
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)
import torch
from build_Pregrasp_data.io.rollout_schema import SCHEMA as ROLLOUT_SCHEMA  # noqa: E402
from residual_se_grasp.se_residual_grasp import SEResidualGrasp  # noqa: E402
from residual_se_grasp.train_se_reslearn import (  # noqa: E402
    TASK_NAME,
    _prepare_embedded_baseline_if_needed,
    configure_se_training,
    se_residual_metadata,
)
from residual_tilt_grasp.train_tilted_hand_only_reslearn import (  # noqa: E402
    get_checkpoint_paths,
    log_viewer_device,
)


def _as_float_list(value) -> List[float]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            import json as _json

            return [float(item) for item in _json.loads(text)]
        return [float(item) for item in text.split(",") if item.strip()]
    return [float(item) for item in value]


def _plain_config(value):
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return value


def _checkpoint_residual_metadata(path) -> Dict:
    path_text = str(path or "")
    if not path_text or not Path(path_text).expanduser().is_file():
        return {}
    try:
        payload = torch.load(Path(path_text).expanduser(), map_location="cpu")
    except Exception as exc:
        print(f"Warning: failed to read checkpoint metadata from {path_text}: {exc}", flush=True)
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("residual_metadata"), dict):
        return payload["residual_metadata"]
    return {}


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(eps)


def _shortest_arc_quat(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = _normalize(source)
    target = _normalize(target)
    dot = (source * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    cross = torch.cross(source, target, dim=-1)
    quat = torch.cat([cross, 1.0 + dot], dim=-1)

    opposite = dot.squeeze(-1) < -0.999
    if torch.any(opposite):
        fallback = torch.zeros_like(source)
        fallback[:, 0] = 1.0
        use_y = torch.abs(source[:, 0]) > 0.9
        fallback[use_y] = torch.tensor(
            [0.0, 1.0, 0.0],
            dtype=source.dtype,
            device=source.device,
        )
        axis = _normalize(torch.cross(source, fallback, dim=-1))
        quat[opposite, 0:3] = axis[opposite]
        quat[opposite, 3] = 0.0
    return _normalize(quat)


def _stable_tangent_axes(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = _normalize(direction)
    z_ref = torch.tensor([0.0, 0.0, 1.0], dtype=n.dtype, device=n.device).view(1, 3).expand_as(n)
    x_ref = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device).view(1, 3).expand_as(n)
    use_x = torch.abs((n * z_ref).sum(dim=-1, keepdim=True)) > 0.9
    ref = torch.where(use_x, x_ref, z_ref)
    tilt_axis = _normalize(torch.cross(ref, n, dim=-1))
    pitch_axis = _normalize(torch.cross(n, tilt_axis, dim=-1))
    return tilt_axis, pitch_axis


def _apply_local_angles(direction: torch.Tensor, tilt_deg: torch.Tensor, pitch_deg: torch.Tensor) -> torch.Tensor:
    tilt_axis, pitch_axis = _stable_tangent_axes(direction)
    tilt_quat = quat_from_angle_axis(torch.deg2rad(tilt_deg), tilt_axis)
    pitched_dir = quat_apply(tilt_quat, _normalize(direction))
    pitch_quat = quat_from_angle_axis(torch.deg2rad(pitch_deg), pitch_axis)
    return _normalize(quat_apply(pitch_quat, pitched_dir))


class RandomMTPPregraspCollectGrasp(SEResidualGrasp):
    """SE residual env that exports rollouts for PregraspPrior aggregation."""

    def init_configs(self, cfg):
        env_cfg = cfg["env"]
        self.collect_pregrasp_root = str(env_cfg.get("pregraspCollectPregraspRoot", DEFAULT_PREGRASP_ROOT))
        self.collect_rollout_root = str(env_cfg.get("pregraspCollectRolloutRoot", DEFAULT_ROLLOUT_ROOT))
        self.collect_overwrite_root = bool(env_cfg.get("pregraspCollectOverwriteRoot", True))
        self.collect_init_mode = str(env_cfg.get("pregraspCollectInitMode", "training"))
        self.collect_mtp_mode = str(env_cfg.get("pregraspCollectMtpMode", "cycle"))
        self.collect_shell_radius = str(env_cfg.get("pregraspCollectShellRadius", "base"))
        self.collect_sphere_template_kind = str(env_cfg.get("pregraspCollectSphereTemplate", "geodesic"))
        self.collect_subdivision = int(env_cfg.get("pregraspCollectSubdivision", DEFAULT_SUBDIVISION))
        self.collect_tilt_angles_cfg = _as_float_list(env_cfg.get("pregraspCollectTiltAngles", DEFAULT_TILT_PITCH))
        self.collect_pitch_angles_cfg = _as_float_list(env_cfg.get("pregraspCollectPitchAngles", DEFAULT_TILT_PITCH))
        if self.collect_init_mode not in ("training", "mtp"):
            raise ValueError("pregraspCollectInitMode must be 'training' or 'mtp'")
        if self.collect_mtp_mode not in ("cycle", "random"):
            raise ValueError("pregraspCollectMtpMode must be 'cycle' or 'random'")
        super().init_configs(cfg)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.collect_overwrite_root:
            self._clear_existing_rollout_files()
        pregrasp_root = Path(self.collect_pregrasp_root).expanduser().resolve()
        pregrasp_import_root = pregrasp_root / "src" if (pregrasp_root / "src").is_dir() else pregrasp_root
        if str(pregrasp_import_root) not in sys.path:
            sys.path.insert(0, str(pregrasp_import_root))
        from oc_pregrasp.field.sphere_template import make_sphere_template

        template = make_sphere_template(
            kind=self.collect_sphere_template_kind,
            subdivision=self.collect_subdivision,
            tilt_angles=self.collect_tilt_angles_cfg,
            pitch_angles=self.collect_pitch_angles_cfg,
        )
        self.collect_template_shape = tuple(int(v) for v in template.shape)
        self.collect_num_candidates = int(template.num_candidates)
        self.collect_sphere_dirs = torch.tensor(template.dirs, dtype=torch.float32, device=self.device)
        self.collect_tilt_angles = torch.tensor(template.tilt_angles, dtype=torch.float32, device=self.device)
        self.collect_pitch_angles = torch.tensor(template.pitch_angles, dtype=torch.float32, device=self.device)
        self.collect_cycle_cursor = 0
        self.collect_object_batch_counts: Dict[str, int] = {}
        self.collect_rollout_entries: List[Dict] = []

        self.current_field_mtp_index = torch.zeros((self.num_envs, 3), dtype=torch.long, device=self.device)
        self.current_pregrasp_hand_back_object = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.current_pregrasp_palm_center_object = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.initial_qpos33 = torch.zeros((self.num_envs, 33), dtype=torch.float32, device=self.device)
        self.initial_palm_center_world = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.initial_palm_toward_world = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.initial_object_pose_world = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self.initial_table_height = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        print(
            "Random MTP pregrasp collector: "
            f"init_mode={self.collect_init_mode}, "
            f"template={self.collect_sphere_template_kind}/subdivision={self.collect_subdivision}, "
            f"shape={self.collect_template_shape}, "
            f"tilt={self.collect_tilt_angles_cfg}, pitch={self.collect_pitch_angles_cfg}, "
            f"mode={self.collect_mtp_mode}, "
            f"rollout_root={self.collect_rollout_root}",
            flush=True,
        )

    def _clear_existing_rollout_files(self):
        root = Path(self.collect_rollout_root).expanduser().resolve()
        if not root.exists():
            return

        removed = 0
        for path in root.glob("*/batch_*.npz"):
            if path.is_file():
                path.unlink()
                removed += 1
        manifest = root / "manifest.json"
        if manifest.is_file():
            manifest.unlink()
            removed += 1

        print(
            "Cleared existing pregrasp rollout files: "
            f"root={root}, removed={removed}",
            flush=True,
        )

    def _sample_trajectory_angle_ids(self, env_ids):
        if self.collect_init_mode == "training":
            return super()._sample_trajectory_angle_ids(env_ids)

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = int(len(env_ids))
        m_count, t_count, p_count = self.collect_template_shape
        if self.collect_mtp_mode == "random":
            flat = torch.randint(self.collect_num_candidates, (count,), dtype=torch.long, device=self.device)
        else:
            flat = (
                torch.arange(count, dtype=torch.long, device=self.device)
                + int(self.collect_cycle_cursor)
            ) % self.collect_num_candidates
            self.collect_cycle_cursor = (self.collect_cycle_cursor + count) % self.collect_num_candidates
        m_id = flat // (t_count * p_count)
        rest = flat % (t_count * p_count)
        t_id = rest // p_count
        p_id = rest % p_count
        mtp = torch.stack([m_id, t_id, p_id], dim=-1)
        self.current_field_mtp_index[env_ids] = mtp
        return t_id * p_count + p_id

    def _set_env_trajectory_rotations(self, env_ids, angle_ids):
        if self.collect_init_mode == "training":
            super()._set_env_trajectory_rotations(env_ids, angle_ids)
            self._record_training_distribution_bins(env_ids)
            return

        self._ensure_base_tracking_reference()
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        angle_ids = angle_ids.to(device=self.device, dtype=torch.long)
        mtp = self.current_field_mtp_index[env_ids]
        m_id, t_id, p_id = mtp[:, 0], mtp[:, 1], mtp[:, 2]

        position_dir = self.collect_sphere_dirs[m_id]
        tilt_deg = self.collect_tilt_angles[t_id]
        pitch_deg = self.collect_pitch_angles[p_id]
        hand_back_dir = _apply_local_angles(position_dir, tilt_deg, pitch_deg)

        base_pos = self.base_tracking_reference["wrist_initobj_pos"][env_ids]
        base_quat = self.base_tracking_reference["wrist_quat"][env_ids]
        base_dir = _normalize(base_pos[:, 0])
        q_pos = _shortest_arc_quat(base_dir, position_dir)
        q_ang = _shortest_arc_quat(position_dir, hand_back_dir)
        q_total = quat_mul(q_ang, q_pos)

        q_pos_seq = q_pos.unsqueeze(1).expand(-1, self.T_ref, -1)
        q_total_seq = q_total.unsqueeze(1).expand(-1, self.T_ref, -1)
        transformed_pos = quat_apply(q_pos_seq, base_pos)
        if self.collect_shell_radius != "base":
            radius = float(self.collect_shell_radius)
            current_radius = torch.linalg.vector_norm(transformed_pos[:, 0], dim=-1, keepdim=True).clamp_min(1e-6)
            transformed_pos = transformed_pos * (radius / current_radius).view(-1, 1, 1)
        transformed_quat = quat_mul(q_total_seq, base_quat)

        for key, value in self.base_tracking_reference.items():
            self.tracking_reference[key][env_ids] = value[env_ids]
        self.tracking_reference["wrist_initobj_pos"][env_ids] = transformed_pos
        self.tracking_reference["wrist_quat"][env_ids] = transformed_quat

        primary_rad = torch.deg2rad(tilt_deg)
        pitch_rad = torch.deg2rad(pitch_deg)
        self.env_angle_ids[env_ids] = angle_ids
        self.env_se_yaw[env_ids] = tilt_deg
        self.env_se_pitch[env_ids] = pitch_deg
        self.env_se_guide_obs[env_ids] = torch.stack(
            [torch.sin(primary_rad), torch.cos(primary_rad), torch.sin(pitch_rad), torch.cos(pitch_rad)],
            dim=-1,
        )
        self.env_traj_angles[env_ids] = torch.linalg.vector_norm(torch.stack([tilt_deg, pitch_deg], dim=-1), dim=-1)
        self.env_traj_quat[env_ids] = q_total
        self.env_traj_quat_inv[env_ids] = quat_conjugate(q_total)
        table_z = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device).expand(len(env_ids), -1)
        self.env_traj_normal[env_ids] = quat_apply(q_total, table_z)
        self.current_world_tilt = float(self.env_traj_angles[env_ids].float().mean().item())
        self.current_pregrasp_hand_back_object[env_ids] = hand_back_dir
        self.current_pregrasp_palm_center_object[env_ids] = transformed_pos[:, 0]

    def _record_training_distribution_bins(self, env_ids):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        palm_obj = self.tracking_reference["wrist_initobj_pos"][env_ids, 0].to(dtype=torch.float32)
        palm_dir = _normalize(palm_obj)
        m_id = torch.argmax(palm_dir @ self.collect_sphere_dirs.transpose(0, 1), dim=-1)
        t_id = torch.argmin(
            torch.abs(self.env_se_yaw[env_ids].view(-1, 1) - self.collect_tilt_angles.view(1, -1)),
            dim=-1,
        )
        p_id = torch.argmin(
            torch.abs(self.env_se_pitch[env_ids].view(-1, 1) - self.collect_pitch_angles.view(1, -1)),
            dim=-1,
        )
        self.current_field_mtp_index[env_ids] = torch.stack([m_id, t_id, p_id], dim=-1)
        self.current_pregrasp_palm_center_object[env_ids] = palm_obj
        self.current_pregrasp_hand_back_object[env_ids] = palm_dir

    def reset_idx(self, env_ids, object_init_pose=None, **kwargs):
        result = super().reset_idx(env_ids, object_init_pose=object_init_pose, **kwargs)
        self._snapshot_initial_rollout_state(env_ids)
        return result

    def _snapshot_initial_rollout_state(self, env_ids):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        dof = self.robot_dof_pos[env_ids]
        qpos33 = torch.zeros((len(env_ids), 33), dtype=torch.float32, device=self.device)
        cols = min(33, int(dof.shape[1]))
        qpos33[:, :cols] = dof[:, :cols]
        object_pose = self.object_init_states[env_ids, 0:7].to(dtype=torch.float32)
        object_pos = object_pose[:, 0:3]
        object_quat = object_pose[:, 3:7]

        self.initial_qpos33[env_ids] = qpos33
        self.initial_object_pose_world[env_ids] = object_pose
        self.initial_table_height[env_ids] = self.table_heights[env_ids].to(dtype=torch.float32)
        if self.collect_init_mode == "training" and hasattr(self, "palm_center_pos"):
            palm_world = self.palm_center_pos[env_ids].to(dtype=torch.float32)
            self.initial_palm_center_world[env_ids] = palm_world
            self.initial_palm_toward_world[env_ids] = _normalize(object_pos - palm_world)
        else:
            palm_obj = self.current_pregrasp_palm_center_object[env_ids]
            toward_obj = -self.current_pregrasp_hand_back_object[env_ids]
            self.initial_palm_center_world[env_ids] = object_pos + quat_apply(object_quat, palm_obj)
            self.initial_palm_toward_world[env_ids] = quat_apply(object_quat, toward_obj)

    def _object_name_for_env_id(self, env_id: int) -> str:
        object_fn = self.object_fns[int(env_id) % len(self.object_fns)]
        return Path(object_fn).stem

    def export_pregrasp_rollout_batch(self, success_values: torch.Tensor, batch_id: int) -> List[Path]:
        root = Path(self.collect_rollout_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        success = success_values.detach().to(device=self.device).reshape(-1).bool()
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        by_object: Dict[str, List[int]] = {}
        for env_id in env_ids.detach().cpu().tolist():
            by_object.setdefault(self._object_name_for_env_id(int(env_id)), []).append(int(env_id))

        written: List[Path] = []
        for object_name, ids in sorted(by_object.items()):
            index = torch.tensor(ids, dtype=torch.long, device=self.device)
            object_root = root / object_name
            object_root.mkdir(parents=True, exist_ok=True)
            local_batch_id = self.collect_object_batch_counts.get(object_name, 0)
            self.collect_object_batch_counts[object_name] = local_batch_id + 1
            path = object_root / f"batch_{local_batch_id:06d}.npz"

            points = np.empty((0, 3), dtype=np.float32)
            if hasattr(self, "obj_pcl_buf"):
                points = self.obj_pcl_buf[index[0]].detach().cpu().numpy().astype(np.float32, copy=False)
            candidate_index = (
                self.current_field_mtp_index[index, 0] * (self.collect_template_shape[1] * self.collect_template_shape[2])
                + self.current_field_mtp_index[index, 1] * self.collect_template_shape[2]
                + self.current_field_mtp_index[index, 2]
            )
            if self.collect_init_mode == "training":
                candidate_index = torch.full_like(candidate_index, -1)
                field_mtp_index = torch.full_like(self.current_field_mtp_index[index], -1)
            else:
                field_mtp_index = self.current_field_mtp_index[index]

            np.savez_compressed(
                path,
                schema=np.asarray(ROLLOUT_SCHEMA),
                object_name=np.asarray(object_name),
                sample_id=np.asarray(f"{object_name}_batch_{local_batch_id:06d}"),
                object_pose_world=self.initial_object_pose_world[index].detach().cpu().numpy().astype(np.float32),
                table_height=self.initial_table_height[index].detach().cpu().numpy().astype(np.float32),
                initial_qpos33=self.initial_qpos33[index].detach().cpu().numpy().astype(np.float32),
                initial_palm_center_world=self.initial_palm_center_world[index].detach().cpu().numpy().astype(np.float32),
                initial_palm_toward_world=self.initial_palm_toward_world[index].detach().cpu().numpy().astype(np.float32),
                success=success[index].detach().cpu().numpy().astype(bool),
                candidate_index=candidate_index.detach().cpu().numpy().astype(np.int64),
                field_mtp_index=field_mtp_index.detach().cpu().numpy().astype(np.int64),
                object_point_cloud_object=points,
                points=points,
            )
            entry = {
                "global_batch_id": int(batch_id),
                "object_name": object_name,
                "path": str(path),
                "env_count": int(len(ids)),
                "success_total": int(success[index].sum().item()),
            }
            self.collect_rollout_entries.append(entry)
            written.append(path)
        return written

    def write_pregrasp_rollout_manifest(self, config: Dict) -> Path:
        root = Path(self.collect_rollout_root).expanduser().resolve()
        payload = {
            "schema": "PregraspPPORolloutManifest/v1",
            "root": str(root),
            "object_batch_counts": self.collect_object_batch_counts,
            "entries": self.collect_rollout_entries,
            "config": config,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


class PregraspRolloutExportPPO:
    """Small wrapper around ResidualPPO's deterministic test rollout."""

    def __init__(self, runner, max_batches: int, manifest_config: Dict):
        self.runner = runner
        self.max_batches = int(max_batches)
        self.manifest_config = manifest_config

    def run(self):
        runner = self.runner
        env = runner.vec_env
        env_ids = torch.arange(env.num_envs, device=runner.device)
        runner.actor_critic.eval()
        success_rates = []
        with torch.no_grad():
            for batch_id in range(self.max_batches):
                obs = env.reset_idx(env_ids)["obs"]
                states = env.get_state()
                baseline_actions = runner._baseline_actions(obs, states)
                residual_actions = runner.actor_critic(obs, states, inference=True)
                env.generate_residual_reaching_plan_idx(
                    env_ids,
                    baseline_actions=baseline_actions,
                    residual_actions=residual_actions,
                )
                effective_success = None
                for step in range(env.max_episode_length):
                    env_action = env.compute_reference_actions()
                    env.step(env_action)
                    if step == env.max_episode_length - 2:
                        effective_success = torch.where(
                            env.has_hit_table,
                            torch.zeros_like(env.successes),
                            env.successes,
                        )
                        break
                if effective_success is None:
                    raise RuntimeError("rollout finished without computing success")
                paths = env.export_pregrasp_rollout_batch(effective_success, batch_id=batch_id)
                success_rate = float(effective_success.float().mean().item())
                success_rates.append(success_rate)
                print(
                    f"Pregrasp rollout batch {batch_id + 1}/{self.max_batches}: "
                    f"success={success_rate:.4f}, files={len(paths)}",
                    flush=True,
                )

        manifest = env.write_pregrasp_rollout_manifest(self.manifest_config)
        values = torch.tensor(success_rates, dtype=torch.float32, device=runner.device)
        print(
            "Pregrasp rollout export summary: "
            f"batches={len(success_rates)}, mean_success={values.mean().item():.4f}, "
            f"manifest={manifest}",
            flush=True,
        )


def _build_runner(cfg, env):
    _prepare_embedded_baseline_if_needed(cfg)
    baseline_checkpoint, residual_checkpoint = get_checkpoint_paths(cfg)
    if not residual_checkpoint:
        raise ValueError("collection requires +residual_checkpoint=/path/to/model.pt")

    from residual_tilt_grasp.residual_actor_critic import ResidualActorCritic
    from residual_tilt_grasp.residual_ppo import ResidualPPO

    runner = ResidualPPO(
        vec_env=env,
        actor_critic_class=ResidualActorCritic,
        train_param=cfg.train.params,
        log_dir=None,
        apply_reset=False,
        action_dim=env.residual_action_dim,
        baseline_checkpoint=baseline_checkpoint,
        residual_metadata=se_residual_metadata(cfg, env),
        baseline_checkpoint_hash_override=cfg.get("embedded_baseline_checkpoint_hash", ""),
        per_angle_advantage=bool(cfg.get("per_angle_advantage", True)),
        advantage_eps=float(cfg.get("advantage_eps", 1e-8)),
        advantage_min_samples=int(cfg.get("advantage_min_samples", 32)),
        zero_inactive_angle_advantage=bool(cfg.get("zero_inactive_angle_advantage", True)),
        freeze_residual_backbone=bool(cfg.get("residual_freeze_backbone", True)),
        initialize_residual_features=bool(cfg.get("residual_initialize_from_baseline", True)),
    )
    runner.load_residual(residual_checkpoint, resume=False)
    runner.actor_critic.eval()
    manifest_config = {
        "source": "build_Pregrasp_data.collect.collect_random_pregrasp_rollouts",
        "residual_checkpoint": str(residual_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "num_envs": int(cfg.task.env.numEnvs),
        "max_batches": int(cfg.get("pregrasp_collect_max_batches", 1)),
        "sphere_template": str(cfg.task.env.pregraspCollectSphereTemplate),
        "subdivision": int(cfg.task.env.pregraspCollectSubdivision),
        "tilt_angles": list(cfg.task.env.pregraspCollectTiltAngles),
        "pitch_angles": list(cfg.task.env.pregraspCollectPitchAngles),
        "init_mode": str(cfg.task.env.pregraspCollectInitMode),
        "mtp_mode": str(cfg.task.env.pregraspCollectMtpMode),
        "shell_radius": str(cfg.task.env.pregraspCollectShellRadius),
        "multi_object_list": str(cfg.task.env.asset.multiObjectList),
        "reset_position_range": _plain_config(cfg.task.env.resetPositionRange),
        "reset_random_rot": str(cfg.task.env.resetRandomRot),
    }
    return PregraspRolloutExportPPO(
        runner,
        max_batches=int(cfg.get("pregrasp_collect_max_batches", 1)),
        manifest_config=manifest_config,
    )


def hydra_main():
    import gym
    import hydra
    import isaacgymenvs
    from isaacgym import gymapi  # noqa: F401
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from omegaconf import DictConfig, open_dict

    import tasks  # noqa: F401

    @hydra.main(version_base="1.3", config_path="../../tasks", config_name="config")
    def _main(cfg: DictConfig):
        set_np_formatting()
        checkpoint_metadata = _checkpoint_residual_metadata(
            cfg.get("residual_checkpoint", "") or cfg.get("checkpoint", "")
        )
        default_tilt_angles = checkpoint_metadata.get("se_tilt_angles", DEFAULT_TILT_PITCH)
        default_pitch_angles = checkpoint_metadata.get("se_pitch_angles", DEFAULT_TILT_PITCH)
        with open_dict(cfg):
            cfg.test = True
            cfg.se_tilt_angles = cfg.get(
                "pregrasp_collect_tilt_angles",
                cfg.get("se_tilt_angles", default_tilt_angles),
            )
            cfg.se_pitch_angles = cfg.get(
                "pregrasp_collect_pitch_angles",
                cfg.get("se_pitch_angles", default_pitch_angles),
            )
            cfg.se_guide_mode = "tilt_pitch"
            cfg.se_sampling = "random"
        configure_se_training(cfg)
        with open_dict(cfg):
            cfg.test = True
            cfg.train.params.test = True
            cfg.task.env.pregraspCollectPregraspRoot = str(
                Path(cfg.get("pregrasp_root", DEFAULT_PREGRASP_ROOT)).expanduser()
            )
            cfg.task.env.pregraspCollectRolloutRoot = str(
                Path(cfg.get("pregrasp_collect_rollout_root", DEFAULT_ROLLOUT_ROOT)).expanduser()
            )
            cfg.task.env.pregraspCollectMtpMode = cfg.get("pregrasp_collect_mtp_mode", "cycle")
            cfg.task.env.pregraspCollectShellRadius = str(cfg.get("pregrasp_collect_shell_radius", "base"))
            cfg.task.env.pregraspCollectSphereTemplate = cfg.get("pregrasp_collect_sphere_template", "geodesic")
            cfg.task.env.pregraspCollectSubdivision = int(
                cfg.get("pregrasp_collect_subdivision", DEFAULT_SUBDIVISION)
            )
            cfg.task.env.pregraspCollectTiltAngles = list(cfg.se_tilt_angles)
            cfg.task.env.pregraspCollectPitchAngles = list(cfg.se_pitch_angles)
            cfg.task.env.pregraspCollectOverwriteRoot = bool(cfg.get("pregrasp_collect_overwrite_root", True))
            cfg.task.env.pregraspCollectInitMode = str(cfg.get("pregrasp_collect_init_mode", "training"))
            cfg.task.env.worldTiltLogInterval = int(cfg.get("pregrasp_collect_log_interval", 50))
        log_viewer_device(cfg)
        print(
            "Random MTP pregrasp rollout collection: "
            f"rollout_root={cfg.task.env.pregraspCollectRolloutRoot}, "
            f"init_mode={cfg.task.env.pregraspCollectInitMode}, "
            f"tilt={list(cfg.task.env.pregraspCollectTiltAngles)}, "
            f"pitch={list(cfg.task.env.pregraspCollectPitchAngles)}, "
            f"batches={cfg.get('pregrasp_collect_max_batches', 1)}, "
            f"num_envs={cfg.task.env.numEnvs}, "
            f"object_list={cfg.task.env.asset.multiObjectList}, "
            f"residual_checkpoint={cfg.get('residual_checkpoint', '')}",
            flush=True,
        )

        isaacgym_task_map[TASK_NAME] = RandomMTPPregraspCollectGrasp
        rank = int(os.getenv("RANK", "0"))
        cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)
        env = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
        )
        if cfg.capture_video:
            env.is_vector_env = True
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{TASK_NAME}_random_mtp_collect",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        runner = _build_runner(cfg, env)
        runner.run()

    _main()


if __name__ == "__main__":
    hydra_main()
