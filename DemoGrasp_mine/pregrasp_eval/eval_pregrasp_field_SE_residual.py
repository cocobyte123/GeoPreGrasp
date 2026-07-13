"""Evaluate SE residual PPO with object-level PregraspPrior heatmaps.

This is a minimal closed-loop test entry:

  field store target_field[M,T,P]
    -> select a high-scoring M/T/P bin for each object
    -> initialize DemoGrasp SE residual from that bin
    -> run deterministic PPO rollout and report/export success.

It intentionally reuses the random-MTP collector environment because that
environment already owns the M/T/P-to-reference-pose mapping.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ISAACGYM_ROOT = PROJECT_ROOT / "thirdparty" / "isaacgym" / "python"
ISAACGYM_ENVS_ROOT = PROJECT_ROOT / "thirdparty" / "IsaacGymEnvs"
DEFAULT_PREGRASP_ROOT = WORKSPACE_ROOT / "PregraspPrior"
DEFAULT_FIELD_ROOT = DEFAULT_PREGRASP_ROOT / "data" / "fields" / "ppo_random_mtp"
DEFAULT_ROLLOUT_ROOT = DEFAULT_PREGRASP_ROOT / "data" / "rollouts" / "ppo_field_guided_eval"
DEFAULT_SUBDIVISION = 3
DEFAULT_TILT_PITCH = [0.0, 15.0, 30.0]


def _add_project_paths() -> None:
    for path in (PROJECT_ROOT, ISAACGYM_ROOT, ISAACGYM_ENVS_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


_add_project_paths()

from build_Pregrasp_data.collect.collect_random_pregrasp_rollouts import (  # noqa: E402
    RandomMTPPregraspCollectGrasp,
    _build_runner,
    _checkpoint_residual_metadata,
    _plain_config,
)
import torch  # noqa: E402
from isaacgymenvs.utils.torch_jit_utils import quat_apply, quat_conjugate  # noqa: E402


def _as_float_list(value) -> List[float]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            import json

            return [float(item) for item in json.loads(text)]
        return [float(item) for item in text.split(",") if item.strip()]
    return [float(item) for item in value]


def _expanded_angle_list(values, step_deg: float) -> List[float]:
    base = _as_float_list(values)
    if not base:
        return []
    step = max(float(step_deg), 1e-6)
    start = min(base)
    end = max(base)
    count = int(np.floor((end - start) / step + 1e-6)) + 1
    expanded = [start + step * index for index in range(count)]
    if expanded[-1] < end - 1e-6:
        expanded.append(end)
    merged = sorted({round(float(v), 6) for v in [*expanded, *base]})
    return [float(v) for v in merged]


def _scalar_to_str(value) -> str:
    array = np.asarray(value)
    if array.shape == ():
        value = array.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _write_field_object_list(field_root: str, requested_multi_object_list: str) -> str:
    """Write an asset YAML containing exactly objects present in field_root."""
    import yaml

    field_root_path = Path(field_root).expanduser().resolve()
    manifest = json.loads((field_root_path / "manifest.json").read_text())
    requested = Path(str(requested_multi_object_list))
    dataset_dir = requested.parts[0] if requested.parts else "union_ycb_unidex"
    asset_root = PROJECT_ROOT / "assets"
    output_rel = f"{dataset_dir}/pregrasp_field_objects.yaml"
    output_path = asset_root / output_rel

    names = []
    missing = []
    for entry in manifest.get("entries", []):
        object_name = str(entry.get("object_name") or entry.get("object_key"))
        filename = f"{object_name}.urdf"
        if not (asset_root / dataset_dir / "urdf" / filename).exists():
            missing.append(filename)
            continue
        names.append(filename)
    if not names:
        raise ValueError(f"No field objects have matching URDFs under {dataset_dir}")
    if missing:
        print(
            "Field object list skipped missing URDFs: "
            + ", ".join(missing[:20]),
            flush=True,
        )
    output_path.write_text(yaml.safe_dump(names, sort_keys=False), encoding="utf-8")
    return output_rel


class FieldGuidedPregraspCollectGrasp(RandomMTPPregraspCollectGrasp):
    """Random-MTP collector variant that selects M/T/P from a field store."""

    def init_configs(self, cfg):
        env_cfg = cfg["env"]
        self.pregrasp_field_root = str(
            env_cfg.get("pregraspFieldRoot", DEFAULT_FIELD_ROOT)
        )
        self.pregrasp_field_topk = int(env_cfg.get("pregraspFieldTopK", 1))
        self.pregrasp_field_selection = str(
            env_cfg.get("pregraspFieldSelection", "cycle")
        )
        self.pregrasp_field_control = str(env_cfg.get("pregraspFieldControl", "mtp"))
        self.pregrasp_field_pose_filter = bool(
            env_cfg.get("pregraspFieldPoseFilter", True)
        )
        self.pregrasp_field_min_polar_deg = float(
            env_cfg.get("pregraspFieldMinPolarDeg", 0.0)
        )
        self.pregrasp_field_max_polar_deg = float(
            env_cfg.get("pregraspFieldMaxPolarDeg", 30.0)
        )
        self.pregrasp_field_min_table_clearance = float(
            env_cfg.get("pregraspFieldMinTableClearance", 0.03)
        )
        self.pregrasp_field_confidence_weight = float(
            env_cfg.get("pregraspFieldConfidenceWeight", 0.01)
        )
        self.pregrasp_field_require_attempt = bool(
            env_cfg.get("pregraspFieldRequireAttempt", True)
        )
        if self.pregrasp_field_topk <= 0:
            raise ValueError("pregraspFieldTopK must be positive")
        if self.pregrasp_field_selection not in ("cycle", "random"):
            raise ValueError("pregraspFieldSelection must be cycle or random")
        if self.pregrasp_field_control not in (
            "m",
            "mtp",
            "training_weighted",
            "expanded_training_arc",
        ):
            raise ValueError(
                "pregraspFieldControl must be m, mtp, training_weighted, "
                "or expanded_training_arc"
            )
        if self.pregrasp_field_min_polar_deg < 0.0:
            raise ValueError("pregraspFieldMinPolarDeg must be non-negative")
        if self.pregrasp_field_max_polar_deg < self.pregrasp_field_min_polar_deg:
            raise ValueError("pregraspFieldMaxPolarDeg must be >= min polar")
        super().init_configs(cfg)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pregrasp_root = Path(self.collect_pregrasp_root).expanduser().resolve()
        pregrasp_import_root = pregrasp_root / "src" if (pregrasp_root / "src").is_dir() else pregrasp_root
        if str(pregrasp_import_root) not in sys.path:
            sys.path.insert(0, str(pregrasp_import_root))
        from oc_pregrasp.field.field_store import FieldStore

        self.pregrasp_field_store = FieldStore(self.pregrasp_field_root)
        self.pregrasp_field_top_bins: Dict[str, np.ndarray] = {}
        self.pregrasp_field_m_scores: Dict[str, np.ndarray] = {}
        self.pregrasp_field_object_cursors: Dict[str, int] = {}
        self.pregrasp_field_pending_object_pose = torch.zeros(
            (self.num_envs, 7), dtype=torch.float32, device=self.device
        )
        self.pregrasp_field_has_pending_pose = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self.pregrasp_field_filter_fallback_count = 0
        for object_fn in self.object_fns:
            object_name = Path(object_fn).stem
            self.pregrasp_field_top_bins[object_name] = self._load_top_bins(
                object_name
            )
        preview = {
            name: bins[: min(3, len(bins))].tolist()
            for name, bins in list(self.pregrasp_field_top_bins.items())[:5]
        }
        print(
            "Field-guided pregrasp eval: "
            f"field_root={self.pregrasp_field_root}, "
            f"control={self.pregrasp_field_control}, "
            f"pose_filter={self.pregrasp_field_pose_filter}, "
            f"polar=[{self.pregrasp_field_min_polar_deg:g},"
            f"{self.pregrasp_field_max_polar_deg:g}]deg, "
            f"table_clearance={self.pregrasp_field_min_table_clearance:g}m, "
            f"topk={self.pregrasp_field_topk}, "
            f"selection={self.pregrasp_field_selection}, "
            f"objects={len(self.pregrasp_field_top_bins)}, "
            f"preview_top_bins={preview}",
            flush=True,
        )

    def _load_top_bins(self, object_name: str) -> np.ndarray:
        field = self.pregrasp_field_store.load(object_name)
        target = np.asarray(field["target_field"], dtype=np.float32)
        confidence = np.asarray(
            field.get("confidence", np.zeros_like(target)), dtype=np.float32
        )
        attempt = np.asarray(
            field.get("attempt_count", confidence > 0), dtype=np.float32
        )
        if tuple(target.shape) != tuple(self.collect_template_shape):
            raise ValueError(
                f"{object_name} field shape {target.shape} does not match "
                f"collector template {self.collect_template_shape}"
            )
        if (
            "tilt_angles" in field
            and not np.allclose(field["tilt_angles"], self.collect_tilt_angles_cfg)
        ):
            print(
                f"Warning: {object_name} field tilt angles "
                f"{field['tilt_angles'].tolist()} differ from eval "
                f"{self.collect_tilt_angles_cfg}",
                flush=True,
            )
        if (
            "pitch_angles" in field
            and not np.allclose(field["pitch_angles"], self.collect_pitch_angles_cfg)
        ):
            print(
                f"Warning: {object_name} field pitch angles "
                f"{field['pitch_angles'].tolist()} differ from eval "
                f"{self.collect_pitch_angles_cfg}",
                flush=True,
            )

        if self.pregrasp_field_control in (
            "m",
            "training_weighted",
            "expanded_training_arc",
        ):
            m_attempt = attempt.sum(axis=(1, 2))
            m_success = np.asarray(field.get("success_count", np.zeros_like(target)), dtype=np.float32).sum(axis=(1, 2))
            m_target = np.zeros_like(m_attempt, dtype=np.float32)
            nonzero = m_attempt > 0
            m_target[nonzero] = m_success[nonzero] / m_attempt[nonzero]
            max_attempt = float(m_attempt.max())
            m_confidence = np.zeros_like(m_attempt, dtype=np.float32)
            if max_attempt > 0:
                m_confidence = m_attempt / max_attempt
            flat_score = (m_target + self.pregrasp_field_confidence_weight * m_confidence).astype(np.float64)
            flat_attempt = m_attempt.reshape(-1)
            self.pregrasp_field_m_scores[object_name] = flat_score.astype(np.float32)
        else:
            score = target + self.pregrasp_field_confidence_weight * confidence
            flat_score = score.reshape(-1).astype(np.float64)
            flat_attempt = attempt.reshape(-1)
        if self.pregrasp_field_require_attempt and np.any(flat_attempt > 0):
            flat_score = flat_score.copy()
            flat_score[flat_attempt <= 0] = -np.inf
        order = np.argsort(-flat_score, kind="stable")
        order = order[np.isfinite(flat_score[order])]
        if len(order) == 0:
            fallback = (
                m_target
                if self.pregrasp_field_control in (
                    "m",
                    "training_weighted",
                    "expanded_training_arc",
                )
                else target.reshape(-1)
            )
            order = np.argsort(-fallback.reshape(-1), kind="stable")
        return order.astype(np.int64)

    def _candidate_m_ids_for_training_distribution(self, env_id: int) -> torch.Tensor:
        self._ensure_base_tracking_reference()
        angle_ids = torch.arange(
            len(self.se_angle_pairs), dtype=torch.long, device=self.device
        )
        env_ids = torch.full_like(angle_ids, int(env_id))
        pairs = self.se_angle_pairs[angle_ids]
        primary_deg = pairs[:, 0]
        pitch_deg = pairs[:, 1]
        if self.se_guide_mode == "tilt_pitch":
            guide_quat = self._guide_quat_from_tilt_pitch(
                env_ids, primary_deg, pitch_deg
            )
        else:
            guide_quat = self._guide_quat_from_yaw_pitch(
                env_ids, primary_deg, pitch_deg
            )
        base_pos = self.base_tracking_reference["wrist_initobj_pos"][int(env_id), 0]
        palm_offset_world = quat_apply(
            guide_quat, base_pos.view(1, 3).expand(len(angle_ids), -1)
        )
        object_quat = self.pregrasp_field_pending_object_pose[int(env_id), 3:7]
        object_quat_inv = quat_conjugate(
            object_quat.view(1, 4).expand(len(angle_ids), -1)
        )
        palm_offset_object = quat_apply(object_quat_inv, palm_offset_world)
        direction = palm_offset_object / torch.linalg.vector_norm(
            palm_offset_object, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return torch.argmax(direction @ self.collect_sphere_dirs.transpose(0, 1), dim=-1)

    def _choose_training_weighted_angle_id(self, object_name: str, env_id: int) -> int:
        m_ids = self._candidate_m_ids_for_training_distribution(env_id)
        scores_np = self.pregrasp_field_m_scores.get(object_name)
        if scores_np is None:
            bins = self.pregrasp_field_top_bins[object_name]
            scores_np = np.zeros((self.collect_template_shape[0],), dtype=np.float32)
            scores_np[np.asarray(bins[: self.pregrasp_field_topk], dtype=np.int64)] = 1.0
        scores = torch.tensor(scores_np, dtype=torch.float32, device=self.device)[m_ids]
        scores = torch.clamp(scores, min=0.0)
        if self.pregrasp_field_selection == "random":
            total = float(scores.sum().item())
            if total <= 1e-8:
                return int(torch.randint(len(m_ids), (1,), device=self.device).item())
            return int(torch.multinomial(scores / total, 1).item())

        best_score = float(scores.max().item()) if len(scores) else 0.0
        best = torch.nonzero(scores == best_score, as_tuple=False).reshape(-1)
        if len(best) == 0:
            return 0
        cursor = self.pregrasp_field_object_cursors.get(object_name, 0)
        self.pregrasp_field_object_cursors[object_name] = cursor + 1
        return int(best[cursor % len(best)].item())

    def _ensure_pending_object_poses(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        object_pose = self._sample_local_object_poses(env_ids)
        self.pregrasp_field_pending_object_pose[env_ids] = object_pose.to(
            dtype=torch.float32
        )
        self.pregrasp_field_has_pending_pose[env_ids] = True

    def _prepare_tilted_reset(self, env_ids):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if not bool(torch.all(self.pregrasp_field_has_pending_pose[env_ids]).item()):
            return super()._prepare_tilted_reset(env_ids)

        count = len(env_ids)
        world_quat = self.identity_quat.view(1, 4).expand(count, -1)
        table_pivot = self.canonical_table_top[env_ids]
        self.root_state_tensor[self.table_indices[env_ids], 0:3] = table_pivot
        self.root_state_tensor[self.table_indices[env_ids], 3:7] = world_quat
        self.root_state_tensor[self.table_indices[env_ids], 7:13] = 0.0
        effective_height = table_pivot[:, 2] + self.table_thickness * 0.5
        self.table_height_range = torch.stack(
            [effective_height.min(), effective_height.max()]
        )
        self.pregrasp_field_has_pending_pose[env_ids] = False
        return self.pregrasp_field_pending_object_pose[env_ids]

    def _candidate_bins_for_env(self, object_name: str, env_id: int) -> np.ndarray:
        bins = self.pregrasp_field_top_bins[object_name]
        if (
            not self.pregrasp_field_pose_filter
            or env_id is None
            or len(bins) == 0
        ):
            return bins[: self.pregrasp_field_topk]

        t_count, p_count = self.collect_template_shape[1], self.collect_template_shape[2]
        bins_tensor = torch.tensor(bins, dtype=torch.long, device=self.device)
        if self.pregrasp_field_control == "m":
            m_ids = bins_tensor
        else:
            m_ids = bins_tensor // (t_count * p_count)

        object_pose = self.pregrasp_field_pending_object_pose[int(env_id)]
        object_quat = object_pose[3:7].view(1, 4).expand(len(m_ids), -1)
        dirs_obj = self.collect_sphere_dirs[m_ids]
        dirs_world = quat_apply(object_quat, dirs_obj)
        z = dirs_world[:, 2].clamp(-1.0, 1.0)
        polar_deg = torch.rad2deg(torch.arccos(z))

        if self.collect_shell_radius == "base":
            radius = torch.linalg.vector_norm(
                self.base_tracking_reference["wrist_initobj_pos"][int(env_id), 0]
            )
        else:
            radius = torch.tensor(
                float(self.collect_shell_radius), dtype=torch.float32, device=self.device
            )
        hand_z = object_pose[2] + z * radius
        min_hand_z = (
            self.canonical_table_top[int(env_id), 2]
            + float(self.pregrasp_field_min_table_clearance)
        )
        valid = (
            (polar_deg >= float(self.pregrasp_field_min_polar_deg))
            & (polar_deg <= float(self.pregrasp_field_max_polar_deg))
            & (hand_z >= min_hand_z)
        )
        valid_bins = bins_tensor[valid].detach().cpu().numpy().astype(np.int64)
        if len(valid_bins) == 0:
            self.pregrasp_field_filter_fallback_count += 1
            if self.pregrasp_field_filter_fallback_count <= 10:
                print(
                    "Warning: no pose-filtered pregrasp bins; "
                    f"object={object_name}, env={env_id}, fallback=unfiltered",
                    flush=True,
                )
            return bins[: self.pregrasp_field_topk]
        return valid_bins[: self.pregrasp_field_topk]

    def _choose_flat_bin(self, object_name: str, env_id: int | None = None) -> int:
        bins = self._candidate_bins_for_env(object_name, env_id)
        if self.pregrasp_field_selection == "random":
            index = int(torch.randint(len(bins), (1,), device=self.device).item())
        else:
            cursor = self.pregrasp_field_object_cursors.get(object_name, 0)
            index = cursor % len(bins)
            self.pregrasp_field_object_cursors[object_name] = cursor + 1
        return int(bins[index])

    def _sample_trajectory_angle_ids(self, env_ids):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._ensure_pending_object_poses(env_ids)
        m_count, t_count, p_count = self.collect_template_shape
        if self.pregrasp_field_control in (
            "training_weighted",
            "expanded_training_arc",
        ):
            angle_ids = []
            for env_id in [int(v) for v in env_ids.detach().cpu().tolist()]:
                object_name = self._object_name_for_env_id(env_id)
                angle_ids.append(
                    self._choose_training_weighted_angle_id(object_name, env_id)
                )
            return torch.tensor(angle_ids, dtype=torch.long, device=self.device)

        if self.pregrasp_field_control == "m":
            angle_tensor = super(RandomMTPPregraspCollectGrasp, self)._sample_trajectory_angle_ids(env_ids)
            mtp_rows = []
            for env_id, angle_id in zip(
                [int(v) for v in env_ids.detach().cpu().tolist()],
                [int(v) for v in angle_tensor.detach().cpu().tolist()],
            ):
                object_name = self._object_name_for_env_id(env_id)
                m_id = self._choose_flat_bin(object_name, env_id=env_id)
                t_id = angle_id // p_count
                p_id = angle_id % p_count
                mtp_rows.append([m_id, t_id, p_id])
            self.current_field_mtp_index[env_ids] = torch.tensor(
                mtp_rows, dtype=torch.long, device=self.device
            )
            return angle_tensor

        mtp_rows = []
        angle_ids = []
        for env_id in [int(v) for v in env_ids.detach().cpu().tolist()]:
            object_name = self._object_name_for_env_id(env_id)
            flat = self._choose_flat_bin(object_name, env_id=env_id)
            m_id = flat // (t_count * p_count)
            rest = flat % (t_count * p_count)
            t_id = rest // p_count
            p_id = rest % p_count
            mtp_rows.append([m_id, t_id, p_id])
            angle_ids.append(t_id * p_count + p_id)
        mtp = torch.tensor(mtp_rows, dtype=torch.long, device=self.device)
        angle_tensor = torch.tensor(angle_ids, dtype=torch.long, device=self.device)
        self.current_field_mtp_index[env_ids] = mtp
        return angle_tensor


def hydra_main():
    import gym
    import hydra
    import isaacgymenvs
    from isaacgym import gymapi  # noqa: F401
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from omegaconf import DictConfig, open_dict

    import tasks  # noqa: F401
    from residual_se_grasp.train_se_reslearn import (
        TASK_NAME,
        configure_se_training,
    )
    from residual_tilt_grasp.train_tilted_hand_only_reslearn import log_viewer_device

    @hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
    def _main(cfg: DictConfig):
        set_np_formatting()
        checkpoint_metadata = _checkpoint_residual_metadata(
            cfg.get("residual_checkpoint", "") or cfg.get("checkpoint", "")
        )
        default_tilt_angles = checkpoint_metadata.get(
            "se_tilt_angles", DEFAULT_TILT_PITCH
        )
        default_pitch_angles = checkpoint_metadata.get(
            "se_pitch_angles", DEFAULT_TILT_PITCH
        )
        field_control = cfg.get("pregrasp_field_control", "mtp")
        field_tilt_angles = cfg.get(
            "pregrasp_field_tilt_angles",
            cfg.get("pregrasp_collect_tilt_angles", default_tilt_angles),
        )
        field_pitch_angles = cfg.get(
            "pregrasp_field_pitch_angles",
            cfg.get("pregrasp_collect_pitch_angles", default_pitch_angles),
        )
        if field_control == "expanded_training_arc":
            expanded_step_deg = float(cfg.get("pregrasp_field_expanded_step_deg", 5.0))
            se_tilt_angles = cfg.get(
                "pregrasp_field_expanded_tilt_angles",
                _expanded_angle_list(field_tilt_angles, expanded_step_deg),
            )
            se_pitch_angles = cfg.get(
                "pregrasp_field_expanded_pitch_angles",
                _expanded_angle_list(field_pitch_angles, expanded_step_deg),
            )
        else:
            se_tilt_angles = field_tilt_angles
            se_pitch_angles = field_pitch_angles
        with open_dict(cfg):
            cfg.test = True
            if cfg.get("checkpoint", "") and not cfg.get("residual_checkpoint", ""):
                cfg.residual_checkpoint = cfg.checkpoint
            cfg.se_tilt_angles = se_tilt_angles
            cfg.se_pitch_angles = se_pitch_angles
            cfg.pregrasp_field_template_tilt_angles = field_tilt_angles
            cfg.pregrasp_field_template_pitch_angles = field_pitch_angles
            cfg.se_guide_mode = "tilt_pitch"
            cfg.se_sampling = "random"
        configure_se_training(cfg)
        with open_dict(cfg):
            cfg.test = True
            cfg.train.params.test = True
            field_root_text = str(
                Path(cfg.get("pregrasp_field_root", DEFAULT_FIELD_ROOT)).expanduser()
            )
            if bool(cfg.get("pregrasp_field_auto_object_list", True)):
                cfg.task.env.asset.multiObjectList = _write_field_object_list(
                    field_root_text, cfg.task.env.asset.multiObjectList
                )
            cfg.task.env.pregraspCollectPregraspRoot = str(
                Path(cfg.get("pregrasp_root", DEFAULT_PREGRASP_ROOT)).expanduser()
            )
            cfg.task.env.pregraspCollectRolloutRoot = str(
                Path(
                    cfg.get("pregrasp_collect_rollout_root", DEFAULT_ROLLOUT_ROOT)
                ).expanduser()
            )
            cfg.task.env.pregraspCollectOverwriteRoot = bool(
                cfg.get("pregrasp_collect_overwrite_root", True)
            )
            cfg.task.env.pregraspCollectInitMode = (
                "training"
                if field_control in ("training_weighted", "expanded_training_arc")
                else "mtp"
            )
            cfg.task.env.pregraspCollectMtpMode = "cycle"
            cfg.task.env.pregraspCollectShellRadius = str(
                cfg.get("pregrasp_collect_shell_radius", "base")
            )
            cfg.task.env.pregraspCollectSphereTemplate = cfg.get(
                "pregrasp_collect_sphere_template", "geodesic"
            )
            cfg.task.env.pregraspCollectSubdivision = int(
                cfg.get("pregrasp_collect_subdivision", DEFAULT_SUBDIVISION)
            )
            cfg.task.env.pregraspCollectTiltAngles = _as_float_list(
                cfg.pregrasp_field_template_tilt_angles
            )
            cfg.task.env.pregraspCollectPitchAngles = _as_float_list(
                cfg.pregrasp_field_template_pitch_angles
            )
            cfg.task.env.pregraspFieldRoot = field_root_text
            cfg.task.env.pregraspFieldTopK = int(
                cfg.get("pregrasp_field_topk", 1)
            )
            cfg.task.env.pregraspFieldSelection = cfg.get(
                "pregrasp_field_selection", "cycle"
            )
            cfg.task.env.pregraspFieldControl = field_control
            cfg.task.env.pregraspFieldPoseFilter = bool(
                cfg.get("pregrasp_field_pose_filter", True)
            )
            cfg.task.env.pregraspFieldMinPolarDeg = float(
                cfg.get("pregrasp_field_min_polar_deg", 0.0)
            )
            cfg.task.env.pregraspFieldMaxPolarDeg = float(
                cfg.get("pregrasp_field_max_polar_deg", 30.0)
            )
            cfg.task.env.pregraspFieldMinTableClearance = float(
                cfg.get("pregrasp_field_min_table_clearance", 0.03)
            )
            cfg.task.env.pregraspFieldConfidenceWeight = float(
                cfg.get("pregrasp_field_confidence_weight", 0.01)
            )
            cfg.task.env.pregraspFieldRequireAttempt = bool(
                cfg.get("pregrasp_field_require_attempt", True)
            )
            cfg.task.env.worldTiltLogInterval = int(
                cfg.get("pregrasp_eval_log_interval", 0)
            )
        log_viewer_device(cfg)
        print(
            "Field-guided SE residual eval: "
            f"field_root={cfg.task.env.pregraspFieldRoot}, "
            f"control={cfg.task.env.pregraspFieldControl}, "
            f"pose_filter={cfg.task.env.pregraspFieldPoseFilter}, "
            f"polar=[{cfg.task.env.pregraspFieldMinPolarDeg:g},"
            f"{cfg.task.env.pregraspFieldMaxPolarDeg:g}]deg, "
            f"table_clearance={cfg.task.env.pregraspFieldMinTableClearance:g}m, "
            f"rollout_root={cfg.task.env.pregraspCollectRolloutRoot}, "
            f"topk={cfg.task.env.pregraspFieldTopK}, "
            f"selection={cfg.task.env.pregraspFieldSelection}, "
            f"batches={cfg.get('pregrasp_collect_max_batches', 1)}, "
            f"num_envs={cfg.task.env.numEnvs}, "
            f"object_list={cfg.task.env.asset.multiObjectList}, "
            f"se_tilt={list(cfg.se_tilt_angles)}, "
            f"se_pitch={list(cfg.se_pitch_angles)}, "
            f"field_tilt={list(cfg.task.env.pregraspCollectTiltAngles)}, "
            f"field_pitch={list(cfg.task.env.pregraspCollectPitchAngles)}, "
            f"residual_checkpoint={cfg.get('residual_checkpoint', '')}",
            flush=True,
        )

        isaacgym_task_map[TASK_NAME] = FieldGuidedPregraspCollectGrasp
        rank = int(os.getenv("RANK", "0"))
        cfg.seed = set_seed(
            cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank
        )
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
                f"videos/{TASK_NAME}_field_guided_eval",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        runner = _build_runner(cfg, env)
        runner.manifest_config.update(
            {
                "source": "pregrasp_eval.eval_pregrasp_field_SE_residual",
                "field_root": str(cfg.task.env.pregraspFieldRoot),
                "field_control": str(cfg.task.env.pregraspFieldControl),
                "field_pose_filter": bool(cfg.task.env.pregraspFieldPoseFilter),
                "field_min_polar_deg": float(cfg.task.env.pregraspFieldMinPolarDeg),
                "field_max_polar_deg": float(cfg.task.env.pregraspFieldMaxPolarDeg),
                "field_min_table_clearance": float(
                    cfg.task.env.pregraspFieldMinTableClearance
                ),
                "se_tilt_angles": list(cfg.se_tilt_angles),
                "se_pitch_angles": list(cfg.se_pitch_angles),
                "field_topk": int(cfg.task.env.pregraspFieldTopK),
                "field_selection": str(cfg.task.env.pregraspFieldSelection),
                "reset_position_range": _plain_config(cfg.task.env.resetPositionRange),
                "reset_random_rot": str(cfg.task.env.resetRandomRot),
            }
        )
        runner.run()

    _main()


if __name__ == "__main__":
    hydra_main()
