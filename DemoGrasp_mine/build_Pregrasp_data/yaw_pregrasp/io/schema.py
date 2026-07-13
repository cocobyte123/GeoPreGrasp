"""Schema constants for yaw-expanded pregrasp rollout files.

The new dataset labels candidate pregrasp parameters directly:

    P(success | object, object_pose, yaw, tilt, pitch)

The policy still receives canonical hand observations; yaw is an external
geometric expansion used for execution and labeling.
"""

from __future__ import annotations

ROLLOUT_REQUIRED_KEYS = (
    "object_id",
    "object_asset",
    "object_pose_world",
    "yaw_deg",
    "tilt_deg",
    "pitch_deg",
    "success",
    "hit_table",
    "min_clearance",
)

ROLLOUT_OPTIONAL_KEYS = (
    "real_palm_pose_world",
    "canonical_palm_pose_world",
    "world_yaw_deg",
    "object_yaw_deg",
    "reference_yaw_deg",
    "angle_id",
    "env_id",
    "batch_id",
)

FIELD_REQUIRED_KEYS = (
    "object_id",
    "object_asset",
    "yaw_angles",
    "tilt_angles",
    "pitch_angles",
    "success_count",
    "trial_count",
    "success_rate",
)
