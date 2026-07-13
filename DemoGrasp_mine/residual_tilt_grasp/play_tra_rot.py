'''
DemoGrasp: sr_shadow_hand trajectory-rotation replay.

The table, object, and gravity stay in the canonical horizontal world. The hand
reference trajectory is rotated around the object so the palm approaches the
object from different table-relative directions while still facing the object.
The default axis_tilt mode rotates the whole trajectory around a horizontal
table-plane axis through the object center. With a large angle, this produces
the side-grasp / palm-near-table behavior instead of a flat yaw turn.

Usage:
  python residual_tilt_grasp/play_tra_rot.py --num_envs 16 --traj_rot 90 --rot_axis 1 0 0
  python residual_tilt_grasp/play_tra_rot.py --num_envs 16 --traj_rot 0 45 90 135 180
'''
import os
import sys
import argparse
import pickle
import yaml
import math
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
import torch

ASSET_ROOT = "./assets"


# ─── Hand configuration ─────────────────────────────────────────────────────────
@dataclass
class HandConfig:
    name: str
    urdf_path: str                          # relative to ASSET_ROOT
    ref_file: str                           # reference trajectory .pkl
    base_joint_names: List[str]             # virtual base joints (empty if floating base)
    hand_joint_names: List[str]             # active hand joints in trajectory order
    mimic: Dict[str, str] = field(default_factory=dict)   # child → parent name
    num_hand_dofs: int = 18                 # active hand DOF count in trajectory
    fix_base_link: bool = True              # True for joints-as-base, False for floating
    default_dof_pos: Optional[List[float]] = None  # initial DOF positions
    base_rotation_order: str = "zyx"       # actual axis order of the three base rotation joints
    wrist_quat_perm: tuple = (0, 1, 2, 3)  # (qx, qy, qz, qw) permutation+negation from traj to URDF frame.
                                            # e.g. (0, -2, 1, 3) means: qx'=qx, qy'=-qz, qz'=qy, qw'=qw
    wrist_quat_offset: tuple = (0, 0, 0, 1)  # fixed quaternion [x,y,z,w] applied AFTER perm, as right-multiply.
                                               # e.g. 90° around -Z: (0, 0, -0.7071068, 0.7071068)
    palm_offset_from_forearm: tuple = (0.0, 0.0, 0.0)  # (x,y,z) offset in hand-local frame from the
                                                         # base-positioned link (forearm for sr_shadow_hand)
                                                         # to the functional palm. Subtracted from target
                                                         # position so the palm reaches the trajectory point.
    align_default_base_to_ref: bool = False  # initialize base rotation from reference frame 0
    ref_qpos_prefix_zeros: int = 0           # prepend zero hand DOFs to older reference trajectories

    @property
    def num_base_dofs(self) -> int:
        return len(self.base_joint_names)

    @property
    def all_joint_names(self) -> List[str]:
        return self.base_joint_names + self.hand_joint_names + list(self.mimic.keys())


HAND_CONFIGS: Dict[str, HandConfig] = {
    "shadow_simple": HandConfig(
        name="shadow_simple",
        urdf_path="shadow_hand_simple/right_with_base.urdf",
        ref_file="tasks/grasp_ref_shadow.pkl",
        base_joint_names=['baseJX', 'baseJY', 'baseJZ',
                          'baseJROLL', 'baseJPITCH', 'baseJYAW'],
        hand_joint_names=[
            "rh_FFJ4", "rh_FFJ3", "rh_FFJ2",
            "rh_MFJ4", "rh_MFJ3", "rh_MFJ2",
            "rh_RFJ4", "rh_RFJ3", "rh_RFJ2",
            "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2",
            "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
        ],
        mimic={"rh_FFJ1": "rh_FFJ2", "rh_MFJ1": "rh_MFJ2",
                "rh_RFJ1": "rh_RFJ2", "rh_LFJ1": "rh_LFJ2"},
        num_hand_dofs=18,
        fix_base_link=True,
        default_dof_pos=[0.5, -0.1, 0.4, 0, 1.57, 0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ),
    "sr_shadow_hand": HandConfig(
        name="sr_shadow_hand",
        urdf_path="urdf/sr_grasp_description/urdf/pregrasp_movable_shadowhand.urdf",
        ref_file="tasks/grasp_ref_sr_shadow.pkl",
        base_joint_names=['baseJX', 'baseJY', 'baseJZ',
                          'baseJROLL', 'baseJPITCH', 'baseJYAW'],
        hand_joint_names=[
            "WRJ2", "WRJ1",
            "FFJ4", "FFJ3", "FFJ2", "FFJ1",
            "MFJ4", "MFJ3", "MFJ2", "MFJ1",
            "RFJ4", "RFJ3", "RFJ2", "RFJ1",
            "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
            "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
        ],
        mimic={},           # all 24 hand joints active, no mimic
        num_hand_dofs=24,
        fix_base_link=True,
        base_rotation_order="xyz",
        default_dof_pos=[0.5, -0.1, 0.4, 0, 1.57, 0,
                         0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0],
        # forearm → WRJ2(xyz=0,-0.01,0.213) → wrist → WRJ1(xyz=0,0,0.034) → palm
        # Total offset from forearm (where base joints position) to palm:
        palm_offset_from_forearm=(0.0, -0.01, 0.247),
        align_default_base_to_ref=True,
        # OLD hand: palm = wrist * Rz(90°); NEW hand: palm = wrist (WRJ1 rpy=0)
        # To align palm orientations, apply 90° around Z to trajectory quat:
        wrist_quat_offset=(0, 0, 0.7071068, 0.7071068),  # 90° around Z
    ),
    "new_sr_hand_simple": HandConfig(
        name="new_sr_hand_simple",
        urdf_path="new_sr_hand_simple/right_with_wrist.urdf",
        ref_file="tasks/grasp_ref_shadow.pkl",
        base_joint_names=['baseJX', 'baseJY', 'baseJZ',
                          'baseJROLL', 'baseJPITCH', 'baseJYAW'],
        hand_joint_names=[
            "WRJ2", "WRJ1",
            "rh_FFJ4", "rh_FFJ3", "rh_FFJ2",
            "rh_MFJ4", "rh_MFJ3", "rh_MFJ2",
            "rh_RFJ4", "rh_RFJ3", "rh_RFJ2",
            "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2",
            "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
        ],
        mimic={"rh_FFJ1": "rh_FFJ2", "rh_MFJ1": "rh_MFJ2",
                "rh_RFJ1": "rh_RFJ2", "rh_LFJ1": "rh_LFJ2"},
        num_hand_dofs=20,
        fix_base_link=True,
        default_dof_pos=[0.5, -0.1, 0.4, 0, 1.57, 0,
                         0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0],
        # Reuse the shadow_simple reference and object center. The two added
        # wrist joints are inserted as leading zero DOFs in memory only.
        align_default_base_to_ref=True,
        ref_qpos_prefix_zeros=2,
    ),
    #
    # To add a new hand (e.g. floating-base, no arm):
    #   "my_hand": HandConfig(
    #       name="my_hand",
    #       urdf_path="my_hand/my_hand.urdf",
    #       ref_file="tasks/grasp_ref_my_hand.pkl",
    #       base_joint_names=[],       # empty = floating base
    #       hand_joint_names=[...],
    #       mimic={...},
    #       num_hand_dofs=N,
    #       fix_base_link=False,       # floating base → use root state tensor
    #       wrist_quat_perm=(0, -2, 1, 3),  # [x,y,z]→[x,-z,y] if frames differ
    #       wrist_quat_offset=(0, 0, -0.7071068, 0.7071068),  # 90° around -Z
    #   ),
}

# ─── Utility: quaternion [x,y,z,w] → euler [roll,pitch,yaw] (ZYX extrinsic)
def quat_to_euler(q):
    x, y, z, w = q.unbind(-1)
    r = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    p = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
    y_ = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return torch.stack([r, p, y_], -1)


def quat_to_base_angles(q, rotation_order):
    """Return the three serial base-joint angles for the URDF rotation chain."""
    if rotation_order == "zyx":
        # shadow_simple joints are named ROLL/PITCH/YAW but rotate around Z/Y/X.
        roll, pitch, yaw = quat_to_euler(q).unbind(-1)
        return torch.stack([yaw, pitch, roll], dim=-1)
    if rotation_order == "xyz":
        # sr_shadow_hand joints rotate around X/Y/Z in that serial order.
        x, y, z, w = q.unbind(-1)
        angle_x = torch.atan2(
            2 * (w * x - y * z),
            1 - 2 * (x * x + y * y),
        )
        angle_y = torch.asin((2 * (w * y + x * z)).clamp(-1, 1))
        angle_z = torch.atan2(
            2 * (w * z - x * y),
            1 - 2 * (y * y + z * z),
        )
        return torch.stack([angle_x, angle_y, angle_z], dim=-1)
    raise ValueError(f"Unsupported base rotation order: {rotation_order}")


def base_angles_to_quat(angles, rotation_order):
    """Compose the serial base-joint rotations as a quaternion."""
    first, second, third = angles.unbind(-1)
    zeros = torch.zeros_like(first)

    if rotation_order == "zyx":
        q_first = quat_from_euler(zeros, zeros, first)
        q_second = quat_from_euler(zeros, second, zeros)
        q_third = quat_from_euler(third, zeros, zeros)
    elif rotation_order == "xyz":
        q_first = quat_from_euler(first, zeros, zeros)
        q_second = quat_from_euler(zeros, second, zeros)
        q_third = quat_from_euler(zeros, zeros, third)
    else:
        raise ValueError(f"Unsupported base rotation order: {rotation_order}")

    return quat_mul(quat_mul(q_first, q_second), q_third)


def apply_quat_perm(q, perm):
    """Apply component permutation + negation to a quaternion [x,y,z,w].

    perm: 4-tuple (ix, iy, iz, iw) where negative means negate that component.
          e.g. (0, -2, 1, 3) → qx'=q[0], qy'=-q[2], qz'=q[1], qw'=q[3]
    """
    # Map: 0→q[...,0], 1→q[...,1], 2→q[...,2], 3→q[...,3]
    # Negative → negate the component
    idx = torch.tensor([abs(p) for p in perm], dtype=torch.long, device=q.device)
    sign = torch.tensor([1 if p >= 0 else -1 for p in perm], dtype=q.dtype, device=q.device)
    return q[..., idx] * sign


def quat_mul(a, b):
    """Multiply quaternions a * b, each [x, y, z, w]."""
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], -1)


def quat_conjugate(q):
    result = q.clone()
    result[..., :3] = -result[..., :3]
    return result


def quat_apply(q, v):
    """Rotate vectors v by quaternions q, supporting leading dimensions."""
    zeros = torch.zeros_like(v[..., :1])
    pure_vector = torch.cat([v, zeros], dim=-1)
    return quat_mul(quat_mul(q, pure_vector), quat_conjugate(q))[..., :3]


def normalize_vec(v, eps=1e-8):
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)


def quat_from_two_vectors(src, dst):
    """Shortest quaternion rotating normalized vector src to normalized dst."""
    src = normalize_vec(src)
    dst = normalize_vec(dst)
    cross = torch.cross(src, dst, dim=-1)
    dot = (src * dst).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    quat = torch.cat([cross, 1.0 + dot], dim=-1)

    opposite = (dot.squeeze(-1) < -0.9999)
    if opposite.any():
        fallback = torch.zeros_like(src)
        use_x = src[..., 0].abs() < 0.9
        fallback[..., 0] = use_x.to(src.dtype)
        fallback[..., 1] = (~use_x).to(src.dtype)
        axis = normalize_vec(torch.cross(src, fallback, dim=-1))
        quat = torch.where(
            opposite.unsqueeze(-1),
            torch.cat([axis, torch.zeros_like(dot)], dim=-1),
            quat,
        )
    return normalize_vec(quat)


def quat_from_axis_angle(axis, angle_deg):
    """Convert a normalized axis and angles in degrees to [x, y, z, w]."""
    angles = torch.deg2rad(angle_deg) * 0.5
    sin_half = torch.sin(angles).unsqueeze(-1)
    return torch.cat(
        [axis.unsqueeze(0) * sin_half, torch.cos(angles).unsqueeze(-1)],
        dim=-1,
    )


def wrist_fk_sr(wrj2, wrj1):
    """Return forearm->palm orientation and palm origin offset for sr wrist."""
    zeros = torch.zeros_like(wrj2)
    q_wrj2 = quat_from_euler(zeros, wrj2, zeros)
    q_wrj1 = quat_from_euler(wrj1, zeros, zeros)
    wrist_quat = quat_mul(q_wrj2, q_wrj1)
    t_forearm_wrist = torch.tensor(
        [0.0, -0.01, 0.213], dtype=wrj2.dtype, device=wrj2.device
    )
    t_wrist_palm = torch.tensor(
        [0.0, 0.0, 0.034], dtype=wrj2.dtype, device=wrj2.device
    )
    offset = t_forearm_wrist + quat_apply(q_wrj2, t_wrist_palm)
    return wrist_quat, offset


def wrist_fk_zero_offset(wrj2, wrj1):
    """Return local WRJ2/WRJ1 orientation without changing the palm origin."""
    zeros = torch.zeros_like(wrj2)
    q_wrj2 = quat_from_euler(zeros, wrj2, zeros)
    q_wrj1 = quat_from_euler(wrj1, zeros, zeros)
    wrist_quat = quat_mul(q_wrj2, q_wrj1)
    offset = torch.zeros((*wrj2.shape, 3), dtype=wrj2.dtype, device=wrj2.device)
    return wrist_quat, offset


def wrist_delta_from_rotation(axis, angle_deg, device, dtype=torch.float32):
    """Map a requested palm rotation to sr wrist joints within URDF limits."""
    axis = axis.to(device=device, dtype=dtype)
    angle = torch.deg2rad(torch.tensor(angle_deg, dtype=dtype, device=device))

    # WRJ2 rotates around local Y, WRJ1 rotates around local X. There is no
    # wrist Z DOF; Z components are intentionally left for forearm compensation.
    wrj2 = angle * axis[1]
    wrj1 = angle * axis[0]
    wrj2 = wrj2.clamp(
        min=math.radians(-30.0),
        max=math.radians(10.0),
    )
    wrj1 = wrj1.clamp(
        min=math.radians(-45.0),
        max=math.radians(35.0),
    )
    return wrj2, wrj1


def quat_from_euler(roll, pitch, yaw):
    """Convert ZYX Euler components to quaternion [x, y, z, w]."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dim=-1)


def quat_slerp(q0, q1, t):
    """Shortest-path spherical interpolation for [x, y, z, w] quaternions."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    linear = sin_theta.abs() < 1e-6
    w0 = torch.sin((1.0 - t) * theta) / sin_theta.clamp(min=1e-6)
    w1 = torch.sin(t * theta) / sin_theta.clamp(min=1e-6)
    result = w0 * q0 + w1 * q1
    result = torch.where(linear, (1.0 - t) * q0 + t * q1, result)
    return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True)


def unwrap_angles(angles, time_dim=0):
    """Remove +/-pi discontinuities along a selected time dimension."""
    if angles.shape[time_dim] < 2:
        return angles
    moved = angles.movedim(time_dim, 0)
    delta = moved[1:] - moved[:-1]
    wrapped_delta = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
    # Keep an exact +pi step positive instead of arbitrarily changing its sign.
    wrapped_delta = torch.where(
        (wrapped_delta == -math.pi) & (delta > 0), -wrapped_delta, wrapped_delta
    )
    result = moved.clone()
    result[1:] = moved[0] + torch.cumsum(wrapped_delta, dim=0)
    return result.movedim(0, time_dim)


def nearest_equivalent_angles(target, reference):
    """Choose the 2*pi-equivalent target closest to reference."""
    return reference + torch.remainder(target - reference + math.pi, 2 * math.pi) - math.pi


def expand_reference(ref, num_envs):
    """Copy the unchanged hand trajectory for every environment."""
    return {
        key: value.unsqueeze(0).expand(num_envs, *value.shape).clone()
        for key, value in ref.items()
    }


def split_traj_rot_args(argv):
    """Remove --traj_rot and its values so child runs can add one angle."""
    result = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--traj_rot":
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue
        if arg.startswith("--traj_rot="):
            index += 1
            continue
        result.append(arg)
        index += 1
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", type=str, default="sr_shadow_hand",
                    choices=list(HAND_CONFIGS.keys()),
                    help="Hand model to use. Choices: " + ", ".join(HAND_CONFIGS.keys()))
    ap.add_argument("--num_envs", type=int, default=16,
                    help="Number of environments in each independent simulation")
    ap.add_argument("--object_list", default="union_ycb_unidex/example.yaml")
    ap.add_argument("--ref_file", type=str, default=None,
                    help="Override reference trajectory file (default: from hand config)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max_len", type=int, default=None,
                    help="Episode length override; default resets when the trajectory ends")
    ap.add_argument("--decimation", type=int, default=20,
                    help="Physics steps per trajectory timestep (task default: 20)")
    ap.add_argument("--lift_frame", type=int, default=11,
                    help="First trajectory frame that lifts the wrist")
    ap.add_argument("--grasp_hold_steps", type=int, default=0,
                    help="Extra trajectory timesteps to hold the closed grasp before lifting")
    ap.add_argument("--no_reaching_plan", action="store_true",
                    help="Start directly at reference frame 0")
    ap.add_argument("--reaching_speed", type=float, default=1.0,
                    help="Approach speed multiplier relative to tasks/grasp.py")
    ap.add_argument("--hand_stiffness", type=float, default=600.0)
    ap.add_argument("--hand_damping", type=float, default=20.0)
    ap.add_argument("--contact_friction", type=float, default=1.0)
    ap.add_argument("--hand_vhacd", action="store_true",
                    help="Enable convex decomposition for hand collision meshes")
    ap.add_argument("--gpu_pipeline", action="store_true",
                    help="Use GPU PhysX and GPU pipeline. Default matches play_hand_only.py CPU pipeline.")
    ap.add_argument(
        "--traj_rot", type=float, nargs="+", default=[0.0],
        help="Rotate the hand trajectory around the object; multiple values run separate sims"
    )
    ap.add_argument("--rot_axis", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                    metavar=("X", "Y", "Z"),
                    help="Trajectory rotation axis through the object center; use 1 0 0 or 0 1 0 for side-grasp tilt")
    ap.add_argument(
        "--rotation_mode",
        choices=["axis_tilt", "orbit_xy", "base_only", "sr_wrist_tilt"],
        default="axis_tilt",
        help="axis_tilt rotates the full trajectory around a table-plane axis through the object"
    )
    ap.add_argument(
        "--face_object",
        action="store_true",
        help="After rotating the trajectory, correct orientation so the frame-0 hand-facing axis points at the object"
    )
    ap.add_argument(
        "--episodes", type=int, default=0,
        help="Stop after this many episodes; 0 runs until the viewer is closed"
    )
    args = ap.parse_args()

    if args.episodes < 0:
        ap.error("--episodes must be non-negative")
    if args.rotation_mode == "sr_wrist_tilt" and args.hand not in (
        "sr_shadow_hand", "new_sr_hand_simple"
    ):
        ap.error("--rotation_mode sr_wrist_tilt requires a hand with WRJ2/WRJ1")
    if len(args.traj_rot) > 1:
        if os.environ.get("PLAY_TRA_ROT_SINGLE_SIM") == "1":
            ap.error("an individual sim accepts exactly one --traj_rot value")
        if args.headless and args.episodes == 0:
            ap.error("multi-angle headless runs require --episodes > 0")
        child_argv = split_traj_rot_args(sys.argv[1:])
        child_env = os.environ.copy()
        child_env["PLAY_TRA_ROT_SINGLE_SIM"] = "1"
        for run_index, angle in enumerate(args.traj_rot, start=1):
            print(
                f"\n=== Trajectory rotation run {run_index}/{len(args.traj_rot)}: "
                f"{angle:g}deg ===",
                flush=True,
            )
            command = [
                sys.executable,
                os.path.abspath(__file__),
                *child_argv,
                "--traj_rot",
                str(angle),
            ]
            result = subprocess.run(command, env=child_env)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
        return

    hcfg = HAND_CONFIGS[args.hand]
    ref_path = args.ref_file or hcfg.ref_file
    n = args.num_envs
    if n < 1:
        ap.error("--num_envs must be at least 1")
    max_len = args.max_len
    decimation = args.decimation
    if decimation < 1:
        ap.error("--decimation must be at least 1")
    if args.reaching_speed <= 0:
        ap.error("--reaching_speed must be positive")
    if args.rotation_mode == "orbit_xy":
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    else:
        axis = torch.tensor(args.rot_axis, dtype=torch.float32)
    axis_norm = torch.linalg.vector_norm(axis)
    if axis_norm < 1e-8:
        ap.error("--rot_axis must be non-zero")
    axis /= axis_norm
    traj_rot = float(args.traj_rot[0])
    traj_rot_quat = quat_from_axis_angle(
        axis, torch.tensor([traj_rot], dtype=torch.float32)
    )[0]
    world_rot_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    table_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)

    # ─── Load trajectory data ─────────────────────────────────────────────────
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)
    for k in ref:
        ref[k] = torch.from_numpy(ref[k]).float()
    T = ref["wrist_initobj_pos"].shape[0]
    if not 1 <= args.lift_frame < T:
        ap.error(f"--lift_frame must be in [1, {T - 1}]")
    if args.grasp_hold_steps < 0:
        ap.error("--grasp_hold_steps must be non-negative")

    # The reference starts lifting immediately after the fingers close. Repeat
    # the last pre-lift pose so contacts can settle before upward motion.
    if args.grasp_hold_steps:
        hold_idx = args.lift_frame - 1
        for key in ref:
            value = ref[key]
            hold = value[hold_idx:hold_idx + 1].repeat(
                args.grasp_hold_steps, *([1] * (value.ndim - 1))
            )
            ref[key] = torch.cat(
                [value[:args.lift_frame], hold, value[args.lift_frame:]], dim=0
            )
        T = ref["wrist_initobj_pos"].shape[0]

    if hcfg.ref_qpos_prefix_zeros:
        prefix = hcfg.ref_qpos_prefix_zeros
        raw_hand_dofs = ref["hand_qpos"].shape[1]
        if raw_hand_dofs + prefix != hcfg.num_hand_dofs:
            raise ValueError(
                f"{hcfg.name} expects {hcfg.num_hand_dofs} hand DOFs after "
                f"prefixing {prefix} zeros, but reference has {raw_hand_dofs}"
            )
        zeros = ref["hand_qpos"].new_zeros(ref["hand_qpos"].shape[0], prefix)
        ref["hand_qpos"] = torch.cat([zeros, ref["hand_qpos"]], dim=1)

    traj_hand_dofs = ref["hand_qpos"].shape[1]
    print(f"Hand: {hcfg.name}  |  Trajectory: T={T}, hand_dofs={traj_hand_dofs}", flush=True)
    assert traj_hand_dofs == hcfg.num_hand_dofs, \
        f"Config expects {hcfg.num_hand_dofs} hand DOFs, trajectory has {traj_hand_dofs}"

    ref = expand_reference(ref, n)
    group_name = f"traj{traj_rot:+g}deg"
    print(
        f"Trajectory rotation: {traj_rot:+.2f}deg around {axis.tolist()}  |  "
        f"table normal: {table_normal.tolist()}  |  gravity: {gravity.tolist()}  |  "
        f"rotation mode: {args.rotation_mode}  |  envs: {n}",
        flush=True,
    )

    # Convert the reference to the palm-frame convention used by this replay,
    # then rotate the palm trajectory around the object in the horizontal world.
    ref_palm_quat = apply_quat_perm(ref["wrist_quat"], hcfg.wrist_quat_perm)
    quat_offset = torch.tensor(hcfg.wrist_quat_offset, dtype=ref_palm_quat.dtype)
    ref_palm_quat = quat_mul(ref_palm_quat, quat_offset)
    face_axis_local = None
    if args.face_object:
        frame0_to_obj = normalize_vec(-ref["wrist_initobj_pos"][0, 0])
        face_axis_local = quat_apply(
            quat_conjugate(ref_palm_quat[0, 0]),
            frame0_to_obj,
        )
    traj_rot_quat_expanded = traj_rot_quat.view(1, 1, 4).expand_as(ref_palm_quat)
    ref_palm_quat = quat_mul(traj_rot_quat_expanded, ref_palm_quat)
    ref["wrist_initobj_pos"] = quat_apply(
        traj_rot_quat.view(1, 1, 4).expand_as(ref_palm_quat),
        ref["wrist_initobj_pos"],
    )
    if face_axis_local is not None:
        desired_to_obj = normalize_vec(-ref["wrist_initobj_pos"])
        current_to_obj = normalize_vec(
            quat_apply(
                ref_palm_quat,
                face_axis_local.view(1, 1, 3).expand_as(desired_to_obj),
            )
        )
        align_quat = quat_from_two_vectors(current_to_obj, desired_to_obj)
        ref_palm_quat = quat_mul(align_quat, ref_palm_quat)

    forearm_offset = None
    if args.rotation_mode == "sr_wrist_tilt":
        wrj2_delta, wrj1_delta = wrist_delta_from_rotation(
            axis, traj_rot, device=ref["hand_qpos"].device, dtype=ref["hand_qpos"].dtype
        )
        ref["hand_qpos"][:, :, 0] = (
            ref["hand_qpos"][:, :, 0] + wrj2_delta
        ).clamp(math.radians(-30.0), math.radians(10.0))
        ref["hand_qpos"][:, :, 1] = (
            ref["hand_qpos"][:, :, 1] + wrj1_delta
        ).clamp(math.radians(-45.0), math.radians(35.0))
        wrist_fk = wrist_fk_zero_offset if hcfg.name == "new_sr_hand_simple" else wrist_fk_sr
        wrist_quat, forearm_offset = wrist_fk(
            ref["hand_qpos"][:, :, 0],
            ref["hand_qpos"][:, :, 1],
        )
        ref_base_quat = quat_mul(ref_palm_quat, quat_conjugate(wrist_quat))
        print(
            "sr wrist rotation target: "
            f"WRJ2={math.degrees(wrj2_delta.item()):+.2f}deg, "
            f"WRJ1={math.degrees(wrj1_delta.item()):+.2f}deg",
            flush=True,
        )
    elif hcfg.name == "sr_shadow_hand":
        wrist_quat, forearm_offset = wrist_fk_sr(
            ref["hand_qpos"][:, :, 0],
            ref["hand_qpos"][:, :, 1],
        )
        ref_base_quat = quat_mul(ref_palm_quat, quat_conjugate(wrist_quat))
    else:
        ref_base_quat = ref_palm_quat
        offset_vec = torch.tensor(
            hcfg.palm_offset_from_forearm,
            dtype=ref["wrist_initobj_pos"].dtype,
            device=ref["wrist_initobj_pos"].device,
        )
        forearm_offset = offset_vec.view(1, 1, 3).expand(
            ref["wrist_initobj_pos"].shape[0],
            ref["wrist_initobj_pos"].shape[1],
            3,
        )

    # Convert the complete forearm trajectory once and keep each Euler component
    # continuous. Frame-by-frame conversion can jump between -pi and +pi,
    # causing a revolute base joint to make an unnecessary full turn.
    ref_base_angles = unwrap_angles(
        quat_to_base_angles(ref_base_quat, hcfg.base_rotation_order), time_dim=1
    )

    with open(os.path.join(ASSET_ROOT, args.object_list)) as f:
        obj_files = yaml.safe_load(f)
    print(f"Objects: {len(obj_files)} types", flush=True)

    # ─── IsaacGym ─────────────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = 1/60
    sp.substeps = 2
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(
        gravity[0].item(), gravity[1].item(), gravity[2].item()
    )
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 8
    sp.physx.num_velocity_iterations = 0
    sp.physx.num_threads = 4
    sp.physx.use_gpu = args.gpu_pipeline
    sp.physx.contact_offset = 0.002
    sp.physx.rest_offset = 0.0
    sp.physx.bounce_threshold_velocity = 0.2
    sp.physx.max_depenetration_velocity = 1000.0
    sp.physx.default_buffer_size_multiplier = 5.0
    sp.use_gpu_pipeline = args.gpu_pipeline

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    if sim is None:
        print("FAILED to create sim")
        sys.exit(1)

    viewer = None
    if not args.headless:
        print("Creating viewer...", flush=True)
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("No viewer, running headless")
        else:
            print("Viewer created.", flush=True)

    # ─── Hand asset ───────────────────────────────────────────────────────────
    ao = gymapi.AssetOptions()
    ao.fix_base_link = hcfg.fix_base_link
    # Preserve the visual origins and rotations authored in the URDF. Flipping
    # attachments rotates each mesh independently from its rigid body frame.
    ao.flip_visual_attachments = False
    ao.disable_gravity = True
    ao.collapse_fixed_joints = False
    ao.angular_damping = 0.01
    ao.use_physx_armature = True
    if args.hand_vhacd:
        ao.vhacd_enabled = True
        ao.vhacd_params = gymapi.VhacdParams()
        ao.vhacd_params.resolution = 300000
    ao.armature = 0.001
    ao.thickness = 0.001

    print(f"Loading hand asset: {hcfg.urdf_path}", flush=True)
    hand_asset = gym.load_asset(sim, ASSET_ROOT, hcfg.urdf_path, ao)
    if hand_asset is None:
        print(f"FAILED to load hand from {hcfg.urdf_path}")
        sys.exit(1)

    ndof = gym.get_asset_dof_count(hand_asset)
    nbodies = gym.get_asset_rigid_body_count(hand_asset)
    nshapes = gym.get_asset_rigid_shape_count(hand_asset)
    dnames = [gym.get_asset_dof_name(hand_asset, j) for j in range(ndof)]
    print(f"Hand asset: {nbodies} rigid bodies, {nshapes} collision shapes", flush=True)
    print(f"Hand DOFs: {ndof}  names={dnames}", flush=True)
    if nshapes == 0:
        raise RuntimeError("Hand asset contains no collision shapes")

    # Index maps
    n_base = hcfg.num_base_dofs
    n_traj = hcfg.num_hand_dofs

    # base_joint_names → DOF index
    base_name2dof = {name: dnames.index(name) for name in hcfg.base_joint_names}
    base_xyz = [
        base_name2dof[name] for name in ("baseJX", "baseJY", "baseJZ")
    ] if n_base > 0 else []
    base_rot = [
        base_name2dof[name]
        for name in ("baseJROLL", "baseJPITCH", "baseJYAW")
    ] if n_base > 0 else []

    # trajectory hand joint index → DOF index
    traj2dof = torch.zeros(n_traj, dtype=torch.long)
    for i, name in enumerate(hcfg.hand_joint_names):
        traj2dof[i] = dnames.index(name)

    # mimic child DOF → parent DOF
    mimic2parent = torch.full((ndof,), -1, dtype=torch.long)
    for c, p in hcfg.mimic.items():
        if c in dnames and p in dnames:
            mimic2parent[dnames.index(c)] = dnames.index(p)

    # ─── Object assets ────────────────────────────────────────────────────────
    obj_dir = os.path.join(args.object_list.split('/')[0], "urdf")
    obj_assets = []
    for asset_index, fn in enumerate(obj_files, start=1):
        print(f"Loading object asset {asset_index}/{len(obj_files)}: {fn}", flush=True)
        oo = gymapi.AssetOptions()
        oo.density = 500
        oo.fix_base_link = False
        oo.flip_visual_attachments = False
        oo.collapse_fixed_joints = True
        oo.thickness = 0.001
        # Dynamic triangle meshes need convex decomposition for reliable
        # PhysX contacts. This matches tasks/grasp.py's object asset setup.
        oo.vhacd_enabled = True
        oo.vhacd_params = gymapi.VhacdParams()
        oo.vhacd_params.resolution = 300000
        a = gym.load_asset(sim, ASSET_ROOT, os.path.join(obj_dir, fn), oo)
        if a:
            obj_assets.append(a)
        else:
            print(f"  skipped object asset: {fn}", flush=True)
    if not obj_assets:
        print("No objects loaded")
        sys.exit(1)

    # ─── Environments ─────────────────────────────────────────────────────────
    spacing = 1.2
    per_row = int(math.sqrt(n))
    el = gymapi.Vec3(-spacing, -spacing, 0)
    eu = gymapi.Vec3(spacing, spacing, spacing)

    td = gymapi.Vec3(0.8, 0.8, 0.02)
    table_options = gymapi.AssetOptions()
    table_options.fix_base_link = True
    table_options.disable_gravity = True
    box_asset = gym.create_box(sim, td.x, td.y, td.z, table_options)

    envs = []
    hand_handles = []
    obj_handles = []
    obj_init_pos = np.zeros((n, 3), dtype=np.float32)

    print(f"Creating {n} envs...", flush=True)
    for i in range(n):
        env = gym.create_env(sim, el, eu, per_row)
        col, row = i % per_row, i // per_row
        ox = col * spacing - 0.2
        oy = row * spacing - 0.2

        # Table
        tp = gymapi.Transform()
        tp.p = gymapi.Vec3(0.5 + ox, oy, 0)
        tp.r = gymapi.Quat(
            world_rot_quat[0].item(),
            world_rot_quat[1].item(),
            world_rot_quat[2].item(),
            world_rot_quat[3].item(),
        )
        gym.create_actor(env, box_asset, tp, "table", i, 0)

        # Hand
        hp = gymapi.Transform()
        hp.p = gymapi.Vec3(ox, oy, 0)
        hh = gym.create_actor(env, hand_asset, hp, "hand", i, 1)

        color = (0.20, 0.65, 1.00)
        for body_idx in range(nbodies):
            gym.set_rigid_body_color(
                env,
                hh,
                body_idx,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*color),
            )

        hand_shape_props = gym.get_actor_rigid_shape_properties(env, hh)
        for shape_prop in hand_shape_props:
            shape_prop.friction = args.contact_friction
            shape_prop.restitution = 0.0
        gym.set_actor_rigid_shape_properties(env, hh, hand_shape_props)

        # DOF properties (PD position control for all joints)
        pr = gym.get_actor_dof_properties(env, hh)
        for j in range(ndof):
            pr["driveMode"][j] = gymapi.DOF_MODE_POS
            name = dnames[j]
            if n_base > 0 and name in hcfg.base_joint_names:
                # Stiffer for base joints (positioning the hand)
                pr["stiffness"][j] = 10000.0
                pr["damping"][j] = 500.0
            elif (
                hcfg.name == "new_sr_hand_simple"
                and args.rotation_mode != "sr_wrist_tilt"
                and name in ("WRJ2", "WRJ1")
            ):
                # In normal trajectory replay WRJ2/WRJ1 are only structural
                # joints. Lock them hard so traj_rot=0 remains physically close
                # to shadow_simple's fixed wrist.
                pr["stiffness"][j] = 10000.0
                pr["damping"][j] = 500.0
            else:
                # Match tasks/grasp.py by default. Excess damping prevents the
                # fingers from reaching the commanded grasp before lifting.
                pr["stiffness"][j] = args.hand_stiffness
                pr["damping"][j] = args.hand_damping
            pr["friction"][j] = 0.01
            pr["armature"][j] = 0.001
        gym.set_actor_dof_properties(env, hh, pr)

        # Object
        op = gymapi.Transform()
        # Release along the table normal. The object and table receive the same
        # initial rotation, while gravity points into the table.
        release_distance = td.z / 2 + 0.1
        release_pos = torch.tensor(
            [tp.p.x, tp.p.y, tp.p.z], dtype=torch.float32
        ) + table_normal * release_distance
        op.p = gymapi.Vec3(
            release_pos[0].item(),
            release_pos[1].item(),
            release_pos[2].item(),
        )
        op.r = gymapi.Quat(
            world_rot_quat[0].item(),
            world_rot_quat[1].item(),
            world_rot_quat[2].item(),
            world_rot_quat[3].item(),
        )
        oh = gym.create_actor(env, obj_assets[i % len(obj_assets)], op, f"obj_{i}", i, -1)

        obj_shape_props = gym.get_actor_rigid_shape_properties(env, oh)
        for shape_prop in obj_shape_props:
            shape_prop.friction = args.contact_friction
            shape_prop.restitution = 0.0
        gym.set_actor_rigid_shape_properties(env, oh, obj_shape_props)

        envs.append(env)
        hand_handles.append(hh)
        obj_handles.append(oh)
        obj_init_pos[i] = release_pos.numpy()

    print(f"Created {n} envs, {len(obj_assets)} obj types", flush=True)
    print("Preparing sim...", flush=True)
    gym.prepare_sim(sim)

    # ─── Camera ───────────────────────────────────────────────────────────────
    if viewer is not None:
        gym.viewer_camera_look_at(viewer, None,
                                  gymapi.Vec3(1.8, -3.2, 3.0),
                                  gymapi.Vec3(1.8, 5.0, 0.0))

    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(n, ndof, 2)
    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_state = gymtorch.wrap_tensor(root_state_tensor).view(-1, 13)
    sim_device = root_state.device
    print(f"Torch sim tensors on: {sim_device}", flush=True)

    # Keep all runtime tensors on the same device as Isaac Gym's tensors. With
    # GPU pipeline enabled, gymtorch tensors are CUDA tensors and CPU targets or
    # index tensors will fail when passed back to Gym.
    ref = {
        key: value.to(device=sim_device)
        for key, value in ref.items()
    }
    ref_base_quat = ref_base_quat.to(device=sim_device)
    ref_base_angles = ref_base_angles.to(device=sim_device)
    forearm_offset = forearm_offset.to(device=sim_device)
    traj2dof = traj2dof.to(device=sim_device)
    mimic2parent = mimic2parent.to(device=sim_device)
    table_normal_t = table_normal.to(device=sim_device).float()
    traj_rot_quat_t = traj_rot_quat.to(device=sim_device).float()

    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    hand_actor_ids = torch.arange(n, dtype=torch.int32, device=sim_device) * 3 + 1
    object_actor_ids = torch.arange(n, dtype=torch.int32, device=sim_device) * 3 + 2
    hand_actor_origin_t = root_state[hand_actor_ids.long(), 0:3].clone()
    object_release_states = root_state[object_actor_ids.long()].clone()

    # ─── Root-state tensor (only needed for floating-base hands) ──────────────
    hand_root_ids = None
    if not hcfg.fix_base_link:
        hand_root_ids = torch.arange(n, dtype=torch.long, device=sim_device) * 3 + 1
        num_actors = gym.get_sim_actor_count(sim)
        print(f"Floating base mode: {num_actors} total actors, hand IDs: {hand_root_ids.cpu().tolist()}", flush=True)

    # ─── Per-env simulation state ─────────────────────────────────────────────
    env_t = torch.zeros(n, dtype=torch.long, device=sim_device)
    env_ok = torch.zeros(n, dtype=torch.float32, device=sim_device)
    env_contact = torch.zeros(n, dtype=torch.float32, device=sim_device)
    obj_init_pos_t = torch.from_numpy(obj_init_pos).float().to(device=sim_device)

    dof_tgt = torch.zeros((n * ndof,), dtype=torch.float32, device=sim_device)
    prev_dof_tgt = torch.zeros_like(dof_tgt)
    off = torch.arange(n, device=sim_device) * ndof

    default_dof_pos = None
    default_dof_pos_template = None
    raw_default_palm_local = None
    parked_dof_pos = None
    if hcfg.default_dof_pos is not None:
        default_dof_pos = torch.tensor(
            hcfg.default_dof_pos, dtype=torch.float32, device=sim_device
        )
        if default_dof_pos.numel() != ndof:
            raise ValueError(
                f"default_dof_pos has {default_dof_pos.numel()} values, asset has {ndof} DOFs"
            )
        if n_base > 0 and hcfg.align_default_base_to_ref:
            # sr_shadow_hand's hard-coded default base orientation is a forearm
            # pose, while the tracked trajectory is expressed in the corrected
            # palm/wrist frame. Start from reference frame 0 so initialization
            # and tracking use the same orientation convention.
            default_dof_pos[base_rot] = ref_base_angles[0, 0]
            for tj in range(n_traj):
                default_dof_pos[traj2dof[tj].item()] = ref["hand_qpos"][0, 0, tj]
        default_dof_pos_template = default_dof_pos.clone()
        if n_base > 0:
            raw_default_palm_local = torch.tensor(
                hcfg.default_dof_pos,
                dtype=torch.float32,
                device=sim_device,
            )[base_xyz].clone()
        parked_dof_pos = default_dof_pos_template.clone()
        if "baseJZ" in base_name2dof:
            parked_dof_pos[base_name2dof["baseJZ"]] = max(
                parked_dof_pos[base_name2dof["baseJZ"]].item(), 1.0
            )
        dof_state[:, :, 0] = parked_dof_pos
        dof_state[:, :, 1] = 0.0
        prev_dof_tgt.view(n, ndof)[:] = parked_dof_pos
        dof_tgt.view(n, ndof)[:] = parked_dof_pos
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
        gym.set_dof_position_target_tensor(
            sim, gymtorch.unwrap_tensor(prev_dof_tgt)
        )

    def build_rotated_default_dof_pos(env_ids_for_pose, object_positions):
        """Apply the same object-centered trajectory transform to start poses."""
        pose_count = len(env_ids_for_pose)
        pose = default_dof_pos_template.unsqueeze(0).repeat(pose_count, 1)
        if n_base <= 0:
            return pose
        env_ids_for_pose = env_ids_for_pose.to(
            device=sim_device, dtype=torch.long
        )
        object_local = object_positions - hand_actor_origin_t[env_ids_for_pose]
        start_rel = raw_default_palm_local.view(1, 3) - object_local
        rotated_start_rel = quat_apply(
            traj_rot_quat_t.view(1, 4).expand(pose_count, -1),
            start_rel,
        )
        palm_local = object_local + rotated_start_rel
        base_quat = base_angles_to_quat(
            pose[:, base_rot],
            hcfg.base_rotation_order,
        )
        offset_world = quat_apply(
            base_quat,
            forearm_offset[env_ids_for_pose, 0],
        )
        pose[:, base_xyz] = palm_local - offset_world
        return pose

    # Let objects settle exactly as tasks/grasp.py does before establishing the
    # object-relative trajectory origin.
    settle_steps = int(2.0 / sp.dt)
    print(f"Settling objects for {settle_steps} sim steps...", flush=True)
    for _ in range(settle_steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim)
    settled_object_states = root_state[object_actor_ids.long()].clone()
    obj_init_pos_t.copy_(settled_object_states[:, 0:3])
    obj_init_pos[:] = settled_object_states[:, 0:3].cpu().numpy()

    # Only initialize the hand after every object has settled. This prevents the
    # reaching plan from being built against a release pose that later slides.
    if default_dof_pos is not None:
        all_env_ids = torch.arange(n, dtype=torch.long, device=sim_device)
        default_dof_pos = build_rotated_default_dof_pos(
            all_env_ids,
            obj_init_pos_t,
        )
        prev_dof_tgt.view(n, ndof)[:] = default_dof_pos
        dof_tgt.view(n, ndof)[:] = default_dof_pos
        dof_state[:, :, 0] = default_dof_pos
        dof_state[:, :, 1] = 0.0
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
        gym.set_dof_position_target_tensor(
            sim, gymtorch.unwrap_tensor(prev_dof_tgt)
        )

    # Build the same kind of approach phase used by tasks/grasp.py. The base
    # moves to reference frame 0 before reference time starts, while the hand
    # holds its frame-0 pose.
    reaching_steps = 0
    reaching_start = prev_dof_tgt.view(n, ndof).clone()
    reaching_plan = None
    if n_base > 0 and not args.no_reaching_plan:
        wp0 = ref["wrist_initobj_pos"][:, 0]
        start_pos = reaching_start[:, base_xyz]
        target_pos = obj_init_pos_t - hand_actor_origin_t + wp0
        start_quat = base_angles_to_quat(
            reaching_start[:, base_rot],
            hcfg.base_rotation_order,
        )
        target_quat = ref_base_quat[:, 0]
        # Correct for mechanical offset between the base-positioned link and the
        # functional palm (e.g. sr_shadow_hand has forearm→WRJ2→wrist→WRJ1→palm).
        offset_world = quat_apply(target_quat, forearm_offset[:, 0])
        target_pos = target_pos - offset_world
        trans_steps = torch.ceil(
            torch.linalg.vector_norm(target_pos - start_pos, dim=1)
            / (0.04 * args.reaching_speed)
        )
        quat_dot = (start_quat * target_quat).sum(dim=1).abs().clamp(max=1.0)
        rot_angle = 2.0 * torch.acos(quat_dot)
        rot_steps = torch.ceil(
            rot_angle / (0.1 * args.reaching_speed)
        )
        reaching_steps = int(torch.maximum(trans_steps, rot_steps).max().item())
        reaching_steps = max(reaching_steps, 1)

        alpha = torch.arange(
            1, reaching_steps + 1, dtype=torch.float32, device=sim_device
        ).view(-1, 1, 1) / reaching_steps
        plan_pos = start_pos.unsqueeze(0) + alpha * (
            target_pos - start_pos
        ).unsqueeze(0)
        plan_quat = quat_slerp(
            start_quat.unsqueeze(0).expand(reaching_steps, -1, -1),
            target_quat.unsqueeze(0).expand(reaching_steps, -1, -1),
            alpha,
        )
        plan_base_angles = unwrap_angles(
            quat_to_base_angles(plan_quat, hcfg.base_rotation_order)
        )
        reaching_plan = reaching_start.unsqueeze(0).repeat(reaching_steps, 1, 1)
        reaching_plan[:, :, base_xyz] = plan_pos
        reaching_plan[:, :, base_rot] = plan_base_angles
        for tj in range(n_traj):
            reaching_plan[:, :, traj2dof[tj]] = (
                ref["hand_qpos"][:, 0, tj].unsqueeze(0)
            )
        for di in range(ndof):
            pi = mimic2parent[di].item()
            if pi >= 0:
                reaching_plan[:, :, di] = reaching_plan[:, :, pi]

        # Continue the tracking Euler sequence from the exact reaching branch.
        end_base_angles = plan_base_angles[-1]
        ref_base_angles = nearest_equivalent_angles(
            ref_base_angles, end_base_angles.unsqueeze(1)
        )
        print(f"Reaching plan: {reaching_steps} trajectory timesteps", flush=True)

    episode_len = reaching_steps + T if max_len is None else max_len
    if episode_len < 1:
        ap.error("--max_len must be at least 1")

    # ─── Main loop ────────────────────────────────────────────────────────────
    print("\nRunning...", flush=True)
    step = 0
    ep = 0

    while True:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break

        in_reaching = env_t < reaching_steps
        tracking_t = (env_t - reaching_steps).clamp(min=0, max=T - 1)
        ti = tracking_t

        # 1. Set DOF position targets
        dof_tgt.zero_()

        if n_base > 0:
            tracking_target = dof_tgt.view(n, ndof)
            env_ids = torch.arange(n, device=sim_device)
            wp = ref["wrist_initobj_pos"][env_ids, ti]
            base_angles = ref_base_angles[env_ids, ti]
            hq = ref["hand_qpos"][env_ids, ti]

            desired_pos = obj_init_pos_t - hand_actor_origin_t + wp
            # Correct for mechanical offset between base-positioned link and palm.
            base_quat = base_angles_to_quat(
                base_angles, hcfg.base_rotation_order
            )
            offset_world = quat_apply(base_quat, forearm_offset[env_ids, ti])
            desired_pos = desired_pos - offset_world
            tracking_target[:, base_name2dof["baseJX"]] = desired_pos[:, 0]
            tracking_target[:, base_name2dof["baseJY"]] = desired_pos[:, 1]
            tracking_target[:, base_name2dof["baseJZ"]] = desired_pos[:, 2]
            tracking_target[:, base_rot] = base_angles

            for tj in range(n_traj):
                tracking_target[:, traj2dof[tj]] = hq[:, tj]

            if reaching_steps:
                reach_idx = env_t.clamp(max=reaching_steps - 1)
                tracking_target[in_reaching] = reaching_plan[
                    reach_idx[in_reaching],
                    torch.arange(n, device=sim_device)[in_reaching],
                ]
        else:
            # Floating base mode: set root state + finger targets
            env_ids = torch.arange(n, device=sim_device)
            wp = ref["wrist_initobj_pos"][env_ids, ti]
            wq = ref_base_quat[env_ids, ti]
            hq = ref["hand_qpos"][env_ids, ti]

            desired_pos = obj_init_pos_t + wp
            offset_world = quat_apply(wq, forearm_offset[env_ids, ti])
            desired_pos = desired_pos - offset_world
            desired_quat = wq

            root_state = gym.acquire_actor_root_state_tensor(sim)
            root_state = gymtorch.wrap_tensor(root_state)
            root_state[hand_root_ids, 0:3] = desired_pos
            root_state[hand_root_ids, 3:7] = desired_quat
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))

            for tj in range(n_traj):
                di = traj2dof[tj].item()
                dof_tgt[off + di] = hq[:, tj]

        # Mimic-driven joints: copy parent target
        for di in range(ndof):
            pi = mimic2parent[di].item()
            if pi >= 0:
                dof_tgt[off + di] = dof_tgt[off + pi]

        # Match the task environment: one reference timestep spans multiple
        # physics steps, with smooth target interpolation between frames.
        for substep in range(decimation):
            alpha = (substep + 1) / decimation
            interp_tgt = prev_dof_tgt + alpha * (dof_tgt - prev_dof_tgt)
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(interp_tgt))
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Render every physics step, as the task environment does. Rendering
            # only after all decimation steps makes playback appear 20x faster.
            if viewer is not None:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

            # A nonzero object contact force confirms that collision response is
            # active, independently of whether the grasp succeeds.
            for i in range(n):
                object_states = gym.get_actor_rigid_body_states(
                    envs[i], obj_handles[i], gymapi.STATE_VEL
                )
                linear_vel = object_states["vel"]["linear"]
                speed_sq = (
                    linear_vel["x"] ** 2
                    + linear_vel["y"] ** 2
                    + linear_vel["z"] ** 2
                )
                if np.any(speed_sq > 1e-8):
                    env_contact[i] = 1.0

        prev_dof_tgt.copy_(dof_tgt)

        # 3. Check success as vertical clearance away from the horizontal table.
        gym.refresh_actor_root_state_tensor(sim)
        object_displacement = (
            root_state[object_actor_ids.long(), 0:3] - obj_init_pos_t
        )
        normal_clearance = torch.sum(
            object_displacement * table_normal_t.unsqueeze(0), dim=1
        )
        env_ok[normal_clearance > 0.1] = 1.0

        env_t += 1

        # 4. Reset finished episodes
        reset = (env_t >= episode_len)
        if reset.any():
            rid = reset.nonzero(as_tuple=False).squeeze(-1)
            sr = env_ok[rid].mean().item()
            contact_rate = env_contact[rid].mean().item()
            ep += 1
            print(
                f"  Step {step:6d}  Ep {ep:4d}  "
                f"{group_name}: {sr:.3f}  Object moved: {contact_rate:.3f}"
            )

            gym.refresh_actor_root_state_tensor(sim)
            reset_object_ids = object_actor_ids[rid]
            root_state[reset_object_ids.long()] = object_release_states[rid]
            root_state[reset_object_ids.long(), 7:13] = 0.0
            gym.set_actor_root_state_tensor_indexed(
                sim,
                gymtorch.unwrap_tensor(root_state),
                gymtorch.unwrap_tensor(reset_object_ids),
                reset_object_ids.numel(),
            )

            env_t[rid] = 0
            env_ok[rid] = 0.0
            env_contact[rid] = 0.0
            if parked_dof_pos is not None:
                gym.refresh_dof_state_tensor(sim)
                prev_dof_tgt.view(n, ndof)[rid] = parked_dof_pos
                dof_tgt.view(n, ndof)[rid] = parked_dof_pos
                dof_state[rid, :, 0] = parked_dof_pos
                dof_state[rid, :, 1] = 0.0
                gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(prev_dof_tgt)
                )

            # Repeat the same release-and-settle initialization every episode.
            old_obj_pos = obj_init_pos_t[rid].clone()
            for _ in range(settle_steps):
                gym.simulate(sim)
                gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim)
            new_settled_states = root_state[reset_object_ids.long()]
            obj_init_pos_t[rid] = new_settled_states[:, 0:3]
            obj_init_pos[rid.cpu().numpy()] = new_settled_states[:, 0:3].cpu().numpy()

            # Initialize the hand only after the new settled object center is
            # known. The next main-loop iteration starts the reaching plan.
            if default_dof_pos is not None:
                gym.refresh_dof_state_tensor(sim)
                default_dof_pos[rid] = build_rotated_default_dof_pos(
                    rid,
                    obj_init_pos_t[rid],
                )
                prev_dof_tgt.view(n, ndof)[rid] = default_dof_pos[rid]
                dof_tgt.view(n, ndof)[rid] = default_dof_pos[rid]
                dof_state[rid, :, 0] = default_dof_pos[rid]
                dof_state[rid, :, 1] = 0.0
                gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(prev_dof_tgt)
                )

            if reaching_plan is not None:
                settled_delta = obj_init_pos_t[rid] - old_obj_pos
                reaching_plan[:, rid, base_name2dof["baseJX"]] += (
                    settled_delta[:, 0].view(1, -1)
                )
                reaching_plan[:, rid, base_name2dof["baseJY"]] += (
                    settled_delta[:, 1].view(1, -1)
                )
                reaching_plan[:, rid, base_name2dof["baseJZ"]] += (
                    settled_delta[:, 2].view(1, -1)
                )

            if args.episodes and ep >= args.episodes:
                break

        step += 1

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done.")


if __name__ == "__main__":
    main()
