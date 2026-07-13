'''
DemoGrasp: Hand-only demo replay (CPU pipeline).

Replays a demonstration trajectory with just the hand (no arm).
Supports swappable hand configs — add new hands by creating a HandConfig entry.

Currently supported hands:
  shadow_simple  - Shadow Hand with 6 virtual base joints (right_with_base.urdf)

To add a new hand: create a HandConfig with its URDF, joint names, and mimic mapping.

Usage:
  python play_hand_only.py --hand shadow_simple --num_envs 16
  python play_hand_only.py --hand shadow_simple --num_envs 16 --headless
'''
import os
import sys
import argparse
import pickle
import yaml
import math
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
    wrist_quat_perm: tuple = (0, 1, 2, 3)  # (qx, qy, qz, qw) permutation+negation from traj to URDF frame.
                                            # e.g. (0, -2, 1, 3) means: qx'=qx, qy'=-qz, qz'=qy, qw'=qw
    wrist_quat_offset: tuple = (0, 0, 0, 1)  # fixed quaternion [x,y,z,w] applied AFTER perm, as right-multiply.
                                               # e.g. 90° around -Z: (0, 0, -0.7071068, 0.7071068)

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


def apply_quat_perm(q, perm):
    """Apply component permutation + negation to a quaternion [x,y,z,w].

    perm: 4-tuple (ix, iy, iz, iw) where negative means negate that component.
          e.g. (0, -2, 1, 3) → qx'=q[0], qy'=-q[2], qz'=q[1], qw'=q[3]
    """
    # Map: 0→q[...,0], 1→q[...,1], 2→q[...,2], 3→q[...,3]
    # Negative → negate the component
    idx = torch.tensor([abs(p) for p in perm], dtype=torch.long)
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


def unwrap_angles(angles):
    """Remove +/-pi discontinuities along the trajectory time dimension."""
    if angles.shape[0] < 2:
        return angles
    delta = angles[1:] - angles[:-1]
    wrapped_delta = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
    # Keep an exact +pi step positive instead of arbitrarily changing its sign.
    wrapped_delta = torch.where(
        (wrapped_delta == -math.pi) & (delta > 0), -wrapped_delta, wrapped_delta
    )
    result = angles.clone()
    result[1:] = angles[0] + torch.cumsum(wrapped_delta, dim=0)
    return result


def nearest_equivalent_angles(target, reference):
    """Choose the 2*pi-equivalent target closest to reference."""
    return reference + torch.remainder(target - reference + math.pi, 2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", type=str, default="shadow_simple",
                    choices=list(HAND_CONFIGS.keys()),
                    help="Hand model to use")
    ap.add_argument("--num_envs", type=int, default=16)
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
    args = ap.parse_args()

    hcfg = HAND_CONFIGS[args.hand]
    ref_path = args.ref_file or hcfg.ref_file
    n = args.num_envs
    max_len = args.max_len
    decimation = args.decimation
    if decimation < 1:
        ap.error("--decimation must be at least 1")
    if args.reaching_speed <= 0:
        ap.error("--reaching_speed must be positive")

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

    traj_hand_dofs = ref["hand_qpos"].shape[1]
    print(f"Hand: {hcfg.name}  |  Trajectory: T={T}, hand_dofs={traj_hand_dofs}")
    assert traj_hand_dofs == hcfg.num_hand_dofs, \
        f"Config expects {hcfg.num_hand_dofs} hand DOFs, trajectory has {traj_hand_dofs}"

    # Convert the complete wrist trajectory once and keep each Euler component
    # continuous. Frame-by-frame conversion can jump between -pi and +pi,
    # causing a revolute base joint to make an unnecessary full turn.
    ref_wrist_quat = apply_quat_perm(ref["wrist_quat"], hcfg.wrist_quat_perm)
    quat_offset = torch.tensor(hcfg.wrist_quat_offset, dtype=ref_wrist_quat.dtype)
    ref_wrist_quat = quat_mul(ref_wrist_quat, quat_offset)
    ref_wrist_euler = unwrap_angles(quat_to_euler(ref_wrist_quat))

    with open(os.path.join(ASSET_ROOT, args.object_list)) as f:
        obj_files = yaml.safe_load(f)
    print(f"Objects: {len(obj_files)} types")

    # ─── IsaacGym (CPU pipeline) ──────────────────────────────────────────────
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = 1/60
    sp.substeps = 2
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, -9.81)
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 8
    sp.physx.num_velocity_iterations = 0
    sp.physx.num_threads = 4
    sp.physx.use_gpu = False
    sp.physx.contact_offset = 0.002
    sp.physx.rest_offset = 0.0
    sp.physx.bounce_threshold_velocity = 0.2
    sp.physx.max_depenetration_velocity = 1000.0
    sp.physx.default_buffer_size_multiplier = 5.0
    sp.use_gpu_pipeline = False

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    if sim is None:
        print("FAILED to create sim")
        sys.exit(1)

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("No viewer, running headless")

    gp = gymapi.PlaneParams()
    gp.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, gp)

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

    hand_asset = gym.load_asset(sim, ASSET_ROOT, hcfg.urdf_path, ao)
    if hand_asset is None:
        print(f"FAILED to load hand from {hcfg.urdf_path}")
        sys.exit(1)

    ndof = gym.get_asset_dof_count(hand_asset)
    nbodies = gym.get_asset_rigid_body_count(hand_asset)
    nshapes = gym.get_asset_rigid_shape_count(hand_asset)
    dnames = [gym.get_asset_dof_name(hand_asset, j) for j in range(ndof)]
    print(f"Hand asset: {nbodies} rigid bodies, {nshapes} collision shapes")
    print(f"Hand DOFs: {ndof}  names={dnames}")
    if nshapes == 0:
        raise RuntimeError("Hand asset contains no collision shapes")

    # Index maps
    n_base = hcfg.num_base_dofs
    n_traj = hcfg.num_hand_dofs

    # base_joint_names → DOF index
    base_name2dof = {name: dnames.index(name) for name in hcfg.base_joint_names}

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
    for fn in obj_files:
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
    if not obj_assets:
        print("No objects loaded")
        sys.exit(1)

    # ─── Environments ─────────────────────────────────────────────────────────
    spacing = 1.2
    per_row = int(math.sqrt(n))
    el = gymapi.Vec3(-spacing, -spacing, 0)
    eu = gymapi.Vec3(spacing, spacing, spacing)

    td = gymapi.Vec3(0.8, 0.8, 0.02)
    box_asset = gym.create_box(sim, td.x, td.y, td.z, gymapi.AssetOptions())

    envs = []
    hand_handles = []
    obj_handles = []
    obj_z0 = np.zeros(n, dtype=np.float32)
    obj_init_pos = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        env = gym.create_env(sim, el, eu, per_row)
        col, row = i % per_row, i // per_row
        ox = col * spacing - 0.2
        oy = row * spacing - 0.2

        # Table
        tp = gymapi.Transform()
        tp.p = gymapi.Vec3(0.5 + ox, oy, 0)
        gym.create_actor(env, box_asset, tp, "table", i, 0)

        # Hand
        hp = gymapi.Transform()
        hp.p = gymapi.Vec3(ox, oy, 0)
        hh = gym.create_actor(env, hand_asset, hp, "hand", i, 1)

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
        # Release from a fixed height, then use the settled root position as the
        # object-relative trajectory origin.
        oz = tp.p.z + td.z / 2 + 0.1
        op.p = gymapi.Vec3(0.5 + ox, oy, oz)
        oh = gym.create_actor(env, obj_assets[i % len(obj_assets)], op, f"obj_{i}", i, -1)

        obj_shape_props = gym.get_actor_rigid_shape_properties(env, oh)
        for shape_prop in obj_shape_props:
            shape_prop.friction = args.contact_friction
            shape_prop.restitution = 0.0
        gym.set_actor_rigid_shape_properties(env, oh, obj_shape_props)

        envs.append(env)
        hand_handles.append(hh)
        obj_handles.append(oh)
        obj_z0[i] = oz
        obj_init_pos[i] = [0.5 + ox, oy, oz]

    print(f"Created {n} envs, {len(obj_assets)} obj types")
    gym.prepare_sim(sim)

    # ─── Root-state tensor (only needed for floating-base hands) ──────────────
    hand_root_ids = None
    if not hcfg.fix_base_link:
        hand_root_ids = torch.arange(n, dtype=torch.long) * 3 + 1
        num_actors = gym.get_sim_actor_count(sim)
        print(f"Floating base mode: {num_actors} total actors, hand IDs: {hand_root_ids.tolist()}")

    # ─── Camera ───────────────────────────────────────────────────────────────
    if viewer is not None:
        gym.viewer_camera_look_at(viewer, None,
                                  gymapi.Vec3(1.8, -3.2, 3.0),
                                  gymapi.Vec3(1.8, 5.0, 0.0))

    # ─── Per-env simulation state ─────────────────────────────────────────────
    env_t = torch.zeros(n, dtype=torch.long)
    env_ok = torch.zeros(n, dtype=torch.float32)
    env_contact = torch.zeros(n, dtype=torch.float32)
    obj_init_pos_t = torch.from_numpy(obj_init_pos).float()
    obj_z0_t = torch.from_numpy(obj_z0).float()

    dof_tgt = torch.zeros((n * ndof,), dtype=torch.float32)
    prev_dof_tgt = torch.zeros_like(dof_tgt)
    off = torch.arange(n) * ndof
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(n, ndof, 2)
    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_state = gymtorch.wrap_tensor(root_state_tensor).view(-1, 13)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    object_actor_ids = torch.arange(n, dtype=torch.int32) * 3 + 2
    object_release_states = root_state[object_actor_ids.long()].clone()

    # Let objects settle exactly as tasks/grasp.py does before establishing the
    # object-relative trajectory origin.
    settle_steps = int(2.0 / sp.dt)
    for _ in range(settle_steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim)
    settled_object_states = root_state[object_actor_ids.long()].clone()
    obj_init_pos_t.copy_(settled_object_states[:, 0:3])
    obj_z0_t.copy_(settled_object_states[:, 2])
    obj_init_pos[:] = settled_object_states[:, 0:3].cpu().numpy()
    obj_z0[:] = settled_object_states[:, 2].cpu().numpy()

    # Initialize interpolation from the actual configured pose instead of zero.
    if hcfg.default_dof_pos is not None:
        default_dof_pos = torch.tensor(hcfg.default_dof_pos, dtype=torch.float32)
        if default_dof_pos.numel() != ndof:
            raise ValueError(
                f"default_dof_pos has {default_dof_pos.numel()} values, asset has {ndof} DOFs"
            )
        prev_dof_tgt.view(n, ndof)[:] = default_dof_pos

    # Build the same kind of approach phase used by tasks/grasp.py. The base
    # moves to reference frame 0 before reference time starts, while the hand
    # holds its frame-0 pose.
    reaching_steps = 0
    reaching_start = prev_dof_tgt.view(n, ndof).clone()
    reaching_target = reaching_start.clone()
    reaching_plan = None
    if n_base > 0 and not args.no_reaching_plan:
        wp0 = ref["wrist_initobj_pos"][0]
        base_rot = [
            base_name2dof[name]
            for name in ("baseJROLL", "baseJPITCH", "baseJYAW")
        ]
        base_xyz = [base_name2dof[name] for name in ("baseJX", "baseJY", "baseJZ")]
        start_pos = reaching_start[:, base_xyz]
        target_pos = torch.stack([
            (0.5 + wp0[0]).expand(n),
            wp0[1].expand(n),
            obj_z0_t + wp0[2],
        ], dim=1)
        start_quat = quat_from_euler(
            reaching_start[:, base_rot[2]],
            reaching_start[:, base_rot[1]],
            reaching_start[:, base_rot[0]],
        )
        target_quat = ref_wrist_quat[0].unsqueeze(0).expand(n, -1)
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
            1, reaching_steps + 1, dtype=torch.float32
        ).view(-1, 1, 1) / reaching_steps
        plan_pos = start_pos.unsqueeze(0) + alpha * (
            target_pos - start_pos
        ).unsqueeze(0)
        plan_quat = quat_slerp(
            start_quat.unsqueeze(0).expand(reaching_steps, -1, -1),
            target_quat.unsqueeze(0).expand(reaching_steps, -1, -1),
            alpha,
        )
        plan_euler = unwrap_angles(quat_to_euler(plan_quat))
        reaching_plan = reaching_start.unsqueeze(0).repeat(reaching_steps, 1, 1)
        reaching_plan[:, :, base_xyz] = plan_pos
        reaching_plan[:, :, base_rot[0]] = plan_euler[:, :, 2]
        reaching_plan[:, :, base_rot[1]] = plan_euler[:, :, 1]
        reaching_plan[:, :, base_rot[2]] = plan_euler[:, :, 0]
        for tj in range(n_traj):
            reaching_plan[:, :, traj2dof[tj]] = ref["hand_qpos"][0, tj]
        for di in range(ndof):
            pi = mimic2parent[di].item()
            if pi >= 0:
                reaching_plan[:, :, di] = reaching_plan[:, :, pi]

        # Continue the tracking Euler sequence from the exact reaching branch.
        end_euler = plan_euler[-1, 0]
        ref_wrist_euler = nearest_equivalent_angles(
            ref_wrist_euler, end_euler
        )
        print(f"Reaching plan: {reaching_steps} trajectory timesteps")

    episode_len = reaching_steps + T if max_len is None else max_len
    if episode_len < 1:
        ap.error("--max_len must be at least 1")

    # ─── Main loop ────────────────────────────────────────────────────────────
    print("\nRunning...")
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
            wp = ref["wrist_initobj_pos"][ti]
            euler = ref_wrist_euler[ti]
            hq = ref["hand_qpos"][ti]

            tracking_target[:, base_name2dof["baseJX"]] = 0.5 + wp[:, 0]
            tracking_target[:, base_name2dof["baseJY"]] = wp[:, 1]
            tracking_target[:, base_name2dof["baseJZ"]] = obj_z0_t + wp[:, 2]
            tracking_target[:, base_name2dof["baseJROLL"]] = euler[:, 2]
            tracking_target[:, base_name2dof["baseJPITCH"]] = euler[:, 1]
            tracking_target[:, base_name2dof["baseJYAW"]] = euler[:, 0]

            for tj in range(n_traj):
                tracking_target[:, traj2dof[tj]] = hq[:, tj]

            if reaching_steps:
                reach_idx = env_t.clamp(max=reaching_steps - 1)
                tracking_target[in_reaching] = reaching_plan[
                    reach_idx[in_reaching], torch.arange(n)[in_reaching]
                ]
        else:
            # Floating base mode: set root state + finger targets
            wp = ref["wrist_initobj_pos"][ti]
            wq = apply_quat_perm(ref["wrist_quat"][ti], hcfg.wrist_quat_perm)
            q_off = torch.tensor(hcfg.wrist_quat_offset, dtype=wq.dtype, device=wq.device)
            wq = quat_mul(wq, q_off)
            hq = ref["hand_qpos"][ti]

            desired_pos = obj_init_pos_t + wp
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

        # 3. Check success
        for i in range(n):
            states = gym.get_actor_rigid_body_states(envs[i], obj_handles[i],
                                                     gymapi.STATE_POS)
            z = states['pose']['p']['z']
            if z - obj_z0[i] > 0.1:
                env_ok[i] = 1.0

        env_t += 1

        # 4. Reset finished episodes
        reset = (env_t >= episode_len)
        if reset.any():
            rid = reset.nonzero(as_tuple=False).squeeze(-1)
            sr = env_ok[rid].mean().item()
            contact_rate = env_contact[rid].mean().item()
            ep += 1
            print(f"  Step {step:6d}  Ep {ep:4d}  Success: {sr:.3f}  "
                  f"Object moved: {contact_rate:.3f}")

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
            if hcfg.default_dof_pos is not None:
                gym.refresh_dof_state_tensor(sim)
                prev_dof_tgt.view(n, ndof)[rid] = default_dof_pos
                dof_tgt.view(n, ndof)[rid] = default_dof_pos
                dof_state[rid, :, 0] = default_dof_pos
                dof_state[rid, :, 1] = 0.0
                gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
                gym.set_dof_position_target_tensor(
                    sim, gymtorch.unwrap_tensor(prev_dof_tgt)
                )

            # Repeat the same release-and-settle initialization every episode.
            old_obj_z = obj_z0_t[rid].clone()
            for _ in range(settle_steps):
                gym.simulate(sim)
                gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim)
            new_settled_states = root_state[reset_object_ids.long()]
            obj_init_pos_t[rid] = new_settled_states[:, 0:3]
            obj_z0_t[rid] = new_settled_states[:, 2]
            obj_init_pos[rid.numpy()] = new_settled_states[:, 0:3].cpu().numpy()
            obj_z0[rid.numpy()] = new_settled_states[:, 2].cpu().numpy()
            if reaching_plan is not None:
                reaching_plan[:, rid, base_name2dof["baseJZ"]] += (
                    obj_z0_t[rid] - old_obj_z
                ).view(1, -1)

        step += 1

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done.")


if __name__ == "__main__":
    main()
