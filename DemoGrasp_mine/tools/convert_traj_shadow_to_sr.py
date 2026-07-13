#!/usr/bin/env python3
"""Convert a shadow_simple (18-hand-DOF) reference trajectory to sr_shadow_hand
(24-hand-DOF) format.

shadow_simple order (18):
  FFJ4, FFJ3, FFJ2, MFJ4, MFJ3, MFJ2, RFJ4, RFJ3, RFJ2,
  LFJ5, LFJ4, LFJ3, LFJ2, THJ5, THJ4, THJ3, THJ2, THJ1

sr_shadow_hand order (24):
  WRJ2, WRJ1, FFJ4, FFJ3, FFJ2, FFJ1, MFJ4, MFJ3, MFJ2, MFJ1,
  RFJ4, RFJ3, RFJ2, RFJ1, LFJ5, LFJ4, LFJ3, LFJ2, LFJ1,
  THJ5, THJ4, THJ3, THJ2, THJ1

Usage:
  python tools/convert_traj_shadow_to_sr.py
"""

import numpy as np
import pickle
import os

def convert():
    src_path = "tasks/grasp_ref_shadow.pkl"
    dst_path = "tasks/grasp_ref_sr_shadow.pkl"

    with open(src_path, "rb") as f:
        ref = pickle.load(f)

    # Convert to numpy first for easier manipulation
    for k in ref:
        if not isinstance(ref[k], np.ndarray):
            ref[k] = np.array(ref[k])

    T = ref["wrist_initobj_pos"].shape[0]
    old_hand = np.array(ref["hand_qpos"])  # (T, 18)

    # Build new hand_qpos: (T, 24)
    new_hand = np.zeros((T, 24), dtype=np.float32)

    # WRJ2, WRJ1 → 0 (neutral wrist, demo has no wrist motion)
    new_hand[:, 0] = 0.0  # WRJ2
    new_hand[:, 1] = 0.0  # WRJ1

    # FFJ4, FFJ3, FFJ2 → shadow[0,1,2]
    new_hand[:, 2] = old_hand[:, 0]  # FFJ4
    new_hand[:, 3] = old_hand[:, 1]  # FFJ3
    new_hand[:, 4] = old_hand[:, 2]  # FFJ2
    new_hand[:, 5] = old_hand[:, 2]  # FFJ1 = mimic FFJ2

    # MFJ4, MFJ3, MFJ2 → shadow[3,4,5]
    new_hand[:, 6] = old_hand[:, 3]   # MFJ4
    new_hand[:, 7] = old_hand[:, 4]   # MFJ3
    new_hand[:, 8] = old_hand[:, 5]   # MFJ2
    new_hand[:, 9] = old_hand[:, 5]   # MFJ1 = mimic MFJ2

    # RFJ4, RFJ3, RFJ2 → shadow[6,7,8]
    new_hand[:, 10] = old_hand[:, 6]  # RFJ4
    new_hand[:, 11] = old_hand[:, 7]  # RFJ3
    new_hand[:, 12] = old_hand[:, 8]  # RFJ2
    new_hand[:, 13] = old_hand[:, 8]  # RFJ1 = mimic RFJ2

    # LFJ5, LFJ4, LFJ3, LFJ2 → shadow[9,10,11,12]
    new_hand[:, 14] = old_hand[:, 9]  # LFJ5
    new_hand[:, 15] = old_hand[:, 10] # LFJ4
    new_hand[:, 16] = old_hand[:, 11] # LFJ3
    new_hand[:, 17] = old_hand[:, 12] # LFJ2
    new_hand[:, 18] = old_hand[:, 12] # LFJ1 = mimic LFJ2

    # THJ5, THJ4, THJ3, THJ2, THJ1 → shadow[13,14,15,16,17]
    new_hand[:, 19] = old_hand[:, 13] # THJ5
    new_hand[:, 20] = old_hand[:, 14] # THJ4
    new_hand[:, 21] = old_hand[:, 15] # THJ3
    new_hand[:, 22] = old_hand[:, 16] # THJ2
    new_hand[:, 23] = old_hand[:, 17] # THJ1

    result = {
        "wrist_initobj_pos": ref["wrist_initobj_pos"],
        "wrist_quat": ref["wrist_quat"],
        "hand_qpos": new_hand,
        "obj_initobj_pos": ref.get("obj_initobj_pos",
                                   np.zeros((T, 3), dtype=np.float32)),
    }

    with open(dst_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Converted {src_path} → {dst_path}")
    print(f"  T={T}, hand_dofs: {old_hand.shape[1]} → {new_hand.shape[1]}")
    print(f"  WRJ2/WRJ1 set to 0 (neutral wrist)")
    print(f"  FFJ1/MFJ1/RFJ1/LFJ1 set to mimic J2")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    convert()
