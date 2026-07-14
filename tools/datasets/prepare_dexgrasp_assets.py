#!/usr/bin/env python3
"""Convert DexGrasp-family object assets into DemoGrasp asset layout.

Expected output layout:

  <output-root>/<dataset>/
    meshes/<object_id>.obj
    urdf/<object_id>.urdf
    pointclouds/<object_id>.npy
    train.yaml
    test.yaml
    all.yaml
    assets.yaml
    debug.yaml
    manifest.json

The generated YAML files can be used as:

  task.env.asset.multiObjectList="<dataset>/debug.yaml"
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np
import yaml


DEFAULT_SOURCE_ROOT = Path("/mnt/data1/zju/data/dexgrasp")
DEFAULT_OUTPUT_ROOT = Path("DemoGrasp_mine/assets")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root_name: str
    mesh_mode: str
    align_pointcloud_to_mesh: bool = False


DATASETS = {
    "realdex": DatasetSpec("realdex", "Realdex", "meshdata"),
    "dexgraspnet": DatasetSpec(
        "dexgraspnet",
        "dexgraspnet",
        "obj_scale_urdf",
        align_pointcloud_to_mesh=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RealDex/DexGraspNet objects for DemoGrasp."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"DexGrasp data root. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"DemoGrasp asset root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to convert.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=512,
        help="Number of xyz points saved for each object.",
    )
    parser.add_argument(
        "--sampling",
        choices=["stride", "random"],
        default="stride",
        help="Point downsampling method.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink", "hardlink"],
        default="symlink",
        help="How to place mesh files in the output meshes directory.",
    )
    parser.add_argument(
        "--debug-count",
        type=int,
        default=20,
        help="Number of objects written to debug.yaml.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only convert the first N available objects. 0 means no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing meshes, URDFs, pointclouds, and YAMLs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conversion summary without writing files.",
    )
    parser.add_argument(
        "--no-align-pointcloud-to-mesh",
        action="store_true",
        help=(
            "Disable per-object pointcloud bbox alignment. By default this is "
            "enabled for DexGraspNet because its pkl pointclouds are normalized "
            "larger than the scaled obj meshes."
        ),
    )
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_yaml(path: Path, values: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(values, f, sort_keys=False, allow_unicode=True)


def write_json(path: Path, value: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_urdf(path: Path, mesh_name: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    mesh_ref = escape(f"../meshes/{mesh_name}")
    text = f"""<?xml version="1.0"?>
<robot name="root">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="{mesh_ref}" scale="1.0000E+00 1.0000E+00 1.0000E+00" />
      </geometry>
      <material name="">
        <color rgba="7.50E-01 7.50E-01 7.50E-01 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="{mesh_ref}" scale="1.0000E+00 1.0000E+00 1.0000E+00" />
      </geometry>
    </collision>
  </link>
</robot>
"""
    path.write_text(text, encoding="utf-8")


def place_mesh(src: Path, dst: Path, link_mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "hardlink":
        os.link(src, dst)
    elif link_mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def obj_bbox(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mn = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    mx = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    found_vertex = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertex = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=np.float64,
            )
            mn = np.minimum(mn, vertex)
            mx = np.maximum(mx, vertex)
            found_vertex = True
    if not found_vertex:
        raise ValueError(f"OBJ has no vertices: {path}")
    return mn, mx


def align_points_to_mesh_bbox(points: np.ndarray, mesh_path: Path) -> tuple[np.ndarray, float]:
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    pc_min = xyz.min(axis=0)
    pc_max = xyz.max(axis=0)
    pc_center = (pc_min + pc_max) * 0.5
    pc_diag = float(np.linalg.norm(pc_max - pc_min))

    mesh_min, mesh_max = obj_bbox(mesh_path)
    mesh_center = ((mesh_min + mesh_max) * 0.5).astype(np.float32)
    mesh_diag = float(np.linalg.norm(mesh_max - mesh_min))

    if pc_diag <= 0.0 or mesh_diag <= 0.0:
        return xyz, 1.0
    scale = mesh_diag / pc_diag
    aligned = (xyz - pc_center.astype(np.float32)) * np.float32(scale) + mesh_center
    return aligned.astype(np.float32), scale


def sample_points(points: np.ndarray, count: int, sampling: str, rng: np.random.Generator) -> np.ndarray:
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    if len(xyz) == count:
        return xyz
    if len(xyz) > count:
        if sampling == "stride":
            idx = np.linspace(0, len(xyz) - 1, count, dtype=np.int64)
        else:
            idx = rng.choice(len(xyz), size=count, replace=False)
        return xyz[idx]

    repeats = int(np.ceil(count / max(len(xyz), 1)))
    tiled = np.tile(xyz, (repeats, 1))
    return tiled[:count]


def save_points(
    dst: Path,
    points: np.ndarray,
    count: int,
    sampling: str,
    rng: np.random.Generator,
    overwrite: bool,
) -> None:
    if dst.exists() and not overwrite:
        return
    np.save(dst, sample_points(points, count, sampling, rng))


def object_ids_from_mesh_dir(mesh_dir: Path) -> set[str]:
    return {path.stem for path in mesh_dir.glob("*.obj")}


def filter_split(values: Iterable[str], available: set[str]) -> list[str]:
    return [name for name in values if name in available]


def convert_dataset(
    spec: DatasetSpec,
    source_root: Path,
    output_root: Path,
    points: int,
    sampling: str,
    seed: int,
    link_mode: str,
    debug_count: int,
    limit: int,
    align_pointcloud_to_mesh: bool,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    dataset_root = source_root / spec.root_name
    mesh_dir = dataset_root / spec.mesh_mode
    pcd_path = dataset_root / "object_pcds_nors.pkl"
    grasp_path = dataset_root / "grasp.json"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
    if not pcd_path.exists():
        raise FileNotFoundError(f"Point cloud pickle not found: {pcd_path}")

    pcds = load_pickle(pcd_path)
    splits = load_json(grasp_path) if grasp_path.exists() else {}

    mesh_ids = object_ids_from_mesh_dir(mesh_dir)
    pcd_ids = set(pcds)
    all_available_ids = sorted(mesh_ids & pcd_ids)
    available_ids = all_available_ids[:limit] if limit > 0 else all_available_ids
    missing_pcd = sorted(mesh_ids - pcd_ids)
    missing_mesh = sorted(pcd_ids - mesh_ids)

    out_dir = output_root / spec.name
    out_mesh_dir = out_dir / "meshes"
    out_urdf_dir = out_dir / "urdf"
    out_pcd_dir = out_dir / "pointclouds"

    summary = {
        "dataset": spec.name,
        "source_root": str(dataset_root),
        "output_root": str(out_dir),
        "available_objects": len(all_available_ids),
        "selected_objects": len(available_ids),
        "mesh_without_pointcloud": len(missing_pcd),
        "pointcloud_without_mesh": len(missing_mesh),
        "points_per_object": points,
        "link_mode": link_mode,
        "align_pointcloud_to_mesh": align_pointcloud_to_mesh,
    }

    if dry_run:
        return summary

    out_mesh_dir.mkdir(parents=True, exist_ok=True)
    out_urdf_dir.mkdir(parents=True, exist_ok=True)
    out_pcd_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    urdf_names: list[str] = []
    pointcloud_scales: list[float] = []
    for object_id in available_ids:
        mesh_src = mesh_dir / f"{object_id}.obj"
        mesh_name = f"{object_id}.obj"
        urdf_name = f"{object_id}.urdf"
        object_points = pcds[object_id]
        if align_pointcloud_to_mesh:
            object_points, scale = align_points_to_mesh_bbox(object_points, mesh_src)
            pointcloud_scales.append(scale)

        place_mesh(mesh_src, out_mesh_dir / mesh_name, link_mode, overwrite)
        write_urdf(out_urdf_dir / urdf_name, mesh_name, overwrite)
        save_points(
            out_pcd_dir / f"{object_id}.npy",
            object_points,
            points,
            sampling,
            rng,
            overwrite,
        )
        urdf_names.append(urdf_name)

    split_outputs = {
        "assets.yaml": available_ids,
        "debug.yaml": available_ids[:debug_count],
    }
    split_map = {
        "train.yaml": "_train_split",
        "test.yaml": "_test_split",
        "all.yaml": "_all_split",
    }
    for output_name, split_key in split_map.items():
        if split_key in splits:
            split_outputs[output_name] = filter_split(splits[split_key], set(available_ids))
        elif output_name == "all.yaml":
            split_outputs[output_name] = available_ids

    for output_name, object_ids in split_outputs.items():
        write_yaml(
            out_dir / output_name,
            [f"{object_id}.urdf" for object_id in object_ids],
            overwrite,
        )

    summary["written_objects"] = len(urdf_names)
    if pointcloud_scales:
        scales = np.asarray(pointcloud_scales, dtype=np.float64)
        summary["pointcloud_mesh_alignment_scale"] = {
            "min": float(np.min(scales)),
            "median": float(np.median(scales)),
            "max": float(np.max(scales)),
        }
    summary["splits"] = {
        output_name: len(object_ids) for output_name, object_ids in split_outputs.items()
    }
    write_json(out_dir / "manifest.json", summary, overwrite)
    return summary


def main() -> None:
    args = parse_args()
    dataset_names = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    summaries = []
    for dataset_name in dataset_names:
        summary = convert_dataset(
            spec=DATASETS[dataset_name],
            source_root=args.source_root,
            output_root=args.output_root,
            points=args.points,
            sampling=args.sampling,
            seed=args.seed,
            link_mode=args.link_mode,
            debug_count=args.debug_count,
            limit=args.limit,
            align_pointcloud_to_mesh=(
                DATASETS[dataset_name].align_pointcloud_to_mesh
                and not args.no_align_pointcloud_to_mesh
            ),
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        summaries.append(summary)

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
