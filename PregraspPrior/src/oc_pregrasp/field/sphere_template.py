"""Fixed spherical template used by PregraspPrior outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SphereTemplate:
    dirs: np.ndarray
    tilt_angles: np.ndarray
    pitch_angles: np.ndarray
    faces: Optional[np.ndarray] = None
    neighbors: Optional[Tuple[Tuple[int, ...], ...]] = None
    kind: str = 'fibonacci'
    subdivision: Optional[int] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (len(self.dirs), len(self.tilt_angles), len(self.pitch_angles))

    @property
    def num_candidates(self) -> int:
        m, t, p = self.shape
        return m * t * p

    def unravel_index(self, flat_index: int) -> Tuple[int, int, int]:
        m_count, t_count, p_count = self.shape
        if flat_index < 0 or flat_index >= self.num_candidates:
            raise IndexError(f'flat_index out of range: {flat_index}')
        m = flat_index // (t_count * p_count)
        rest = flat_index % (t_count * p_count)
        t = rest // p_count
        p = rest % p_count
        return m, t, p

    def ravel_index(self, dir_id: int, tilt_id: int, pitch_id: int) -> int:
        _, t_count, p_count = self.shape
        return int(dir_id) * t_count * p_count + int(tilt_id) * p_count + int(pitch_id)


def fibonacci_sphere(num_dirs: int = 256) -> np.ndarray:
    """Generate approximately uniform unit directions on a sphere."""
    if num_dirs < 1:
        raise ValueError('num_dirs must be positive')
    indices = np.arange(num_dirs, dtype=np.float32)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / float(num_dirs)
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = indices * golden_angle
    dirs = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=-1)
    return dirs.astype(np.float32)


def _icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    vertices = np.asarray(
        [
            [-1.0, phi, 0.0],
            [1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [1.0, -phi, 0.0],
            [0.0, -1.0, phi],
            [0.0, 1.0, phi],
            [0.0, -1.0, -phi],
            [0.0, 1.0, -phi],
            [phi, 0.0, -1.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.asarray(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _subdivide_icosphere(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vertex_list = [v.astype(np.float32) for v in vertices]
    midpoint_cache = {}

    def midpoint_index(i: int, j: int) -> int:
        key = tuple(sorted((int(i), int(j))))
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        midpoint = vertex_list[key[0]] + vertex_list[key[1]]
        midpoint = midpoint / max(float(np.linalg.norm(midpoint)), 1e-8)
        vertex_list.append(midpoint.astype(np.float32))
        index = len(vertex_list) - 1
        midpoint_cache[key] = index
        return index

    new_faces = []
    for i, j, k in faces:
        a = midpoint_index(i, j)
        b = midpoint_index(j, k)
        c = midpoint_index(k, i)
        new_faces.extend(
            [
                [i, a, c],
                [j, b, a],
                [k, c, b],
                [a, b, c],
            ]
        )
    return np.asarray(vertex_list, dtype=np.float32), np.asarray(new_faces, dtype=np.int64)


def geodesic_sphere(subdivision: int = 2) -> Tuple[np.ndarray, np.ndarray, Tuple[Tuple[int, ...], ...]]:
    """Generate an icosphere and its 1-ring vertex adjacency."""
    if subdivision < 0:
        raise ValueError('subdivision must be non-negative')
    vertices, faces = _icosahedron()
    for _ in range(int(subdivision)):
        vertices, faces = _subdivide_icosphere(vertices, faces)
    neighbors = build_vertex_neighbors(len(vertices), faces)
    return vertices.astype(np.float32), faces.astype(np.int64), neighbors


def build_vertex_neighbors(num_vertices: int, faces: np.ndarray) -> Tuple[Tuple[int, ...], ...]:
    adjacency = [set() for _ in range(num_vertices)]
    for i, j, k in np.asarray(faces, dtype=np.int64):
        adjacency[int(i)].update([int(j), int(k)])
        adjacency[int(j)].update([int(i), int(k)])
        adjacency[int(k)].update([int(i), int(j)])
    return tuple(tuple(sorted(items)) for items in adjacency)


def make_geodesic_sphere_template(
    subdivision: int = 2,
    tilt_angles=(0.0, 15.0, 30.0),
    pitch_angles=(0.0, 15.0, 30.0),
) -> SphereTemplate:
    dirs, faces, neighbors = geodesic_sphere(subdivision=subdivision)
    return SphereTemplate(
        dirs=dirs,
        tilt_angles=np.asarray(tilt_angles, dtype=np.float32),
        pitch_angles=np.asarray(pitch_angles, dtype=np.float32),
        faces=faces,
        neighbors=neighbors,
        kind='geodesic',
        subdivision=int(subdivision),
    )


def make_sphere_template(
    num_dirs: Optional[int] = None,
    tilt_angles=(0.0, 15.0, 30.0),
    pitch_angles=(0.0, 15.0, 30.0),
    kind: str = 'geodesic',
    subdivision: int = 2,
) -> SphereTemplate:
    if kind == 'geodesic':
        if num_dirs is not None:
            expected = 10 * (4 ** int(subdivision)) + 2
            if int(num_dirs) != expected:
                raise ValueError(
                    f'geodesic subdivision={subdivision} has {expected} dirs, '
                    f'but num_dirs={num_dirs}. Use --subdivision or --sphere-template fibonacci.'
                )
        return make_geodesic_sphere_template(
            subdivision=subdivision,
            tilt_angles=tilt_angles,
            pitch_angles=pitch_angles,
        )
    if kind != 'fibonacci':
        raise ValueError(f'Unknown sphere template kind: {kind}')
    if num_dirs is None:
        num_dirs = 256
    return SphereTemplate(
        dirs=fibonacci_sphere(num_dirs),
        tilt_angles=np.asarray(tilt_angles, dtype=np.float32),
        pitch_angles=np.asarray(pitch_angles, dtype=np.float32),
        kind='fibonacci',
    )


def nearest_direction(direction: np.ndarray, dirs: np.ndarray) -> Tuple[int, float]:
    direction = np.asarray(direction, dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError('direction norm is too small')
    unit = direction / norm
    dots = np.clip(dirs @ unit, -1.0, 1.0)
    index = int(np.argmax(dots))
    angle_deg = float(np.degrees(np.arccos(dots[index])))
    return index, angle_deg


def cone_direction_weights(
    direction: np.ndarray,
    dirs: np.ndarray,
    cone_angle_deg: float = 25.0,
) -> np.ndarray:
    """Cosine-tapered soft weights for directions inside a cone."""
    direction = np.asarray(direction, dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError('direction norm is too small')
    unit = direction / norm
    dots = np.clip(dirs @ unit, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    weights = np.zeros_like(angles, dtype=np.float32)
    inside = angles <= float(cone_angle_deg)
    if np.any(inside):
        weights[inside] = 0.5 * (1.0 + np.cos(np.pi * angles[inside] / float(cone_angle_deg)))
    return weights


def one_ring_direction_weights(
    direction: np.ndarray,
    template: SphereTemplate,
    sigma_deg: float = 8.0,
) -> Tuple[np.ndarray, int, float]:
    """Gaussian direction labels on nearest vertex + its 1-ring neighbors."""
    if template.neighbors is None:
        raise ValueError('one_ring_direction_weights requires a template with neighbors')
    nearest_id, nearest_angle = nearest_direction(direction, template.dirs)
    support_ids = [nearest_id] + list(template.neighbors[nearest_id])
    support_dirs = template.dirs[np.asarray(support_ids, dtype=np.int64)]

    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
    dots = np.clip(support_dirs @ direction, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    sigma = max(float(sigma_deg), 1e-6)
    support_weights = np.exp(-0.5 * (angles / sigma) ** 2).astype(np.float32)
    support_weights = support_weights / max(float(support_weights.sum()), 1e-8)

    weights = np.zeros(len(template.dirs), dtype=np.float32)
    weights[np.asarray(support_ids, dtype=np.int64)] = support_weights
    return weights, nearest_id, nearest_angle


if __name__ == '__main__':
    template = make_sphere_template()
    print(f'dirs: {template.dirs.shape}')
    print(f'kind: {template.kind}')
    print(f'subdivision: {template.subdivision}')
    if template.neighbors is not None:
        neighbor_counts = np.asarray([len(items) for items in template.neighbors], dtype=np.int64)
        print(f'neighbor_counts: min={neighbor_counts.min()}, max={neighbor_counts.max()}')
    print(f'tilt_angles: {template.tilt_angles.tolist()}')
    print(f'pitch_angles: {template.pitch_angles.tolist()}')
    print(f'field_shape: {template.shape}')
    print(f'num_candidates: {template.num_candidates}')
