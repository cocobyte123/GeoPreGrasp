"""Read and write PregraspPrior dense field stores."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from oc_pregrasp.field.sphere_template import SphereTemplate

SCHEMA = 'PregraspPriorStore/v1'


def safe_object_key(name: str) -> str:
    key = str(name).strip().replace('/', '__')
    key = re.sub(r'[^A-Za-z0-9_.-]+', '_', key)
    key = key.strip('_')
    return key or 'object'


@dataclass(frozen=True)
class FieldEntry:
    object_key: str
    object_name: str
    field_path: str
    aliases: List[str]
    grasp_count: int
    used_grasp_count: int


def save_field_npz(
    output_root: str | Path,
    object_key: str,
    object_name: str,
    points: np.ndarray,
    target_field: np.ndarray,
    confidence: np.ndarray,
    template: SphereTemplate,
    grasp_count: int,
    used_grasp_count: int,
    aliases: Optional[Iterable[str]] = None,
    overwrite: bool = False,
) -> FieldEntry:
    output_root = Path(output_root).expanduser().resolve()
    field_dir = output_root / 'fields'
    field_dir.mkdir(parents=True, exist_ok=True)
    object_key = safe_object_key(object_key)
    field_path = field_dir / f'{object_key}.npz'
    if field_path.exists() and not overwrite:
        raise FileExistsError(f'Field file already exists: {field_path}')
    alias_list = sorted({str(item) for item in (aliases or []) if str(item)})
    np.savez_compressed(
        field_path,
        schema=np.asarray(SCHEMA),
        object_key=np.asarray(object_key),
        object_name=np.asarray(str(object_name)),
        aliases=np.asarray(alias_list, dtype=object),
        points=np.asarray(points, dtype=np.float32),
        target_field=np.asarray(target_field, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        sphere_dirs=np.asarray(template.dirs, dtype=np.float32),
        sphere_faces=np.asarray(template.faces if template.faces is not None else np.empty((0, 3)), dtype=np.int64),
        tilt_angles=np.asarray(template.tilt_angles, dtype=np.float32),
        pitch_angles=np.asarray(template.pitch_angles, dtype=np.float32),
        template_kind=np.asarray(template.kind),
        subdivision=np.asarray(-1 if template.subdivision is None else int(template.subdivision), dtype=np.int64),
        grasp_count=np.asarray(int(grasp_count), dtype=np.int64),
        used_grasp_count=np.asarray(int(used_grasp_count), dtype=np.int64),
    )
    return FieldEntry(
        object_key=object_key,
        object_name=str(object_name),
        field_path=str(field_path.relative_to(output_root)),
        aliases=alias_list,
        grasp_count=int(grasp_count),
        used_grasp_count=int(used_grasp_count),
    )


def write_manifest(
    output_root: str | Path,
    dataset: str,
    entries: Iterable[FieldEntry],
    config: Optional[Dict] = None,
) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': SCHEMA,
        'dataset': str(dataset),
        'root': str(output_root),
        'entries': [entry.__dict__ for entry in entries],
        'config': dict(config or {}),
    }
    path = output_root / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


def load_manifest(path: str | Path) -> Dict:
    path = Path(path).expanduser().resolve()
    return json.loads(path.read_text(encoding='utf-8'))


def load_field_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path).expanduser().resolve(), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


class FieldStore:
    """Lookup helper for generated field datasets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = load_manifest(self.root / 'manifest.json')
        self.entries = list(self.manifest.get('entries', []))
        self._by_name: Dict[str, Dict] = {}
        for entry in self.entries:
            names = [entry.get('object_key'), entry.get('object_name'), *entry.get('aliases', [])]
            for name in names:
                if name:
                    self._by_name[str(name)] = entry

    def entry_for(self, object_name: str) -> Dict:
        key = str(object_name)
        if key not in self._by_name:
            raise KeyError(f'Object not found in field store: {object_name}')
        return self._by_name[key]

    def load(self, object_name: str) -> Dict[str, np.ndarray]:
        entry = self.entry_for(object_name)
        return load_field_npz(self.root / entry['field_path'])
