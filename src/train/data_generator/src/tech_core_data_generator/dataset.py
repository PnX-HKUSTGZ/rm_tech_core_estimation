"""@brief Source-view sampling and completed YAML dataset I/O."""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .appearance import sample_appearance
from .camera import CameraConfig, sample_random
from .geometry import _digest, geometry_contract, geometry_sha256
from .lighting import LightState, Team
from .simulator import TechCoreSimulator


def load_manifest(directory: str | Path) -> dict:
    """@brief Load a completed dataset as specified in the data_generator Readme.

    @param[in] directory Dataset root containing manifest.yaml.
    @return Manifest with dataset-relative paths.
    @throws FileNotFoundError If the dataset has no final manifest.
    @throws ValueError For incompatible schema or geometry, or source views shared across splits.
    """
    with (Path(directory) / "manifest.yaml").open(encoding="utf-8") as file:
        manifest = yaml.load(file, Loader=yaml.CSafeLoader)
    if manifest["schema_version"] != 1:
        raise ValueError(f"Unsupported dataset schema version {manifest['schema_version']}")
    geometry = manifest["geometry"]
    if geometry["sha256"] != geometry_sha256() or _digest(geometry) != geometry["sha256"]:
        raise ValueError("Dataset geometry or SHA-256 differs from the installed geometry contract")
    owners = {}
    for record in manifest["records"]:
        view, split = record["source_view"], record["split"]
        if owners.setdefault(view, split) != split:
            raise ValueError(f"Source view {view!r} appears in multiple splits")
    return manifest


def require_empty(out: Path) -> None:
    """@brief Create the output directory, or accept an existing empty one.

    @param[in] out Requested output path.
    @throws FileExistsError If the directory contains any entries.
    """
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out}")


def _pose_key(camera_from_core) -> tuple:
    """@brief Compare stored and newly sampled camera poses at the reference holdout precision."""
    return tuple(np.round(camera_from_core, 12).ravel())


def generate_dataset(
    out: Path,
    renderer: str,
    counts: dict[str, int],
    seed: int,
    camera: CameraConfig,
    argv: list[str],
    *,
    samples: int = 8,
    appearance: str | None = None,
    exclude=(),
):
    """@brief Render independent source views and atomically publish their final manifest.

    @param[in] out New or empty dataset root.
    @param[in] renderer blender for training data or mujoco for holdouts.
    @param[in] counts Number of source views per train/val/test or regular/stress split.
    @param[in] seed Independent nonnegative run seed.
    @param[in] camera Original image calibration (pixel, rad, m).
    @param[in] argv CLI argument vector recorded in generator provenance.
    @param[in] samples Blender Cycles samples per pixel.
    @param[in] appearance Blender appearance policy (random or default).
    @param[in] exclude Completed datasets whose poses are reserved.
    """
    if min(counts.values()) < 0 or not any(counts.values()):
        raise ValueError("Split counts must be nonnegative and request at least one view")
    randomize = renderer == "blender" and appearance != "default"
    known = set()
    for directory in exclude:
        manifest = load_manifest(directory)
        known.update(_pose_key(row["camera_from_core"]) for row in manifest["records"] if not row["negative"])
    records = []
    states = list(LightState)
    offsets = {"train": 0, "val": 100003, "test": 200003, "regular": 0, "stress": 100003}
    require_empty(out)
    with (
        (out / "records.partial.yaml").open("x", encoding="utf-8") as journal,
        TechCoreSimulator(renderer, camera, blender_samples=samples) as simulator,
    ):
        (out / "images").mkdir()
        (out / "masks").mkdir()
        generator = {
            "version": version("tech-core-data-generator"),
            "argv": argv,
            "seed": seed,
            "blender_version": simulator.blender_version,
        }
        for split, count in counts.items():
            distribution = "training" if renderer == "blender" else split
            split_seed = seed + offsets[split]
            rng = np.random.default_rng(split_seed)
            # Keep the source camera and lamp time fixed when comparing appearance policies.
            appearance_rng = np.random.default_rng([split_seed, 1])
            for index in range(count):
                while True:
                    view = sample_random(rng, distribution)
                    pose = _pose_key(view.camera_from_core)
                    if pose not in known:
                        known.add(pose)
                        break
                settings = sample_appearance(appearance_rng, view) if randomize else None
                frame = simulator.render(
                    view,
                    states[index % len(states)],
                    Team.RED if index % 2 else Team.BLUE,
                    time_s=rng.uniform(0, 3),
                    appearance=settings,
                )
                identity = f"{renderer}_{seed}_{split}_{index:05d}"
                record = frame.to_record(identity, split, identity)
                Image.fromarray(frame.rgb).save(out / record["image"])
                Image.fromarray(frame.mask.astype(np.uint8) * 255).save(out / record["mask"])
                yaml.dump(record, journal, Dumper=yaml.CSafeDumper, explicit_start=True, sort_keys=False)
                journal.flush()
                records.append(record)
    manifest = {
        "schema_version": 1,
        "geometry": {**geometry_contract(), "sha256": geometry_sha256()},
        "generator": generator,
        "records": records,
    }
    temporary = out / "manifest.partial.yaml"
    with temporary.open("x", encoding="utf-8") as file:
        yaml.dump(manifest, file, Dumper=yaml.CSafeDumper, sort_keys=False)
    temporary.replace(out / "manifest.yaml")
