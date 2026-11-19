"""@brief Training-split image augmentation using the data_generator manifest contract."""

import shutil
from itertools import count
from pathlib import Path

import cv2
import numpy as np
import yaml
from tech_core_data_generator import load_manifest

from .photometric import _photometric


def background(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """@brief Draw a smooth procedural texture with coloured line segments.

    @param[in,out] rng NumPy generator for the texture.
    @param[in] width Image width (pixel).
    @param[in] height Image height (pixel).
    @return H×W×3 RGB uint8 background.
    """
    grid = rng.uniform(0, 180, (6, 8, 3)).astype(np.float32)
    image = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
    for _ in range(8):
        start, end = rng.integers([0, 0], [width, height], size=(2, 2)).tolist()
        cv2.line(image, start, end, rng.uniform(10, 220, 3).tolist(), int(rng.integers(1, 8)))
    return np.clip(image, 0, 255).astype(np.uint8)


def _derived_record(source: dict, identity: str, parameters: dict) -> dict:
    """@brief Assign output paths and provenance to a training derivative."""
    return {
        **source,
        "id": identity,
        "image": f"images/{identity}.png",
        "mask": f"masks/{identity}.png",
        "source": source["id"],
        "augmentation": parameters,
    }


def _bounds(mask: np.ndarray, pixels: list, visibility: list) -> list[float] | None:
    """@brief Bound the warped silhouette and in-frame labels with clipped exclusive upper edges (pixel)."""
    y, x = np.nonzero(mask)
    points = [point for point, flag in zip(pixels, visibility, strict=True) if flag]
    if points:
        px, py = np.asarray(points).T
        x, y = np.concatenate((x, px)), np.concatenate((y, py))
    if not len(x):
        return None
    height, width = mask.shape
    return [float(x.min()), float(y.min()), float(min(width, x.max() + 1)), float(min(height, y.max() + 1))]


def augment_record(
    source: dict, rgb: np.ndarray, mask: np.ndarray, identity: str, seed: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    """@brief Warp one target, replace its background and update its image-space labels.

    @param[in] source Training target record in the generator schema.
    @param[in] rgb Original H×W×3 RGB uint8 image.
    @param[in] mask Original H×W silhouette, nonzero on the target.
    @param[in] identity New record ID.
    @param[in] seed Seed for replaying this variant independently of the dataset.
    @return Derived record, augmented RGB image and warped binary silhouette (uint8, 0/255).
    @note Synthetic occlusion hides landmarks but preserves the projected target silhouette.
    """
    rng = np.random.default_rng(seed)
    height, width = mask.shape
    scale = rng.uniform(0.8, 1.15)
    size = np.array([width, height])
    offset = (1 - scale) * (size - 1) / 2 + rng.uniform(-0.08, 0.08, 2) * size
    affine = np.array([[scale, 0, offset[0]], [0, scale, offset[1]], [0, 0, 1]])
    foreground = cv2.warpAffine(rgb, affine[:2], (width, height))
    silhouette = cv2.warpAffine((mask != 0).astype(np.uint8), affine[:2], (width, height), flags=cv2.INTER_NEAREST)
    image = background(rng, width, height)
    cv2.copyTo(foreground, silhouette, image)
    params = {"seed": seed, "affine": affine.tolist(), "background": "procedural"}

    rect = None
    if rng.random() < 0.25:
        origin = rng.integers([0, 0], size)
        extent = np.maximum(1, (rng.uniform(0.05, 0.2, 2) * size).astype(int))
        x0, y0 = origin.tolist()
        x1, y1 = np.minimum(size, origin + extent).tolist()
        rect = [x0, y0, x1, y1]
        color = rng.integers(0, 256, 3).tolist()
        image[y0:y1, x0:x1] = color
        params["occlusion"] = {"xyxy": rect, "rgb": color}
    image, params["photometric"] = _photometric(image, rng)

    pixels, visibility = [], []
    for point, flag in zip(source["pixels"], source["visibility"], strict=True):
        if point is None:
            pixels.append(None)
            visibility.append(0)
            continue
        x, y = (scale * np.asarray(point) + offset).tolist()
        if not (0 <= x < width and 0 <= y < height):
            flag = 0
        elif flag and rect and x0 <= x < x1 and y0 <= y < y1:
            flag = 1
        pixels.append([x, y])
        visibility.append(flag)

    record = _derived_record(source, identity, params)
    record.update(
        camera_matrix=(affine @ source["camera_matrix"]).tolist(),
        pixels=pixels,
        visibility=visibility,
        bbox_xyxy=_bounds(silhouette, pixels, visibility),
    )
    return record, image, silhouette * 255


def negative_record(source: dict, identity: str, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """@brief Generate a background-only sample using a training view's image calibration.

    @param[in] source Training record supplying dimensions, calibration and source group.
    @param[in] identity New record ID.
    @param[in] seed Seed for the procedural texture.
    @return Negative record, RGB uint8 image and empty uint8 silhouette.
    """
    record = _derived_record(source, identity, {"background": "procedural", "seed": seed})
    record.update(
        negative=True,
        camera_from_core=None,
        pixels=None,
        visibility=None,
        bbox_xyxy=None,
        light_state=None,
        team=None,
        time=None,
        appearance=None,
    )
    rgb = background(np.random.default_rng(seed), source["width"], source["height"])
    return record, rgb, np.zeros(rgb.shape[:2], np.uint8)


def _copy_asset(data: Path, out: Path, relative: Path) -> None:
    """@brief Preserve a dataset-relative PNG byte-for-byte across filesystems."""
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(data / relative)
    except OSError:
        with (data / relative).open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)


def _save(out: Path, record: dict, rgb: np.ndarray, mask: np.ndarray) -> dict:
    """@brief Write both PNG assets before publishing their record."""
    for key, image in (("image", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), ("mask", mask)):
        _, encoded = cv2.imencode(".png", image)
        with (out / record[key]).open("xb") as file:
            file.write(encoded)
    return record


def _output_records(data, out, originals, copies, negatives, rng):
    """@brief Yield completed records, counting background negatives against all output training targets."""
    used_ids = {record["id"] for record in originals}
    names = (f"enhanced_{index:07d}" for index in count())
    ids = (name for name in names if name not in used_ids)
    targets = []
    for source in originals:
        for key in ("image", "mask"):
            _copy_asset(data, out, Path(source[key]))
        yield source
        if source["split"] != "train" or source["negative"]:
            continue
        targets.append(source)
        if copies:
            rgb = cv2.cvtColor(cv2.imread(str(data / source["image"])), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(data / source["mask"]), cv2.IMREAD_GRAYSCALE)
            for _ in range(copies):
                seed = int(rng.integers(2**63 - 1))
                yield _save(out, *augment_record(source, rgb, mask, next(ids), seed))
    target_count = len(targets) * (1 + copies)
    for index in range(round(negatives * target_count)):
        source = targets[index % len(targets)]
        seed = int(rng.integers(2**63 - 1))
        yield _save(out, *negative_record(source, next(ids), seed))


def run(data: Path, out: Path, copies: int, negatives: float, policy: str, seed: int) -> None:
    """@brief Write an enhanced dataset and publish manifest.yaml after all assets are complete.

    @param[in] data Completed source dataset directory.
    @param[in] out New or empty output directory.
    @param[in] copies Extra variants per training target under the standard policy.
    @param[in] negatives Ratio of new backgrounds to all output training targets, including variants.
    @param[in] policy Standard augmentation (standard) or only originals and backgrounds (none).
    @param[in] seed Nonnegative run seed.
    @throws ValueError If counts are negative or the source manifest is invalid.
    @throws FileExistsError If out contains any entries.
    """
    if copies < 0 or not negatives >= 0:
        raise ValueError("Expected nonnegative --copies and --negatives")
    rng = np.random.default_rng(seed)
    manifest = load_manifest(data)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out}")
    parameters = {"copies": copies, "negatives": negatives, "policy": policy, "seed": seed}
    manifest["generator"]["enhancement"] = parameters
    records = _output_records(data, out, manifest["records"], copies if policy == "standard" else 0, negatives, rng)
    manifest["records"] = []
    with (out / "records.partial.yaml").open("x", encoding="utf-8") as journal:
        (out / "images").mkdir()
        (out / "masks").mkdir()
        for record in records:
            yaml.dump(record, journal, Dumper=yaml.CSafeDumper, explicit_start=True, sort_keys=False)
            journal.flush()
            manifest["records"].append(record)
    temporary = out / "manifest.partial.yaml"
    with temporary.open("x", encoding="utf-8") as file:
        yaml.dump(manifest, file, Dumper=yaml.CSafeDumper, sort_keys=False)
    temporary.replace(out / "manifest.yaml")
