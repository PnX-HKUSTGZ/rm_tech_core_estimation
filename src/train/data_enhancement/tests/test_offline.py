"""通过公开接口测试训练变体、来源信息与数据集发布。"""

import errno
import os
from collections import Counter

import cv2
import numpy as np
import pytest
import yaml
from tech_core_data_enhancement.__main__ import main
from tech_core_data_enhancement.offline import augment_record, run
from tech_core_data_generator import (
    CameraConfig,
    LightState,
    Team,
    Viewpoint,
    geometry_contract,
    geometry_sha256,
    load_manifest,
)
from tech_core_data_generator.camera import project_points
from tech_core_data_generator.frame import Frame


def _frame(theta=0.0, renderer="blender"):
    """@brief 近距离目标，下方关键点靠近图像边界。"""
    camera = CameraConfig(96, 72)
    pose = Viewpoint(r=0.35, theta=theta).camera_from_core
    points, _, inside = project_points(np.asarray(geometry_contract()["points_m"]), camera, pose)
    depth = np.zeros((72, 96), np.float32)
    depth[15:55, 25:70] = 0.35
    rgb = np.zeros((72, 96, 3), np.uint8)
    rgb[depth > 0] = [190, 80, 30]
    return Frame(rgb, depth, camera, pose, points, 2 * inside, renderer, LightState.READY, Team.BLUE, 0, None)


@pytest.fixture
def dataset(tmp_path):
    """@brief 写出生成器格式的原图，覆盖训练、验证、测试与 MuJoCo 各划分。"""
    root = tmp_path / "raw"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    records = []
    for index, split in enumerate(("train", "val", "test", "regular", "stress")):
        renderer = "mujoco" if split in ("regular", "stress") else "blender"
        frame = _frame(0.1 * index, renderer)
        record = frame.to_record(f"{split}_original", split, f"view_{split}")
        assert cv2.imwrite(str(root / record["image"]), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        assert cv2.imwrite(str(root / record["mask"]), frame.mask.astype(np.uint8) * 255)
        records.append(record)
    manifest = {
        "schema_version": 1,
        "geometry": {**geometry_contract(), "sha256": geometry_sha256()},
        "generator": {"seed": 14, "version": "0.1.0"},
        "records": records,
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return root


def test_affine_projection_mask_and_bounds():
    """@brief 用光学重投影与实际变换后的轮廓校验标注。"""
    frame = _frame()
    original = frame.to_record("original", "train", "view")
    record, rgb, mask = augment_record(original, frame.rgb, frame.mask, "variant", 21)
    assert record["camera_from_core"] == original["camera_from_core"]
    pose = np.asarray(record["camera_from_core"])
    optical = geometry_contract()["points_m"] @ pose[:3, :3].T + pose[:3, 3]
    projected = optical @ np.asarray(record["camera_matrix"]).T
    np.testing.assert_allclose(record["pixels"], projected[:, :2] / projected[:, 2:], atol=1e-6, rtol=0)
    assert record["camera_matrix"] != original["camera_matrix"]

    affine = np.array(record["augmentation"]["affine"])
    expected = cv2.warpAffine(frame.mask.astype(np.uint8) * 255, affine[:2], (96, 72), flags=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(mask, expected)
    assert rgb.shape == frame.rgb.shape and rgb.dtype == np.uint8
    assert rgb[mask == 0].std() > 0
    y, x = np.nonzero(mask)
    labelled = np.asarray(record["pixels"])[np.asarray(record["visibility"]) > 0]
    union = np.vstack((np.column_stack((x, y)), labelled))
    bounds = [*union.min(axis=0), *np.minimum(union.max(axis=0) + 1, [96, 72])]
    np.testing.assert_allclose(record["bbox_xyxy"], bounds)


def test_occluded_outside_and_behind_camera():
    """@brief 保留被遮挡与相机后方的标注，并区分新增遮挡与裁剪。"""
    frame = _frame()
    source = frame.to_record("original", "train", "view")
    source["visibility"][0] = 1
    source["pixels"][-1], source["visibility"][-1] = None, 0
    record, _, _ = augment_record(source, frame.rgb, frame.mask, "variant", 21)
    x0, y0, x1, y1 = record["augmentation"]["occlusion"]["xyxy"]
    hidden, outside = [], []
    for point, old, new in zip(
        record["pixels"][:-1], source["visibility"][:-1], record["visibility"][:-1], strict=True
    ):
        x, y = point
        if not (0 <= x < 96 and 0 <= y < 72):
            outside.append(old)
            assert new == 0
        elif x0 <= x < x1 and y0 <= y < y1:
            hidden.append(old)
            assert new == 1
        else:
            assert new == old
    assert 2 in hidden and 2 in outside
    assert record["pixels"][-1] is None and record["visibility"][-1] == 0


def test_split_provenance_negatives_and_repeatability(dataset, tmp_path):
    """@brief 重复完整 CLI 运行，逐一比对 PNG、记录与日志是否与首次一致。"""
    first, second = tmp_path / "first", tmp_path / "second"
    for out in (first, second):
        main(["--data", str(dataset), "--out", str(out), "--copies", "2", "--negatives", "0.6", "--seed", "8"])
    for path in first.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    raw, result = load_manifest(dataset), load_manifest(first)
    assert result["geometry"] == raw["geometry"]
    assert result["generator"] == {
        **raw["generator"],
        "enhancement": {"policy": "standard", "copies": 2, "negatives": 0.6, "seed": 8},
    }
    records = result["records"]
    with (first / "records.partial.yaml").open() as journal:
        assert list(yaml.load_all(journal, Loader=yaml.CSafeLoader)) == records
    assert Counter(r["split"] for r in records) == {"train": 5, "val": 1, "test": 1, "regular": 1, "stress": 1}
    for source in raw["records"]:
        assert source in records
        for key in ("image", "mask"):
            assert (first / source[key]).samefile(dataset / source[key])
    derived = [r for r in records if r["source"] is not None]
    assert len(derived) == 4 and len({r["id"] for r in records}) == len(records)
    assert all(r["source"] == "train_original" and r["source_view"] == "view_train" for r in derived)
    negatives = [r for r in derived if r["negative"]]
    assert len(negatives) == 2
    for row in negatives:
        assert row["camera_from_core"] is row["pixels"] is row["visibility"] is row["bbox_xyxy"] is None
        assert row["augmentation"]["background"] == "procedural"
        assert cv2.imread(str(first / row["image"])).std() > 0
        assert not cv2.imread(str(first / row["mask"]), cv2.IMREAD_GRAYSCALE).any()


def test_none_policy_and_copy_fallback(dataset, tmp_path, monkeypatch):
    """@brief policy 为 none 时保留原图，背景数量只按原图计数。"""

    def cross_device_link(*args, **kwargs):
        """@brief 模拟源与目标位于不同挂载点时的文件系统错误。"""
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(os, "link", cross_device_link)
    out = tmp_path / "none"
    run(dataset, out, copies=3, negatives=1.0, policy="none", seed=1)
    originals = load_manifest(dataset)["records"]
    records = load_manifest(out)["records"]
    for original in originals:
        for key in ("image", "mask"):
            assert (out / original[key]).read_bytes() == (dataset / original[key]).read_bytes()
            assert not (out / original[key]).samefile(dataset / original[key])
    assert records[:-1] == originals
    assert records[-1]["negative"] and records[-1]["split"] == "train"
    with pytest.raises(FileExistsError, match="empty"):
        run(dataset, out, 0, 0, "none", 1)


@pytest.mark.parametrize(("copies", "negatives", "policy"), [(-1, 0, "standard"), (1, -0.1, "none")])
def test_invalid_options(dataset, tmp_path, copies, negatives, policy):
    """@brief 写输出之前拒绝非法的增强参数。"""
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run(dataset, out, copies, negatives, policy, 1)
    assert not out.exists()

