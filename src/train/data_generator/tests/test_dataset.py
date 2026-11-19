"""Installed entry points and the on-disk dataset boundary."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from tech_core_data_generator import geometry_contract, geometry_sha256, load_manifest
from tech_core_data_generator.__main__ import main


@pytest.fixture(scope="module")
def holdout(tmp_path_factory):
    """@brief Render three source views once for the on-disk contract tests."""
    out = tmp_path_factory.mktemp("holdout")
    main(
        [
            "mujoco",
            "--out",
            str(out),
            "--regular",
            "2",
            "--stress",
            "1",
            "--seed",
            "4",
            "--width",
            "120",
            "--height",
            "96",
        ]
    )
    return out


def test_generated_records_and_reprojections(holdout):
    """@brief Check the journal, PNGs and projections against the completed manifest."""
    manifest = load_manifest(holdout)
    assert manifest.keys() == {"geometry", "schema_version", "records", "generator"}
    assert manifest["geometry"] == {**geometry_contract(), "sha256": geometry_sha256()}
    generator = manifest["generator"]
    assert generator["version"] and generator["argv"][0] == "mujoco" and generator["seed"] == 4
    assert generator["blender_version"] is None
    records = manifest["records"]
    with (holdout / "records.partial.yaml").open() as file:
        assert list(yaml.load_all(file, Loader=yaml.CSafeLoader)) == records
    assert [r["split"] for r in records] == ["regular", "regular", "stress"]
    assert len({r["source_view"] for r in records}) == 3
    for record in records:
        assert record.keys() == {
            "id",
            "split",
            "source_view",
            "renderer",
            "image",
            "mask",
            "width",
            "height",
            "camera_matrix",
            "distortion",
            "camera_from_core",
            "pixels",
            "visibility",
            "bbox_xyxy",
            "light_state",
            "team",
            "time",
            "appearance",
            "augmentation",
            "source",
            "negative",
        }
        assert record["renderer"] == "mujoco" and not record["negative"]
        assert record["appearance"] is record["augmentation"] is record["source"] is None
        assert record["distortion"] == [0] * 5
        with Image.open(holdout / record["image"]) as rgb, Image.open(holdout / record["mask"]) as mask:
            assert rgb.mode == "RGB" and rgb.size == mask.size == (record["width"], record["height"])
            assert np.asarray(rgb).dtype == np.uint8
            assert np.unique(mask).tolist() == [0, 255]
            x0, y0, x1, y1 = mask.getbbox()
            left, top, right, bottom = record["bbox_xyxy"]
            assert left <= x0 < x1 <= right and top <= y0 < y1 <= bottom
        pose = np.array(record["camera_from_core"])
        optical = manifest["geometry"]["points_m"] @ pose[:3, :3].T + pose[:3, 3]
        projected = optical @ np.array(record["camera_matrix"]).T
        np.testing.assert_allclose(record["pixels"], projected[:, :2] / projected[:, 2:])


def test_excludes_known_poses(holdout, tmp_path):
    """@brief Exercise exclusion with both original and inherited source poses."""
    excluded = load_manifest(holdout)
    args = ["mujoco", "--out", str(tmp_path / "fresh"), "--regular", "2", "--stress", "1", "--seed", "4", "--exclude"]
    # An enhanced dataset has its own generator seed but retains its original source camera poses.
    excluded["generator"]["seed"] = 123
    (tmp_path / "manifest.yaml").write_text(yaml.dump(excluded, Dumper=yaml.CSafeDumper))
    main([*args, str(tmp_path)])
    previous = [r["camera_from_core"] for r in excluded["records"]]
    fresh = load_manifest(tmp_path / "fresh")["records"]
    assert len(fresh) == 3 and all(r["camera_from_core"] not in previous for r in fresh)


def test_partial_geometry_and_split_boundaries(holdout, tmp_path):
    """@brief Reject interrupted datasets, stale coordinates and leaked source views."""
    shutil.copy(holdout / "records.partial.yaml", tmp_path)
    with pytest.raises(FileNotFoundError, match="manifest.yaml"):
        load_manifest(tmp_path)
    with pytest.raises(FileExistsError, match="empty"):
        main(["mujoco", "--out", str(tmp_path)])
    manifest = load_manifest(holdout)
    path = tmp_path / "manifest.yaml"
    manifest["geometry"]["points_m"][0][0] += 0.001
    path.write_text(yaml.dump(manifest, Dumper=yaml.CSafeDumper))
    with pytest.raises(ValueError, match="geometry"):
        load_manifest(tmp_path)
    manifest = load_manifest(holdout)
    manifest["records"][2]["source_view"] = manifest["records"][0]["source_view"]
    path.write_text(yaml.dump(manifest, Dumper=yaml.CSafeDumper))
    with pytest.raises(ValueError, match="multiple splits"):
        load_manifest(tmp_path)


def test_blender_splits_random_and_studio_appearance(tmp_path):
    """@brief Change the studio policy while holding source views and lamp times fixed."""
    args = [
        "blender",
        "--seed",
        "12",
        "--train",
        "2",
        "--val",
        "1",
        "--test",
        "1",
        "--width",
        "96",
        "--height",
        "72",
        "--samples",
        "1",
    ]
    random, studio = tmp_path / "random", tmp_path / "studio"
    main([*args, "--out", str(random)])
    main([*args, "--out", str(studio), "--appearance", "default"])
    manifest = load_manifest(random)
    assert manifest["generator"]["blender_version"]
    assert [r["split"] for r in manifest["records"]] == ["train", "train", "val", "test"]
    for a, b in zip(manifest["records"], load_manifest(studio)["records"], strict=True):
        assert a["appearance"]["view_transform"] in ("Standard", "Raw") and b["appearance"] is None
        assert a["camera_from_core"] == b["camera_from_core"] and a["time"] == b["time"]
        assert Path(a["image"]).is_relative_to("images")
        with Image.open(random / a["mask"]) as mask_a, Image.open(studio / b["mask"]) as mask_b:
            np.testing.assert_array_equal(mask_a, mask_b)
