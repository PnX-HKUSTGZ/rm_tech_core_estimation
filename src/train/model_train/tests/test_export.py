"""用真实的单目标检测网络与关键点网络完成离线部署发布目录的往返测试。"""

import numpy as np
import pytest
import torch
from tech_core_data_generator.geometry import geometry_sha256
from test_labels import dataset
from ultralytics.nn.tasks import DetectionModel

from tech_core_model_train.evaluate import evaluate
from tech_core_model_train.export import export, verify_release
from tech_core_model_train.labels import read_yaml, sha256, write_yaml
from tech_core_model_train.litehrnet.model import LiteKeypoints


def test_export_and_golden_replay(tmp_path):
    """@brief 端到端校验检出的核心，并拒绝偏移的金标准像素或不同的检测结果。"""
    torch.manual_seed(3)
    torch.set_num_threads(2)
    data, detect, lite = (tmp_path / s for s in ("data", "detect", "lite"))
    dataset(data)
    (detect / "weights").mkdir(parents=True)
    lite.mkdir()
    # 固定的类别偏置使关键点计算图可独立于训练运行或网络访问进行测试。
    model = DetectionModel("yolo11n.yaml", nc=1, verbose=False)
    with torch.no_grad():
        for branch in model.model[-1].cv3:
            branch[-1].weight.zero_()
            branch[-1].bias.fill_(2.0)
    identity = {"geometry_sha256": geometry_sha256()}
    weights = detect / "weights/best.pt"
    torch.save(dict(identity, model=model, train_args={"imgsz": 320, "task": "detect"}), weights)
    write_yaml(
        detect / "experiment.yaml", dict(identity, hyperparameters={"size": "n"}, weights_sha256=sha256(weights))
    )
    torch.save(dict(identity, variant=18, state_dict=LiteKeypoints().state_dict()), lite / "best.pt")
    write_yaml(lite / "experiment.yaml", dict(identity, weights_sha256=sha256(lite / "best.pt")))
    evaluation, release = tmp_path / "evaluation", tmp_path / "release"
    evaluate(data, evaluation, detector=detect, keypoints=lite, keypoint_threshold=0.25)
    export(detect, lite, evaluation, release, verify=True)
    meta = read_yaml(release / "model_meta.yaml")
    assert set(meta) == {"schema_version", "detector", "geometry", "keypoints", "provenance"}
    cases_file = release / "golden/cases.yaml"
    cases = read_yaml(cases_file)
    assert len(cases) == 2 and all(c["bbox_xyxy"] is not None for c in cases)
    assert np.shape(cases[0]["pixels"]) == (16, 2)
    cases[0]["pixels"][0][0] += 0.6
    write_yaml(cases_file, cases)
    with pytest.raises(ValueError, match="Keypoints differ"):
        verify_release(release)
    cases[0]["bbox_xyxy"] = None
    write_yaml(cases_file, cases)
    with pytest.raises(ValueError, match="Detection differs"):
        verify_release(release)
    meta["geometry"]["sha256"] = "changed"
    write_yaml(release / "model_meta.yaml", meta)
    with pytest.raises(ValueError, match="geometry"):
        verify_release(release)
