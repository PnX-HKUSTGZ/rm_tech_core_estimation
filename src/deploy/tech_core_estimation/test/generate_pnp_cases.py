"""在 PYTHONPATH=ref/rm27_tech_core_estimation_demo 下生成 OpenCV 4.6 参考用例。"""

from pathlib import Path

import cv2
import numpy as np
import yaml

from tech_core_keypoint_pos_estimation import (
    KeypointModel, KeypointObservation, PnPEstimator, PoseEstimationError,
)
from tech_core_sim import get_keypoints, sample_random


def main():
    """@brief 投影模拟器关键点，并保存 Python 解算器的结果。"""
    landmarks = get_keypoints()
    model = KeypointModel(landmarks.names, landmarks.points, landmarks.groups)
    solver = PnPEstimator(model)
    camera = np.array([[520., 0., 319.5], [0., 510., 239.5], [0., 0., 1.]])
    distortion = np.array([-0.12, 0.035, 0.001, -0.002, 0.004])
    rng = np.random.default_rng(2716)
    cases = []
    views = sample_random(40, seed=42, r_range=(0.3, 1.), theta_range=(-1.1, 1.1),
                          alpha_range=(-0.6, 0.75), target=(-0.1, 0., 0.13))
    # 四组用例分别覆盖：全部点可用、缺失分组、小支持集、不可解帧。
    scenarios = ("full", "full", "outliers", "outliers", "dropout", "planar", "four", "five", "six", "failure")
    sparse_counts = {"four": 4, "five": 5, "six": 6}
    for index, view in enumerate(views):
        pose = view.camera_from_core
        pixels = cv2.projectPoints(model.points, cv2.Rodrigues(pose[:3, :3])[0],
                                   pose[:3, 3], camera, distortion)[0].reshape(-1, 2)
        pixels += rng.normal(0, (0., 0.4, 1.1)[index % 3], pixels.shape)
        valid = np.ones(len(model.names), dtype=bool)
        scenario = scenarios[index % len(scenarios)]
        if scenario == "dropout":
            valid = rng.random(len(valid)) >= 0.3
        elif scenario == "planar":
            valid = np.array(model.groups) == "marker"
        elif scenario in sparse_counts:
            valid[:] = False
            valid[rng.choice(len(valid), sparse_counts[scenario], replace=False)] = True
        elif scenario == "failure":
            if index == 9:
                valid[3:] = False
            elif index == 19:
                pixels[:, 1] = pixels[:, 0]
            else:
                pixels = rng.uniform([0., 0.], [640., 480.], pixels.shape)
        if scenario in ("outliers", "dropout", "planar", "five", "six"):
            count = 1 if scenario in sparse_counts else 2
            selected = rng.choice(np.flatnonzero(valid), count, replace=False)
            pixels[selected] += rng.uniform(20., 65., (count, 2))
        case = {"id": index, "pixels": pixels.tolist(), "valid": valid.tolist()}
        obs = KeypointObservation(model.names, pixels, visible=valid)
        try:
            result = solver.estimate(obs, camera, distortion, subset="all")
        except PoseEstimationError:
            case["success"] = False
        else:
            case.update(success=True, camera_from_core=result.camera_from_core.tolist(),
                        inliers=[name in result.inlier_names for name in model.names],
                        reprojection_rmse=result.reprojection_rmse_px, inlier_rmse=result.inlier_rmse_px)
        cases.append(case)
    document = dict(points=model.points.tolist(), camera_matrix=camera.tolist(),
                    distortion=distortion.tolist(), cases=cases)
    path = Path(__file__).parent / "data" / "pnp_cases.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=None, width=120))


if __name__ == "__main__":
    main()
