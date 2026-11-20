"""@brief 下标对齐的参考 PnP。"""

from dataclasses import dataclass, field
from threading import Lock

import cv2
import numpy as np
from tech_core_data_generator.geometry import geometry_contract

_RANSAC_LOCK = Lock()


@dataclass(frozen=True)
class KeypointModel:
    """@brief 来自数据集几何契约的有序核心系关键点。"""

    points: np.ndarray = field(
        default_factory=lambda: np.array(geometry_contract()["points_m"])
    )  ##< 核心系 XYZ，单位 m。


def _noncollinear(points):
    """@brief 检验第二奇异值相对于主方向尺度的大小。"""
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return bool(singular[0] > 1e-12 and singular[1] > singular[0] * 1e-7)


def estimate(model, pixels, valid, camera_matrix, distortion=None):
    """@brief 估计核心系在相机光学系中的位姿，使用参考实现的 3 pixel 一致性半径。

    @param[in] model 有序的核心系几何，单位 m。
    @param[in] pixels 下标对齐的原图观测，单位 pixel。
    @param[in] valid 经置信度与图像边界过滤后的逐点观测掩码。
    @param[in] camera_matrix 原始针孔相机内参，单位 pixel。
    @param[in] distortion 标准 OpenCV 畸变参数；针孔相机为 None。
    @return 齐次 camera_from_core（平移单位 m）；不可解帧返回 None。
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(pixels).all(axis=1)
    obj, img = model.points[valid], pixels[valid]
    if len(obj) < 4 or not _noncollinear(obj) or not _noncollinear(img):
        return None
    camera = np.asarray(camera_matrix, dtype=np.float64)
    dist = None if distortion is None else np.asarray(distortion, dtype=np.float64)
    robust = len(obj) >= 5
    groups, seeds = _hypotheses(obj, img, camera, dist, robust)
    for indices in groups:
        solve_obj, solve_img = obj[indices], img[indices]
        if not _noncollinear(solve_obj) or not _noncollinear(solve_img):
            continue
        methods = [cv2.SOLVEPNP_SQPNP]
        singular = np.linalg.svd(solve_obj - solve_obj.mean(axis=0), compute_uv=False)
        if singular[-1] <= singular[0] * 1e-7:
            methods.append(cv2.SOLVEPNP_IPPE)
        for method in methods:
            try:
                ok, rotations, translations, *_ = cv2.solvePnPGeneric(solve_obj, solve_img, camera, dist, flags=method)
            except cv2.error:
                continue
            if ok:
                seeds.extend((r, t, indices) for r, t in zip(rotations, translations, strict=True))

    threshold = 3.0
    best = None
    for rvec, tvec, support in seeds:
        if not _positive_depth(obj, rvec, tvec):
            continue
        for _ in range(4):
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj[support],
                    img[support],
                    camera,
                    dist,
                    rvec.copy(),
                    tvec.copy(),
                    criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-10),
                )
            except cv2.error:
                break
            if not _positive_depth(obj, rvec, tvec):
                break
            residual = _residuals(obj, img, camera, dist, rvec, tvec)
            inliers = np.flatnonzero(residual <= threshold) if robust else np.arange(len(obj))
            consistent = len(inliers) >= 4 and _noncollinear(obj[inliers]) and _noncollinear(img[inliers])
            if not consistent:
                break
            # 在同一组观测上为每个种子打分，离群点按截断代价计入。
            score = float(np.sum(np.minimum(residual**2, threshold**2) if robust else residual**2))
            if np.isfinite(score) and (best is None or score < best[0]):
                best = score, rvec.copy(), tvec.copy(), residual
            if not robust or np.array_equal(support, inliers):
                break
            support = inliers
    if best is None:
        return None
    _, rvec, tvec, residual = best
    if robust:
        rvec, tvec = _refine_huber(obj, img, camera, dist, rvec, tvec, residual)
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = np.asarray(tvec).reshape(3)
    return transform


def _hypotheses(obj, img, camera, dist, robust):
    """@brief 用标定后的 SQPnP/AP3P 及平面单应性生成联合种子与一致集种子。"""
    groups, seeds = [np.arange(len(obj))], []
    if not robust:
        return groups, seeds
    centered = obj - obj.mean(axis=0)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    planar = singular[-1] <= singular[0] * 1e-7
    # 除 AP3P 外，保留上游风格的假设。整体替换
    # 会在这种平面/立柱混合几何上丢失有用的解。
    for method in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_AP3P):
        try:
            with _RANSAC_LOCK:
                cv2.setRNGSeed(0)
                if method == cv2.SOLVEPNP_SQPNP and planar:
                    plane = np.ascontiguousarray(centered @ basis[:2].T)
                    undistorted = cv2.undistortPoints(img.reshape(-1, 1, 2), camera, dist, P=camera).reshape(-1, 2)
                    homography, mask = cv2.findHomography(
                        plane, undistorted, cv2.RANSAC, 3.0, maxIters=300, confidence=0.999
                    )
                    ok = homography is not None and np.isfinite(homography).all()
                    mask = None if mask is None else np.flatnonzero(mask.reshape(-1))
                    rvec = tvec = None
                else:
                    ok, rvec, tvec, mask = cv2.solvePnPRansac(
                        obj,
                        img,
                        camera,
                        dist,
                        iterationsCount=300,
                        reprojectionError=3.0,
                        confidence=0.999,
                        flags=method,
                    )
        except cv2.error:
            continue
        if ok and mask is not None and len(mask) >= 4:
            selected = np.sort(mask.reshape(-1))
            if not any(np.array_equal(selected, existing) for existing in groups):
                groups.append(selected)
            if rvec is not None:
                seeds.append((rvec, tvec, selected))
    # 五个或六个点只能构成少量留一子集。显式检查
    # 这些子集，而不是依赖随机采样碰巧选中。
    if len(obj) <= 6:
        for excluded in range(len(obj)):
            selected = np.delete(np.arange(len(obj)), excluded)
            if not any(np.array_equal(selected, existing) for existing in groups):
                groups.append(selected)
    return groups, seeds


def _refine_huber(obj, img, camera, dist, rvec, tvec, residual):
    """@brief 在邻近观测上做 Huber 加权的局部位姿细化。"""
    threshold = 3.0
    active = np.flatnonzero(residual <= 2 * threshold)
    original = (rvec, tvec)
    if len(active) < 4 or not _noncollinear(obj[active]) or not _noncollinear(img[active]):
        return original
    parameters = np.r_[rvec.reshape(3), tvec.reshape(3)]

    def _objective(value):
        """@brief 计算像素域 Huber 目标函数及位姿的解析雅可比。"""
        projection, jacobian = cv2.projectPoints(obj[active], value[:3], value[3:], camera, dist)
        residual_xy = projection.reshape(-1, 2) - img[active]
        norm = np.linalg.norm(residual_xy, axis=1)
        loss = np.where(norm <= threshold, norm**2, 2 * threshold * norm - threshold**2)
        return float(np.sum(loss)), residual_xy, jacobian[:, :6], norm

    # 邻近观测获得连续权重，远处离群点仍被排除。
    # 使用 OpenCV 解析雅可比，避免额外引入 SciPy 运行时依赖。
    for _ in range(8):
        cost, residual_xy, jacobian, norm = _objective(parameters)
        if not np.isfinite(cost) or not np.isfinite(jacobian).all():
            break
        weights = np.repeat(np.minimum(1.0, threshold / np.maximum(norm, 1e-10)), 2)
        normal = jacobian.T @ (weights[:, None] * jacobian)
        gradient = jacobian.T @ (weights * residual_xy.ravel())
        damping = 1e-6 * np.diag(np.maximum(np.diag(normal), 1.0))
        try:
            step = np.linalg.lstsq(normal + damping, gradient, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125):
            proposal = parameters - scale * step
            if _positive_depth(obj, proposal[:3], proposal[3:]) and _objective(proposal)[0] < cost:
                parameters, accepted = proposal, True
                break
        if not accepted:
            break
    rotation, translation = parameters[:3].reshape(3, 1), parameters[3:].reshape(3, 1)
    residual = _residuals(obj, img, camera, dist, rotation, translation)
    final_inliers = np.flatnonzero(residual <= threshold)
    if (
        len(final_inliers) < 4
        or not _noncollinear(obj[final_inliers])
        or not _noncollinear(img[final_inliers])
        or not _positive_depth(obj, rotation, translation)
    ):
        return original
    return rotation, translation


def _positive_depth(obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> bool:
    """@brief 对参与拟合位姿的每个观测做正深度（cheirality）检验。"""
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return False
    depth = (obj @ cv2.Rodrigues(rvec)[0].T + np.asarray(tvec).reshape(1, 3))[:, 2]
    return bool(np.isfinite(depth).all() and np.all(depth > 1e-9))


def _residuals(
    obj: np.ndarray, img: np.ndarray, camera: np.ndarray, dist: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    """@brief 返回原图像素下的欧氏重投影残差。"""
    projected = cv2.projectPoints(obj, rvec, tvec, camera, dist)[0].reshape(-1, 2)
    return np.linalg.norm(projected - img, axis=1)
