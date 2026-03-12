"""
Align consecutive video frames assuming only rotation + translation (no scale or shear)
using two matched pairs of SIFT features.

For every pair of consecutive frames the script:
1) Detects SIFT keypoints/descriptors in each frame.
2) Matches descriptors and keeps the two best matches after a ratio test.
3) Computes rotation from the angle difference between the two line segments.
4) Computes translation after rotating the current points to the previous frame.
5) Optionally warps the second frame to align with the first and writes it to a video.

Example:
    python align_video.py --input Inputs/boat.mp4 --output aligned_boat.mp4 --save-matrices boat_transforms.npy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Hyperparameters
ANGLE_TOL_DEG = 0.5          # allowable difference between two rotation estimates (degrees)
DIST_TOL_REL = 0.1           # relative tolerance for segment length consistency under rotation
MIN_MATCHES_DEFAULT = 2      # minimum matches to attempt estimation
MAX_MATCHES_DEFAULT = 300    # cap on matches used for estimation
RANSAC_THRESH_DEFAULT = 3.0  # reprojection threshold in pixels
RANSAC_MAX_ITERS_DEFAULT = 1000
PAIRWISE_SMOOTH_RADIUS_DEFAULT = 0
PAIRWISE_ANGLE_THRESH_DEG_DEFAULT = 2.0
PAIRWISE_TRANS_THRESH_DEFAULT = 4.0
SKIP_IDENTITY_THRESH = 0.5


def normalize_angle(delta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while delta > np.pi:
        delta -= 2 * np.pi
    while delta < -np.pi:
        delta += 2 * np.pi
    return delta


def match_sift_points(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    min_matches: int = MIN_MATCHES_DEFAULT,
    max_matches: int = MAX_MATCHES_DEFAULT,
    use_ratio_test: bool = True,
    ratio: float = 0.4,
    sift: Optional[cv2.SIFT] = None,
    bf: Optional[cv2.BFMatcher] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match SIFT points and return (prev_pts, curr_pts) as Nx2 float arrays."""
    if sift is None:
        sift = cv2.SIFT_create()
    kp_prev, des_prev = sift.detectAndCompute(prev_gray, None)
    kp_curr, des_curr = sift.detectAndCompute(curr_gray, None)

    if des_prev is None or des_curr is None or len(kp_prev) < 2 or len(kp_curr) < 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    if use_ratio_test:
        if bf is None:
            bf = cv2.BFMatcher()
        knn = bf.knnMatch(des_curr, des_prev, k=2)
        matches = [m[0] for m in knn if len(m) == 2 and m[0].distance <= ratio * m[1].distance]
    else:
        if bf is None:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des_prev, des_curr)

    if len(matches) < max(min_matches, 4):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    if use_ratio_test:
        curr_pts = np.float32([kp_curr[m.queryIdx].pt for m in matches])
        prev_pts = np.float32([kp_prev[m.trainIdx].pt for m in matches])
    else:
        prev_pts = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([kp_curr[m.trainIdx].pt for m in matches])

    return prev_pts, curr_pts


def estimate_rigid_from_points(
    curr_pts: np.ndarray,
    prev_pts: np.ndarray,
    translation_only: bool = False,
    rotation_zero_thresh_deg: float = 1.0,
    ransac_thresh: float = RANSAC_THRESH_DEFAULT,
    ransac_max_iters: int = RANSAC_MAX_ITERS_DEFAULT,
) -> np.ndarray:
    """Estimate a rigid transform from point correspondences, optionally clamping rotation."""
    if len(curr_pts) < 2 or len(prev_pts) < 2:
        return np.eye(2, 3, dtype=np.float32)

    if translation_only:
        trans = prev_pts - curr_pts
        mean = np.mean(trans, axis=0)
        return np.array([[1.0, 0.0, float(mean[0])], [0.0, 1.0, float(mean[1])]], dtype=np.float32)

    _, inliers = cv2.estimateAffinePartial2D(
        curr_pts,
        prev_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=ransac_max_iters,
        confidence=0.99,
    )
    if inliers is None:
        mat = compute_rigid_transform(curr_pts, prev_pts)
    else:
        mat = compute_rigid_transform(curr_pts, prev_pts, mask=inliers)
    if mat is None:
        return np.eye(2, 3, dtype=np.float32)

    angle = np.degrees(np.arctan2(mat[1, 0], mat[0, 0]))
    if abs(angle) <= rotation_zero_thresh_deg:
        if inliers is not None:
            mask = inliers.reshape(-1).astype(bool)
            src_in = curr_pts[mask]
            dst_in = prev_pts[mask]
        else:
            src_in = curr_pts
            dst_in = prev_pts
        if len(src_in) >= 2:
            trans = dst_in - src_in
            mean = np.mean(trans, axis=0)
            return np.array([[1.0, 0.0, float(mean[0])], [0.0, 1.0, float(mean[1])]], dtype=np.float32)
        return np.array([[1.0, 0.0, mat[0, 2]], [0.0, 1.0, mat[1, 2]]], dtype=np.float32)

    return mat


def               compute_rigid_transform(src: np.ndarray, dst: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Compute a 2x3 rigid (rotation + translation) transform that maps src -> dst.
    src, dst: Nx2 arrays. Optional mask filters correspondences.
    Returns None if computation is degenerate.
    """
    if mask is not None:
        mask = mask.reshape(-1).astype(bool)
        src = src[mask]
        dst = dst[mask]

    if len(src) < 2:
        return None

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    H = src_centered.T @ dst_centered / len(src)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean
    matrix = np.hstack([R, t.reshape(2, 1)])
    return matrix.astype(np.float32)


def angle_between_pairs(
    prev_pts: np.ndarray, curr_pts: np.ndarray, idx_a: int, idx_b: int
) -> Optional[float]:
    """
    Compute rotation angle from one matched pair of points.
    prev_pts/curr_pts: Nx2 arrays of matched keypoints.
    idx_a/idx_b: indices of the two matches to form a segment.
    Returns angle_prev - angle_curr (radians) or None if degenerate.
    """
    v_prev = prev_pts[idx_b] - prev_pts[idx_a]
    v_curr = curr_pts[idx_b] - curr_pts[idx_a]
    if np.linalg.norm(v_prev) < 1e-6 or np.linalg.norm(v_curr) < 1e-6:
        return None
    ang_prev = np.arctan2(v_prev[1], v_prev[0])
    ang_curr = np.arctan2(v_curr[1], v_curr[0])
    return ang_prev - ang_curr


def find_consistent_rotation(
    prev_pts: np.ndarray, curr_pts: np.ndarray, angle_tol_deg: float = ANGLE_TOL_DEG, dist_tol_rel: float = DIST_TOL_REL
) -> Optional[Tuple[float, Tuple[int, int, int, int]]]:
    """
    Find two disjoint matched pairs whose rotation angles agree within tolerance.
    Returns (angle, (i, j, k, l)) where angle is the agreed rotation and i/j/k/l are indices.
    """
    n = len(prev_pts)
    angle_tol = np.deg2rad(angle_tol_deg)

    # Precompute all pair angles and lengths.
    pairs = []
    max_pairs = min(n, 10)
    for i in range(max_pairs):
        for j in range(i + 1, max_pairs):
            ang = angle_between_pairs(prev_pts, curr_pts, i, j)
            if ang is None:
                continue
            len_prev = np.linalg.norm(prev_pts[j] - prev_pts[i])
            len_curr = np.linalg.norm(curr_pts[j] - curr_pts[i])
            pairs.append((i, j, ang, len_prev, len_curr))

    for idx_a in range(len(pairs)):
        i, j, ang1, len_p1, len_c1 = pairs[idx_a]
        for idx_b in range(idx_a + 1, len(pairs)):
            k, l, ang2, len_p2, len_c2 = pairs[idx_b]
            if len({i, j, k, l}) < 4:
                continue
            if len_p1 < 1e-6 or len_p2 < 1e-6:
                continue
            if not (abs(len_p1 - len_c1) / len_p1 <= dist_tol_rel and abs(len_p2 - len_c2) / len_p2 <= dist_tol_rel):
                continue
            diff = normalize_angle(ang1 - ang2)
            if abs(diff) <= angle_tol:
                return ang1, (i, j, k, l)

    return None


def compute_transform_from_pairs(prev_pts: np.ndarray, curr_pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute rotation from the angle difference of the segment defined by two matched pairs,
    then translation after rotating current points onto previous.
    """
    if len(prev_pts) < 2 or len(curr_pts) < 2:
        return None

    p0, p1 = prev_pts[:2]
    q0, q1 = curr_pts[:2]

    v_prev = p1 - p0
    v_curr = q1 - q0

    norm_prev = np.linalg.norm(v_prev)
    norm_curr = np.linalg.norm(v_curr)
    if norm_prev < 1e-6 or norm_curr < 1e-6:
        return None

    angle_prev = np.arctan2(v_prev[1], v_prev[0])
    angle_curr = np.arctan2(v_curr[1], v_curr[0])
    angle = angle_prev - angle_curr

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    t0 = p0 - R @ q0
    t1 = p1 - R @ q1
    t = (t0 + t1) / 2.0

    matrix = np.hstack([R, t.reshape(2, 1)])
    return matrix.astype(np.float32)


def compute_rigid_from_angle(
    angle: float, curr_pts: np.ndarray, prev_pts: np.ndarray
) -> Optional[np.ndarray]:
    """Compute a 2x3 rigid transform from a known rotation angle and point pairs."""
    if len(curr_pts) < 1 or len(prev_pts) < 1:
        return None
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    t = np.mean(prev_pts - (curr_pts @ R.T), axis=0)
    matrix = np.hstack([R, t.reshape(2, 1)])
    return matrix.astype(np.float32)


def estimate_pair_transform(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    min_matches: int = MIN_MATCHES_DEFAULT,
    max_matches: int = MAX_MATCHES_DEFAULT,
    ransac_thresh: float = RANSAC_THRESH_DEFAULT,
    ransac_max_iters: int = RANSAC_MAX_ITERS_DEFAULT,
    use_ratio_test: bool = True,
    ratio: float = 0.4,
    translation_only: bool = False,
    rotation_zero_thresh_deg: float = 2.0,
    sift: Optional[cv2.SIFT] = None,
    bf: Optional[cv2.BFMatcher] = None,
    choice_stats: Optional[Dict[str, int]] = None,
    candidate_mode: str = "auto",
    timing_stats: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Estimate a rigid transform (rotation + translation) mapping current -> previous frame.

    Returns a 2x3 matrix. If not enough matches are found, returns identity.
    """
    if sift is None:
        sift = cv2.SIFT_create()
    t0 = time.perf_counter()
    kp_prev, des_prev = sift.detectAndCompute(prev_gray, None)
    kp_curr, des_curr = sift.detectAndCompute(curr_gray, None)
    t1 = time.perf_counter()
    if timing_stats is not None:
        timing_stats["sift_seconds"] = timing_stats.get("sift_seconds", 0.0) + (t1 - t0)

    if des_prev is None or des_curr is None or len(kp_prev) < 2 or len(kp_curr) < 2:
        if choice_stats is not None:
            choice_stats["insufficient_matches"] = choice_stats.get("insufficient_matches", 0) + 1
        return np.eye(2, 3, dtype=np.float32)

    if use_ratio_test:
        if bf is None:
            bf = cv2.BFMatcher()
        knn = bf.knnMatch(des_curr, des_prev, k=2)
        matches = [m[0] for m in knn if len(m) == 2 and m[0].distance <= ratio * m[1].distance]
    else:
        if bf is None:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des_prev, des_curr)

    if len(matches) < max(min_matches, 4):
        if choice_stats is not None:
            choice_stats["insufficient_matches"] = choice_stats.get("insufficient_matches", 0) + 1
        return np.eye(2, 3, dtype=np.float32)

    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]  # cap matches to reduce work
    if use_ratio_test:
        curr_pts = np.float32([kp_curr[m.queryIdx].pt for m in matches])
        prev_pts = np.float32([kp_prev[m.trainIdx].pt for m in matches])
    else:
        prev_pts = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([kp_curr[m.trainIdx].pt for m in matches])

    # Skip rotation candidate work if we only care about translation.
    if translation_only:
        found = None
    else:
        found = find_consistent_rotation(prev_pts, curr_pts, angle_tol_deg=ANGLE_TOL_DEG, dist_tol_rel=DIST_TOL_REL)

    # Estimate rotation from two disjoint pairs; if not found, assume zero rotation.
    def median_error(mat: Optional[np.ndarray]) -> float:
        if mat is None:
            return float("inf")
        preds = (curr_pts @ mat[:, :2].T) + mat[:, 2]
        errs = np.linalg.norm(preds - prev_pts, axis=1)
        return float(np.median(errs))

    candidate_angle = None
    if found:
        angle, (i, j, k, l) = found
        prev_sel = np.stack([prev_pts[i], prev_pts[j], prev_pts[k], prev_pts[l]], axis=0)
        curr_sel = np.stack([curr_pts[i], curr_pts[j], curr_pts[k], curr_pts[l]], axis=0)
        candidate_angle = compute_rigid_transform(curr_sel, prev_sel)

    if candidate_mode == "angle_only" and not translation_only:
        if found:
            angle, _ = found
            best = compute_rigid_from_angle(angle, curr_pts, prev_pts)
        else:
            best = compute_transform_from_pairs(prev_pts, curr_pts)
        if best is None:
            if choice_stats is not None:
                choice_stats["fallback_identity"] = choice_stats.get("fallback_identity", 0) + 1
            return np.eye(2, 3, dtype=np.float32)
        if choice_stats is not None:
            choice_stats["angle_only_used"] = choice_stats.get("angle_only_used", 0) + 1
        if rotation_zero_thresh_deg > 0.0:
            angle = np.degrees(np.arctan2(best[1, 0], best[0, 0]))
            if abs(angle) <= rotation_zero_thresh_deg:
                trans = prev_pts - curr_pts
                mean = np.mean(trans, axis=0)
                return np.array([[1.0, 0.0, float(mean[0])], [0.0, 1.0, float(mean[1])]], dtype=np.float32)
        return best

    # Robust fit using RANSAC inliers, then rigid solve.
    _, inliers = cv2.estimateAffinePartial2D(
        curr_pts,
        prev_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=ransac_max_iters,
        confidence=0.99,
    )
    if translation_only:
        if choice_stats is not None:
            choice_stats["translation_only"] = choice_stats.get("translation_only", 0) + 1
        if inliers is None:
            if choice_stats is not None:
                choice_stats["translation_only_failed"] = choice_stats.get("translation_only_failed", 0) + 1
            return np.eye(2, 3, dtype=np.float32)
        mask = inliers.reshape(-1).astype(bool)
        src_in = curr_pts[mask]
        dst_in = prev_pts[mask]
        if len(src_in) < 2:
            if choice_stats is not None:
                choice_stats["translation_only_failed"] = choice_stats.get("translation_only_failed", 0) + 1
            return np.eye(2, 3, dtype=np.float32)
        trans = dst_in - src_in
        mean = np.mean(trans, axis=0)
        return np.array([[1.0, 0.0, float(mean[0])], [0.0, 1.0, float(mean[1])]], dtype=np.float32)

    candidate_ransac = compute_rigid_transform(curr_pts, prev_pts, mask=inliers)
    if candidate_ransac is None:
        candidate_ransac = compute_rigid_transform(curr_pts, prev_pts)
    if choice_stats is not None:
        if candidate_angle is not None:
            choice_stats["angle_available"] = choice_stats.get("angle_available", 0) + 1
        if candidate_ransac is not None:
            choice_stats["ransac_available"] = choice_stats.get("ransac_available", 0) + 1

    # Choose the candidate with lower median reprojection error.
    best = None
    err_angle = median_error(candidate_angle)
    err_ransac = median_error(candidate_ransac)
    if candidate_mode == "angle":
        best = candidate_angle if candidate_angle is not None else candidate_ransac
    elif candidate_mode == "ransac":
        best = candidate_ransac if candidate_ransac is not None else candidate_angle
    else:
        if err_angle <= err_ransac:
            best = candidate_angle
        else:
            best = candidate_ransac
    if choice_stats is not None and best is not None:
        if best is candidate_angle:
            choice_stats["angle_used"] = choice_stats.get("angle_used", 0) + 1
        elif best is candidate_ransac:
            choice_stats["ransac_used"] = choice_stats.get("ransac_used", 0) + 1

    if best is None:
        if choice_stats is not None:
            choice_stats["fallback_identity"] = choice_stats.get("fallback_identity", 0) + 1
        return np.eye(2, 3, dtype=np.float32)
    if rotation_zero_thresh_deg > 0.0:
        angle = np.degrees(np.arctan2(best[1, 0], best[0, 0]))
        if abs(angle) <= rotation_zero_thresh_deg:
            if inliers is not None:
                mask = inliers.reshape(-1).astype(bool)
                src_in = curr_pts[mask]
                dst_in = prev_pts[mask]
            else:
                src_in = curr_pts
                dst_in = prev_pts
            if len(src_in) >= 2:
                trans = dst_in - src_in
                mean = np.mean(trans, axis=0)
                return np.array([[1.0, 0.0, float(mean[0])], [0.0, 1.0, float(mean[1])]], dtype=np.float32)
            return np.array([[1.0, 0.0, best[0, 2]], [0.0, 1.0, best[1, 2]]], dtype=np.float32)
    return best


def is_identity(matrix: np.ndarray, tol: float = 1e-3) -> bool:
    """Check if a 2x3 transform is (approximately) identity."""
    if matrix.shape != (2, 3):
        return False
    identity = np.eye(2, 3, dtype=np.float32)
    return np.allclose(matrix, identity, atol=tol)


def transform_params(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Extract angle (rad), tx, ty from a 2x3 rigid transform."""
    angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    tx = float(matrix[0, 2])
    ty = float(matrix[1, 2])
    return angle, tx, ty


def median_recent(values: List[float], window: int) -> float:
    """Median of the most recent window values."""
    if window <= 1 or len(values) == 0:
        return float(values[-1]) if values else 0.0
    window = min(window, len(values))
    return float(np.median(values[-window:]))


def replace_outliers(
    matrix: np.ndarray,
    angles: List[float],
    txs: List[float],
    tys: List[float],
    window: int,
    angle_thresh_deg: float,
    trans_thresh: float,
) -> Tuple[np.ndarray, bool]:
    """
    Replace current transform parameters with rolling median if they deviate too much.
    Uses a causal window of the most recent values (including current).
    """
    angle, tx, ty = transform_params(matrix)
    angles.append(angle)
    txs.append(tx)
    tys.append(ty)

    angle_med = median_recent(angles, window)
    tx_med = median_recent(txs, window)
    ty_med = median_recent(tys, window)

    angle_thresh = np.deg2rad(angle_thresh_deg)
    replaced = False
    use_angle = angle
    use_tx = tx
    use_ty = ty

    if abs(angle - angle_med) > angle_thresh:
        use_angle = angle_med
        replaced = True
    if abs(tx - tx_med) > trans_thresh:
        use_tx = tx_med
        replaced = True
    if abs(ty - ty_med) > trans_thresh:
        use_ty = ty_med
        replaced = True

    if not replaced:
        return matrix, False

    cos_a = float(np.cos(use_angle))
    sin_a = float(np.sin(use_angle))
    replaced_mat = np.array([[cos_a, -sin_a, use_tx], [sin_a, cos_a, use_ty]], dtype=np.float32)
    return replaced_mat, True


def align_video(
    input_path: Path,
    output_path: Optional[Path],
    save_matrices: Optional[Path],
    max_frames: Optional[int] = None,
    min_matches: int = 2,
    max_matches: int = MAX_MATCHES_DEFAULT,
    ransac_thresh: float = RANSAC_THRESH_DEFAULT,
    ransac_max_iters: int = RANSAC_MAX_ITERS_DEFAULT,
    pairwise_smooth_radius: int = PAIRWISE_SMOOTH_RADIUS_DEFAULT,
    pairwise_angle_thresh_deg: float = PAIRWISE_ANGLE_THRESH_DEG_DEFAULT,
    pairwise_trans_thresh: float = PAIRWISE_TRANS_THRESH_DEFAULT,
    use_ratio_test: bool = True,
    ratio: float = 0.4,
    translation_only: bool = False,
    rotation_zero_thresh_deg: float = 2.0,
    candidate_mode: str = "auto",
    return_kept_indices: bool = False,
    progress_every: int = 0,
    load_matrices: Optional[Path] = None,
) -> List[np.ndarray]:
    """
    Align consecutive frames and optionally write an aligned video.

    If load_matrices is provided, reuse stored 2x3 transforms (shape: Nx2x3) instead
    of re-estimating. Still writes the aligned video if output_path is set.
    """
    print(f"[align] Opening input video: {input_path}")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read first frame from {input_path}")

    height, width = prev_frame.shape[:2]
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, cap.get(cv2.CAP_PROP_FPS) or 30.0, (width, height))
        print(f"[align] Writing aligned video to {output_path}")
    else:
        print("[align] No output video requested (estimation only).")

    transforms: List[np.ndarray] = []
    kept_indices: List[int] = []
    frame_idx = 0
    identity_count = 0
    estimated_count = 0
    replaced_count = 0
    angle_history: List[float] = []
    tx_history: List[float] = []
    ty_history: List[float] = []
    choice_stats: Dict[str, int] = {}
    timing_stats: Dict[str, float] = {}
    sift = None
    bf = None

    if load_matrices:
        loaded = np.load(str(load_matrices))
        if loaded.ndim != 3 or loaded.shape[1:] != (2, 3):
            raise ValueError(f"Loaded matrices must have shape (N,2,3); got {loaded.shape}")
        raw_transforms = [np.array(m, dtype=np.float32) for m in loaded]
        transforms = []
        for idx, matrix in enumerate(raw_transforms):
            if is_identity(matrix, tol=SKIP_IDENTITY_THRESH):
                continue
            transforms.append(matrix)
            kept_indices.append(idx + 1)
        if max_frames is not None:
            transforms = transforms[:max_frames]
        print(f"[align] Loaded {len(transforms)} pairwise transforms from {load_matrices}")

        if writer is not None:
            # Walk the video to warp frames using the loaded transforms.
            keep_set = set(kept_indices)
            keep_pos = 0
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                if frame_idx + 1 not in keep_set:
                    frame_idx += 1
                    continue
                if keep_pos >= len(transforms):
                    break
                matrix = transforms[keep_pos]
                keep_pos += 1
                aligned = cv2.warpAffine(curr_frame, matrix, (width, height))
                writer.write(aligned)
                frame_idx += 1
                if progress_every and frame_idx % progress_every == 0:
                    print(f"[align] (reuse) Wrote frame {frame_idx}")
    else:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher() if use_ratio_test else cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            matrix = estimate_pair_transform(
                prev_gray,
                curr_gray,
                min_matches=min_matches,
                max_matches=max_matches,
                ransac_thresh=ransac_thresh,
                ransac_max_iters=ransac_max_iters,
                use_ratio_test=use_ratio_test,
                ratio=ratio,
                translation_only=translation_only,
                rotation_zero_thresh_deg=rotation_zero_thresh_deg,
                sift=sift,
                bf=bf,
                choice_stats=choice_stats,
                candidate_mode=candidate_mode,
                timing_stats=timing_stats,
            )
            if pairwise_smooth_radius > 0:
                window = 2 * pairwise_smooth_radius + 1
                matrix, replaced = replace_outliers(
                    matrix,
                    angle_history,
                    tx_history,
                    ty_history,
                    window=window,
                    angle_thresh_deg=pairwise_angle_thresh_deg,
                    trans_thresh=pairwise_trans_thresh,
                )
                if replaced:
                    replaced_count += 1
            if is_identity(matrix, tol=SKIP_IDENTITY_THRESH):
                frame_idx += 1
                if progress_every and frame_idx % progress_every == 0:
                    print(f"[align] Skipped frame {frame_idx} (near-identity)")
                if max_frames is not None and frame_idx >= max_frames:
                    break
                continue
            transforms.append(matrix)
            estimated_count += 1
            kept_indices.append(frame_idx + 1)
            if is_identity(matrix):
                identity_count += 1

            if writer is not None:
                aligned = cv2.warpAffine(curr_frame, matrix, (width, height))
                writer.write(aligned)

            prev_gray = curr_gray
            frame_idx += 1

            if progress_every and frame_idx % progress_every == 0:
                print(f"[align] Processed frame {frame_idx}")

            if max_frames is not None and frame_idx >= max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()

    if save_matrices:
        np.save(str(save_matrices), np.stack(transforms, axis=0))
        print(f"[align] Saved pairwise matrices to {save_matrices}")

    print(f"[align] Completed {frame_idx} frame pairs.")
    if estimated_count > 0:
        print(f"[align] Identity transforms: {identity_count}/{estimated_count} "
              f"({(100.0 * identity_count / estimated_count):.1f}%)")
        if pairwise_smooth_radius > 0:
            print(f"[align] Replaced outliers: {replaced_count}/{estimated_count} "
                  f"({(100.0 * replaced_count / estimated_count):.1f}%)")
    if choice_stats:
        angle_used = choice_stats.get("angle_used", 0)
        ransac_used = choice_stats.get("ransac_used", 0)
        total_used = angle_used + ransac_used
        if total_used > 0:
            print(f"[align] Candidate choice: angle {angle_used}/{total_used} "
                  f"({(100.0 * angle_used / total_used):.1f}%), "
                  f"ransac {ransac_used}/{total_used} "
                  f"({(100.0 * ransac_used / total_used):.1f}%)")
        if "insufficient_matches" in choice_stats:
            print(f"[align] Insufficient matches (identity): {choice_stats['insufficient_matches']}")
        if "translation_only_failed" in choice_stats:
            print(f"[align] Translation-only failures (identity): {choice_stats['translation_only_failed']}")
    if timing_stats.get("sift_seconds", 0.0) > 0.0:
        print(f"[align] Total SIFT detect+compute time: {timing_stats['sift_seconds']:.2f}s")

    if return_kept_indices:
        return transforms, kept_indices
    return transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align consecutive frames of a video using two matched SIFT pairs (rotation+translation only).")
    parser.add_argument("--input", required=True, type=Path, help="Path to the input video.")
    parser.add_argument("--output", type=Path, help="Optional path to write the aligned video.")
    parser.add_argument("--save-matrices", type=Path, help="Optional path to save the 2x3 affine matrices (NumPy .npy).")
    parser.add_argument("--load-matrices", type=Path, help="Optional path to load precomputed 2x3 matrices and skip estimation.")
    parser.add_argument("--max-frames", type=int, help="Process only the first N frame pairs (for quick tests).")
    parser.add_argument("--min-matches", type=int, default=MIN_MATCHES_DEFAULT, help="Minimum matches required to estimate a transform (min 2).")
    parser.add_argument("--max-matches", type=int, default=MAX_MATCHES_DEFAULT, help="Maximum matches used for estimation.")
    parser.add_argument("--ransac-thresh", type=float, default=RANSAC_THRESH_DEFAULT, help="RANSAC reprojection threshold (pixels).")
    parser.add_argument("--ransac-max-iters", type=int, default=RANSAC_MAX_ITERS_DEFAULT, help="RANSAC maximum iterations.")
    parser.add_argument("--pairwise-smooth-radius", type=int, default=PAIRWISE_SMOOTH_RADIUS_DEFAULT, help="Rolling median radius for pairwise transforms (0 disables).")
    parser.add_argument("--pairwise-angle-thresh-deg", type=float, default=PAIRWISE_ANGLE_THRESH_DEG_DEFAULT, help="Angle outlier threshold in degrees.")
    parser.add_argument("--pairwise-trans-thresh", type=float, default=PAIRWISE_TRANS_THRESH_DEFAULT, help="Translation outlier threshold in pixels.")
    parser.add_argument("--no-ratio-test", action="store_true", help="Disable KNN + Lowe ratio test (use cross-check).")
    parser.add_argument("--ratio", type=float, default=0.4, help="Lowe ratio threshold when using ratio test.")
    parser.add_argument("--translation-only", action="store_true", help="Estimate translation only from RANSAC inliers.")
    parser.add_argument("--rotation-zero-thresh-deg", type=float, default=2.0, help="Clamp small rotations to zero (degrees).")
    parser.add_argument(
        "--candidate-mode",
        choices=["auto", "angle", "ransac", "angle_only"],
        default="auto",
        help="Which candidate to use when estimating the rigid transform.",
    )
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N frames (0 to disable).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transforms = align_video(
        input_path=args.input,
        output_path=args.output,
        save_matrices=args.save_matrices,
        max_frames=args.max_frames,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        ransac_thresh=args.ransac_thresh,
        ransac_max_iters=args.ransac_max_iters,
        pairwise_smooth_radius=args.pairwise_smooth_radius,
        pairwise_angle_thresh_deg=args.pairwise_angle_thresh_deg,
        pairwise_trans_thresh=args.pairwise_trans_thresh,
        use_ratio_test=not args.no_ratio_test,
        ratio=args.ratio,
        translation_only=args.translation_only,
        rotation_zero_thresh_deg=args.rotation_zero_thresh_deg,
        candidate_mode=args.candidate_mode,
        progress_every=args.progress_every,
        load_matrices=args.load_matrices,
    )
    print(f"Computed {len(transforms)} pairwise transforms.")
    if args.save_matrices:
        print(f"Saved matrices to {args.save_matrices}")
    if args.output:
        print(f"Aligned video written to {args.output}")


if __name__ == "__main__":
    main()
