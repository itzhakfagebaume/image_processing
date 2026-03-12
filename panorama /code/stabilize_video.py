"""
Stabilize a video by removing vertical motion and rotation (optional horizontal smoothing).

Steps:
1) Estimate pairwise rigid transforms (current -> previous) using align_video.align_video.
2) Compose them into cumulative transforms mapping each frame back to the first frame.
3) Build stabilized transforms by keeping rotation + vertical correction while optionally
   smoothing horizontal translation.
4) Warp frames using the stabilized transforms and write the stabilized video.

Notes:
- --save-matrices / load_matrices refer to pairwise transforms (same API as align_video).
- --save-stabilized saves the final per-frame stabilization transforms.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from align_video import align_video, is_identity, SKIP_IDENTITY_THRESH


def to_h(mat_2x3: np.ndarray) -> np.ndarray:
    """Convert 2x3 affine to 3x3 homogeneous."""
    return np.vstack([mat_2x3, [0.0, 0.0, 1.0]]).astype(np.float32)


def compose_cumulative(transforms: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    Given pairwise transforms T_i (frame i -> frame i-1), compute cumulative transforms
    C_i that map frame i directly to frame 0.
    C_0 = I, C_i = C_{i-1} @ T_i.
    """
    cumulative: List[np.ndarray] = [np.eye(3, dtype=np.float32)]
    for T in transforms:
        cumulative.append(cumulative[-1] @ to_h(T))
    return cumulative


def extract_trajectory(cumulative: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract rotation angles (radians) and translations (tx, ty) from 3x3 transforms."""
    angles = []
    tx = []
    ty = []
    for M in cumulative:
        angles.append(np.arctan2(M[1, 0], M[0, 0]))
        tx.append(M[0, 2])
        ty.append(M[1, 2])
    angles = np.unwrap(np.array(angles, dtype=np.float32))
    tx = np.array(tx, dtype=np.float32)
    ty = np.array(ty, dtype=np.float32)
    return angles, tx, ty


def moving_average(signal: np.ndarray, radius: int) -> np.ndarray:
    """Simple edge-padded moving average."""
    if radius <= 0:
        return signal.copy()
    kernel = np.ones(2 * radius + 1, dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(signal, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def rigid_matrix(angle: float, tx: float, ty: float) -> np.ndarray:
    """Create a 2x3 rigid transform (rotation about origin + translation)."""
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    return np.array([[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=np.float32)


def pairwise_params(transforms: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract angle (rad), tx, ty from 2x3 pairwise transforms."""
    angles = []
    tx = []
    ty = []
    for M in transforms:
        angles.append(np.arctan2(M[1, 0], M[0, 0]))
        tx.append(M[0, 2])
        ty.append(M[1, 2])
    angles = np.unwrap(np.array(angles, dtype=np.float32))
    tx = np.array(tx, dtype=np.float32)
    ty = np.array(ty, dtype=np.float32)
    return angles, tx, ty


def median_filter_1d(values: np.ndarray, radius: int) -> np.ndarray:
    """Simple median filter with edge padding."""
    if radius <= 0:
        return values.copy()
    padded = np.pad(values, (radius, radius), mode="edge")
    out = np.empty_like(values)
    for i in range(len(values)):
        window = padded[i : i + 2 * radius + 1]
        out[i] = float(np.median(window))
    return out


def smooth_pairwise_transforms(
    transforms: Sequence[np.ndarray],
    radius: int,
    angle_thresh_deg: float,
    trans_thresh: float,
) -> List[np.ndarray]:
    """
    Replace outlier pairwise transforms by a local median estimate.
    This reduces accumulation of biased/noisy transforms in real videos.
    """
    if radius <= 0:
        return [np.array(m, dtype=np.float32) for m in transforms]

    angles, tx, ty = pairwise_params(transforms)
    angles_med = median_filter_1d(angles, radius)
    tx_med = median_filter_1d(tx, radius)
    ty_med = median_filter_1d(ty, radius)

    angle_thresh = np.deg2rad(angle_thresh_deg)
    out = []
    for a, x, y, a_m, x_m, y_m in zip(angles, tx, ty, angles_med, tx_med, ty_med):
        use_a = a_m if abs(a - a_m) > angle_thresh else a
        use_x = x_m if abs(x - x_m) > trans_thresh else x
        use_y = y_m if abs(y - y_m) > trans_thresh else y
        out.append(rigid_matrix(float(use_a), float(use_x), float(use_y)))
    return out


def build_stabilized_transforms(
    cumulative: Sequence[np.ndarray],
    smoothing_radius: int,
    smooth_x: bool,
    width: int,
    height: int,
) -> List[np.ndarray]:
    """
    Build per-frame transforms that remove rotation + vertical motion.
    If smooth_x is True, smooth horizontal correction; otherwise keep horizontal motion.
    """
    tx = horizontal_trajectory(cumulative, smoothing_radius, smooth_x, width, height)

    stabilized: List[np.ndarray] = []
    for C, x in zip(cumulative, tx):
        shift = np.array([[1.0, 0.0, float(x)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        M = (shift @ C).astype(np.float32)
        stabilized.append(M[:2, :])
    return stabilized


def horizontal_trajectory(
    cumulative: Sequence[np.ndarray],
    smoothing_radius: int,
    smooth_x: bool,
    width: int,
    height: int,
) -> np.ndarray:
    """Compute horizontal trajectory of the frame center from cumulative transforms."""
    center = np.array([width / 2.0, height / 2.0, 1.0], dtype=np.float32)
    forward_dx = []
    for C in cumulative:
        A = np.linalg.inv(C)
        p = A @ center
        forward_dx.append(float(p[0] - center[0]))
    tx = np.array(forward_dx, dtype=np.float32)
    if smooth_x:
        tx = moving_average(tx, smoothing_radius)
    tx = tx - tx[0]
    return tx


def compute_crop_bounds(
    input_path: Path,
    transforms: Sequence[np.ndarray],
    width: int,
    height: int,
    threshold: int,
    max_frames: Optional[int],
    frame_indices: Optional[Sequence[int]] = None,
) -> tuple[int, int, int, int]:
    """Compute a global crop box of non-black pixels across all warped frames."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {input_path}")

    min_x, min_y = width - 1, height - 1
    max_x, max_y = 0, 0
    any_valid = False

    frame_idx = 0
    keep_set = set(frame_indices) if frame_indices is not None else None
    keep_pos = 0
    while True:
        frame = first_frame if frame_idx == 0 else cap.read()[1]
        if frame is None:
            break
        if keep_set is None and frame_idx >= len(transforms):
            break
        if keep_set is not None:
            if frame_idx not in keep_set:
                frame_idx += 1
                continue
            if keep_pos >= len(transforms):
                break
            M = transforms[keep_pos]
            keep_pos += 1
        else:
            M = transforms[frame_idx]
        if max_frames is not None and frame_idx >= max_frames:
            break
        warped = cv2.warpAffine(
            frame,
            M,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = gray > threshold
        if mask.any():
            ys, xs = np.where(mask)
            min_x = min(min_x, int(xs.min()))
            max_x = max(max_x, int(xs.max()))
            min_y = min(min_y, int(ys.min()))
            max_y = max(max_y, int(ys.max()))
            any_valid = True
        frame_idx += 1

    cap.release()
    if not any_valid:
        return 0, 0, width - 1, height - 1
    return min_x, min_y, max_x, max_y


def stabilize_video(
    input_path: Path,
    output_path: Path,
    smoothing_radius: int = 5,
    max_frames: Optional[int] = None,
    min_matches: int = 2,
    progress_every: int = 50,
    save_matrices: Optional[Path] = None,
    smooth_x: bool = False,
    load_matrices: Optional[Path] = None,
    save_stabilized: Optional[Path] = None,
    pairwise_smooth_radius: int = 0,
    pairwise_angle_thresh_deg: float = 2.0,
    pairwise_trans_thresh: float = 4.0,
    rotation_zero_thresh_deg: float = 2.0,
    post_crop_black: bool = True,
    post_crop_threshold: int = 40,
    post_crop_margin: int = 0,
    pairwise_transforms: Optional[Sequence[np.ndarray]] = None,
    candidate_mode: str = "auto",
) -> None:
    transforms: List[np.ndarray]
    kept_indices: Optional[List[int]] = None
    if pairwise_transforms is not None:
        transforms = []
        kept_indices = []
        for idx, matrix in enumerate(pairwise_transforms):
            matrix = np.array(matrix, dtype=np.float32)
            if is_identity(matrix, tol=SKIP_IDENTITY_THRESH):
                continue
            transforms.append(matrix)
            kept_indices.append(idx + 1)
        if max_frames is not None:
            transforms = transforms[:max_frames]
        if save_matrices:
            np.save(str(save_matrices), np.stack(transforms, axis=0))
            print(f"[stabilize] Saved pairwise matrices to {save_matrices}")
        print(f"[stabilize] Using {len(transforms)} provided pairwise transforms.")
    elif load_matrices is not None:
        transforms_arr = np.load(str(load_matrices))
        if transforms_arr.ndim != 3 or transforms_arr.shape[1:] != (2, 3):
            raise ValueError(f"Loaded matrices must have shape (N,2,3); got {transforms_arr.shape}")
        transforms = []
        kept_indices = []
        for idx, matrix in enumerate(transforms_arr):
            matrix = np.array(matrix, dtype=np.float32)
            if is_identity(matrix, tol=SKIP_IDENTITY_THRESH):
                continue
            transforms.append(matrix)
            kept_indices.append(idx + 1)
        if max_frames is not None:
            transforms = transforms[:max_frames]
        print(f"[stabilize] Loaded {len(transforms)} pairwise transforms from {load_matrices}")
    else:
        print(f"[stabilize] Estimating pairwise transforms for {input_path}...")
        result = align_video(
            input_path=input_path,
            output_path=None,
            save_matrices=save_matrices,
            max_frames=max_frames,
            min_matches=min_matches,
            progress_every=progress_every,
            load_matrices=None,
            rotation_zero_thresh_deg=rotation_zero_thresh_deg,
            return_kept_indices=True,
            candidate_mode=candidate_mode,
        )
        transforms, kept_indices = result
        print(f"[stabilize] Got {len(transforms)} pairwise transforms.")

    if pairwise_smooth_radius > 0:
        print(
            f"[stabilize] Smoothing pairwise transforms (radius={pairwise_smooth_radius}, "
            f"angle_thresh={pairwise_angle_thresh_deg}deg, trans_thresh={pairwise_trans_thresh}px)..."
        )
        transforms = smooth_pairwise_transforms(
            transforms,
            radius=pairwise_smooth_radius,
            angle_thresh_deg=pairwise_angle_thresh_deg,
            trans_thresh=pairwise_trans_thresh,
        )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read first frame from {input_path}")

    height, width = first_frame.shape[:2]
    cumulative = compose_cumulative(transforms)
    stabilized = build_stabilized_transforms(
        cumulative,
        smoothing_radius,
        smooth_x,
        width=width,
        height=height,
    )
    print(f"[stabilize] Built {len(stabilized)} stabilized transforms.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    crop_box = None
    if post_crop_black:
        frame_indices = None
        if kept_indices is not None:
            frame_indices = [0] + kept_indices
        crop_box = compute_crop_bounds(
            input_path,
            stabilized,
            width,
            height,
            threshold=post_crop_threshold,
            max_frames=max_frames,
            frame_indices=frame_indices,
        )
        x0, y0, x1, y1 = crop_box
        x0 = max(x0 - post_crop_margin, 0)
        y0 = max(y0 - post_crop_margin, 0)
        x1 = min(x1 + post_crop_margin, width - 1)
        y1 = min(y1 + post_crop_margin, height - 1)
        crop_box = (x0, y0, x1, y1)
        crop_w = x1 - x0 + 1
        crop_h = y1 - y0 + 1
        print(f"[stabilize] Cropping to box x[{x0}:{x1}] y[{y0}:{y1}]")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_w, crop_h))
    else:
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {output_path}")

    print(f"[stabilize] Writing stabilized video to {output_path} at {fps:.2f} fps...")
    if kept_indices is not None:
        frame_indices = [0] + kept_indices
        keep_set = set(frame_indices)
        keep_pos = 0
    else:
        frame_indices = None
        keep_set = None
        keep_pos = 0

    frame_idx = 0
    while True:
        frame = first_frame if frame_idx == 0 else cap.read()[1]
        if frame is None:
            break
        if frame_indices is None and frame_idx >= len(stabilized):
            break
        if keep_set is not None:
            if frame_idx not in keep_set:
                frame_idx += 1
                continue
            if keep_pos >= len(stabilized):
                break
            M = stabilized[keep_pos]
            keep_pos += 1
        else:
            M = stabilized[frame_idx]
        warped = cv2.warpAffine(
            frame,
            M,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT if post_crop_black else cv2.BORDER_REPLICATE,
            borderValue=(0, 0, 0),
        )
        if crop_box is not None:
            x0, y0, x1, y1 = crop_box
            warped = warped[y0 : y1 + 1, x0 : x1 + 1]
        writer.write(warped)

        if progress_every and frame_idx % progress_every == 0:
            total = (len(stabilized) - 1) if frame_indices is None else (len(frame_indices) - 1)
            print(f"[stabilize] Wrote frame {frame_idx}/{total}")
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()

    if save_stabilized:
        np.save(str(save_stabilized), np.stack(stabilized, axis=0))
        print(f"[stabilize] Saved stabilized matrices to {save_stabilized}")
    print("[stabilize] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stabilize a video by removing vertical motion and rotation.")
    parser.add_argument("--input", required=True, type=Path, help="Path to input video.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write stabilized video.")
    parser.add_argument("--save-matrices", type=Path, help="Optional path to save pairwise 2x3 transforms (.npy).")
    parser.add_argument("--load-matrices", type=Path, help="Optional path to load precomputed pairwise 2x3 transforms.")
    parser.add_argument("--save-stabilized", type=Path, help="Optional path to save stabilized 2x3 transforms (.npy).")
    parser.add_argument("--no-post-crop-black", action="store_true", help="Disable black-border crop after stabilization.")
    parser.add_argument("--post-crop-threshold", type=int, default=40, help="Threshold for post-crop (0-255).")
    parser.add_argument("--post-crop-margin", type=int, default=0, help="Extra crop margin in pixels.")
    parser.add_argument("--max-frames", type=int, help="Limit to first N frames.")
    parser.add_argument("--min-matches", type=int, default=2, help="Minimum matches for transform estimation.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N frames (0 to disable).")
    parser.add_argument("--smoothing-radius", type=int, default=5, help="Moving average radius for trajectory smoothing.")
    parser.add_argument("--smooth-x", action="store_true", help="Also smooth horizontal translation.")
    parser.add_argument("--pairwise-smooth-radius", type=int, default=0, help="Median filter radius for pairwise transforms (0 disables).")
    parser.add_argument("--pairwise-angle-thresh-deg", type=float, default=2.0, help="Angle outlier threshold in degrees.")
    parser.add_argument("--pairwise-trans-thresh", type=float, default=4.0, help="Translation outlier threshold in pixels.")
    parser.add_argument("--rotation-zero-thresh-deg", type=float, default=2.0, help="Clamp small rotations to zero (degrees).")
    parser.add_argument(
        "--candidate-mode",
        choices=["auto", "angle", "ransac", "angle_only"],
        default="auto",
        help="Which candidate to use when estimating the rigid transform.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stabilize_video(
        input_path=args.input,
        output_path=args.output,
        smoothing_radius=args.smoothing_radius,
        max_frames=args.max_frames,
        min_matches=args.min_matches,
        progress_every=args.progress_every,
        save_matrices=args.save_matrices,
        smooth_x=args.smooth_x,
        load_matrices=args.load_matrices,
        save_stabilized=args.save_stabilized,
        pairwise_smooth_radius=args.pairwise_smooth_radius,
        pairwise_angle_thresh_deg=args.pairwise_angle_thresh_deg,
        pairwise_trans_thresh=args.pairwise_trans_thresh,
        rotation_zero_thresh_deg=args.rotation_zero_thresh_deg,
        post_crop_black=not args.no_post_crop_black,
        post_crop_threshold=args.post_crop_threshold,
        post_crop_margin=args.post_crop_margin,
        candidate_mode=args.candidate_mode,
    )


if __name__ == "__main__":
    main()
