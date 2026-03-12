"""
Create a panorama from a stabilized video by stitching center strips whose width
is based on the per-frame translation length (assume zero rotation).

Workflow:
1) Estimate or load pairwise translation-only transforms (frame i -> frame i-1).
2) Smooth pairwise translations and compose cumulative positions.
3) Compute a global anchor transform from all frame corners.
4) Warp each frame using the anchor and paste a center strip per frame.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from align_video import align_video

def overwrite_strip(
    mosaic: np.ndarray,
    weight: np.ndarray,
    warped: np.ndarray,
    warped_mask: np.ndarray,
    strip_center_x: float,
    strip_width: int,
) -> None:
    """Paste a vertical strip without blending (later frames overwrite)."""
    h, w = warped_mask.shape
    half = strip_width // 2
    # Use floor/ceil to keep adjacent strips overlapping and avoid 1px gaps.
    x0 = max(int(math.floor(strip_center_x - half)), 0)
    x1 = min(int(math.ceil(strip_center_x + half)), w)
    if x0 >= x1:
        return
    strip_mask = np.zeros_like(warped_mask, dtype=np.uint8)
    strip_mask[:, x0:x1] = warped_mask[:, x0:x1]
    mask_bool = strip_mask > 0
    mosaic[mask_bool] = warped[mask_bool]
    weight[mask_bool] = 1.0


def load_or_estimate_pairwise(
    motion_path: Path,
    load_matrices: Optional[Path],
    save_matrices: Optional[Path],
    min_matches: int,
    max_frames: Optional[int],
    progress_every: int,
    translation_only: bool,
    candidate_mode: str = "auto",
) -> List[np.ndarray]:
    if load_matrices is not None:
        loaded = np.load(str(load_matrices))
        if loaded.ndim != 3 or loaded.shape[1:] != (2, 3):
            raise ValueError(f"Loaded matrices must have shape (N,2,3); got {loaded.shape}")
        transforms = [np.array(m, dtype=np.float32) for m in loaded]
        if max_frames is not None:
            transforms = transforms[:max_frames]
        print(f"[panorama] Loaded {len(transforms)} pairwise transforms from {load_matrices}")
        return transforms

    print(f"[panorama] Estimating pairwise transforms for {motion_path}...")
    transforms = align_video(
        input_path=motion_path,
        output_path=None,
        save_matrices=save_matrices,
        max_frames=max_frames,
        min_matches=min_matches,
        progress_every=progress_every,
        load_matrices=None,
        translation_only=translation_only,
        rotation_zero_thresh_deg=0.0,
        candidate_mode=candidate_mode,
    )
    print(f"[panorama] Estimated {len(transforms)} pairwise transforms.")
    return transforms


def moving_average_1d(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.copy()
    kernel = np.ones(2 * radius + 1, dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def compute_crop_box(image: np.ndarray, threshold: int, margin: int) -> tuple[int, int, int, int]:
    """Compute a crop box for non-black pixels in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if not mask.any():
        h, w = image.shape[:2]
        return 0, 0, w - 1, h - 1
    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin, image.shape[1] - 1)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin, image.shape[0] - 1)
    return x0, y0, x1, y1


def crop_frame(frame: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    return frame[y0 : y1 + 1, x0 : x1 + 1]


def warp_with_mask_perspective(frame: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Warp a frame and mask using a 3x3 homography."""
    warped = cv2.warpPerspective(
        frame,
        M,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        mask,
        M,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, warped_mask


def feather_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    """Optionally blur a binary mask for softer seams."""
    if feather <= 0:
        return mask
    kernel = 2 * feather + 1
    return cv2.GaussianBlur(mask, (kernel, kernel), 0)


def pyramid_blend_roi(
    dst: np.ndarray,
    src: np.ndarray,
    mask: np.ndarray,
    x0: int,
    x1: int,
    levels: int = 3,
) -> None:
    """
    Multi-band blend src into dst within [x0:x1) using a float mask [0,1].
    Operates only on the ROI to keep it lightweight.
    """
    if x1 <= x0:
        return
    dst_roi = dst[:, x0:x1].astype(np.float32)
    src_roi = src[:, x0:x1].astype(np.float32)
    mask_roi = mask[:, x0:x1].astype(np.float32)
    # Ensure mask has 3 channels.
    if mask_roi.ndim == 2:
        mask_roi = mask_roi[..., None]
    if mask_roi.shape[2] == 1:
        mask_roi = np.repeat(mask_roi, 3, axis=2)

    # Cap pyramid levels based on smallest dimension.
    max_levels = 0
    h, w = src_roi.shape[:2]
    while min(h, w) >= 2 and max_levels < levels:
        h //= 2
        w //= 2
        max_levels += 1
    levels = max_levels

    gp_src = [src_roi]
    gp_dst = [dst_roi]
    gp_mask = [mask_roi]
    for _ in range(levels):
        gp_src.append(cv2.pyrDown(gp_src[-1]))
        gp_dst.append(cv2.pyrDown(gp_dst[-1]))
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

    lp_src = [gp_src[-1]]
    lp_dst = [gp_dst[-1]]
    for i in range(levels - 1, -1, -1):
        size = (gp_src[i].shape[1], gp_src[i].shape[0])
        src_exp = cv2.pyrUp(gp_src[i + 1], dstsize=size)
        dst_exp = cv2.pyrUp(gp_dst[i + 1], dstsize=size)
        lp_src.append(gp_src[i] - src_exp)
        lp_dst.append(gp_dst[i] - dst_exp)

    blended = lp_dst[0] * (1.0 - gp_mask[-1]) + lp_src[0] * gp_mask[-1]
    for i in range(1, len(lp_src)):
        size = (lp_src[i].shape[1], lp_src[i].shape[0])
        blended = cv2.pyrUp(blended, dstsize=size)
        mask_level = gp_mask[-(i + 1)]
        blended = blended + lp_dst[i] * (1.0 - mask_level) + lp_src[i] * mask_level
    dst[:, x0:x1] = blended.clip(0, 255)


def find_graphcut_seam(
    prev_patch: np.ndarray, prev_mask: np.ndarray, curr_patch: np.ndarray, curr_mask: np.ndarray
) -> Optional[np.ndarray]:
    """
    Run a graph-cut seam finder on two overlapping patches.
    Returns a mask for the current patch or None if unavailable.
    """
    if prev_patch.size == 0 or curr_patch.size == 0:
        return None
    if prev_patch.shape[:2] != curr_patch.shape[:2] or prev_mask.shape != curr_mask.shape:
        return None
    if (prev_mask > 0).sum() == 0 or (curr_mask > 0).sum() == 0:
        return None
    if (cv2.bitwise_and(prev_mask, curr_mask) > 0).sum() == 0:
        return None
    if not hasattr(cv2, "detail_GraphCutSeamFinder"):
        return None
    seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    images = [prev_patch.astype(np.uint8), curr_patch.astype(np.uint8)]
    masks = [prev_mask.copy(), curr_mask.copy()]
    corners = [(0, 0), (0, 0)]
    try:
        seam_finder.find(images, corners, masks)
    except cv2.error:
        return None
    return masks[1]


def blend_with_mask(
    mosaic: np.ndarray,
    weight: np.ndarray,
    warped: np.ndarray,
    mask: np.ndarray,
    feather: int,
    x0: int,
    x1: int,
    blend_mode: str,
    pyramid_levels: int,
) -> None:
    """
    Blend warped into mosaic using a binary mask limited to [x0:x1).
    Updates weight to mark covered pixels.
    """
    if x1 <= x0:
        return
    mask_roi = mask[:, x0:x1]
    if mask_roi.max() == 0:
        return
    mask_roi = feather_mask(mask_roi, feather)
    alpha = mask_roi.astype(np.float32) / 255.0
    use_pyr = blend_mode == "pyramid" and (x1 - x0) >= 8 and pyramid_levels > 0
    if use_pyr:
        alpha3 = alpha if alpha.ndim == 3 else alpha[..., None]
        try:
            pyramid_blend_roi(mosaic, warped, alpha3, x0, x1, levels=pyramid_levels)
        except (cv2.error, ValueError):
            use_pyr = False
    if not use_pyr:
        mosaic_roi = mosaic[:, x0:x1]
        warped_roi = warped[:, x0:x1]
        mosaic[:, x0:x1] = mosaic_roi * (1.0 - alpha[..., None]) + warped_roi * alpha[..., None]
    alpha_gray = alpha[..., 0] if alpha.ndim == 3 else alpha
    weight[:, x0:x1][alpha_gray > 1e-3] = 1.0


def graphcut_strip(
    mosaic: np.ndarray,
    weight: np.ndarray,
    warped: np.ndarray,
    warped_mask: np.ndarray,
    center_x: float,
    strip_width: int,
    seam_expand_ratio: float,
    seam_feather: int,
    blend_mode: str,
    pyramid_levels: int,
) -> None:
    """
    Blend a strip using graph-cut seam finding over an expanded overlap region.
    Falls back to a rectangular strip if graph-cut is unavailable.
    """
    _, w = warped_mask.shape
    extra = max(int(round(strip_width * seam_expand_ratio)), 0)
    half = strip_width // 2 + extra
    # Use floor/ceil to keep adjacent strips overlapping and avoid 1px gaps.
    x0 = max(int(math.floor(center_x - half)), 0)
    x1 = min(int(math.ceil(center_x + half)), w)
    if x1 - x0 < 2:
        return

    prev_mask_roi = (weight[:, x0:x1] > 0).astype(np.uint8) * 255
    curr_mask_roi = warped_mask[:, x0:x1].copy()
    if curr_mask_roi.max() == 0 or prev_mask_roi.shape != curr_mask_roi.shape:
        return

    # If nothing has been written yet, just paste the strip.
    if prev_mask_roi.max() == 0:
        full_mask = np.zeros_like(warped_mask, dtype=np.uint8)
        full_mask[:, x0:x1] = curr_mask_roi
        blend_with_mask(
            mosaic,
            weight,
            warped,
            full_mask,
            seam_feather,
            x0,
            x1,
            blend_mode=blend_mode,
            pyramid_levels=pyramid_levels,
        )
        return

    prev_patch = np.clip(mosaic[:, x0:x1], 0, 255).astype(np.uint8)
    curr_patch = np.clip(warped[:, x0:x1], 0, 255).astype(np.uint8)
    seam_mask_roi = find_graphcut_seam(prev_patch, prev_mask_roi, curr_patch, curr_mask_roi)
    if seam_mask_roi is None:
        # Graph-cut not available; fall back to the rectangular strip.
        full_mask = np.zeros_like(warped_mask, dtype=np.uint8)
        full_mask[:, x0:x1] = curr_mask_roi
        blend_with_mask(
            mosaic,
            weight,
            warped,
            full_mask,
            seam_feather,
            x0,
            x1,
            blend_mode=blend_mode,
            pyramid_levels=pyramid_levels,
        )
        return

    # Keep seam inside the valid warped region.
    seam_mask_roi = cv2.bitwise_and(seam_mask_roi, curr_mask_roi)
    full_mask = np.zeros_like(warped_mask, dtype=np.uint8)
    full_mask[:, x0:x1] = seam_mask_roi
    blend_with_mask(
        mosaic,
        weight,
        warped,
        full_mask,
        seam_feather,
        x0,
        x1,
        blend_mode=blend_mode,
        pyramid_levels=pyramid_levels,
    )


def affine_to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Convert a 2x3 affine matrix to 3x3 homogeneous form."""
    out = np.eye(3, dtype=np.float32)
    out[:2, :3] = matrix
    return out


def ensure_homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Ensure a matrix is 3x3 homogeneous."""
    if matrix.shape == (3, 3):
        return matrix.astype(np.float32)
    return affine_to_homogeneous(matrix)


def compose_cumulative_affine(pairwise_transforms: List[np.ndarray]) -> List[np.ndarray]:
    """Compose frame->reference transforms from pairwise frame->previous transforms."""
    cumulative = [np.eye(3, dtype=np.float32)]
    for transform in pairwise_transforms:
        current = cumulative[-1] @ affine_to_homogeneous(transform)
        cumulative.append(current)
    return cumulative


def get_global_geometry(
    frames: List[np.ndarray], transforms: List[np.ndarray]
) -> tuple[np.ndarray, tuple[int, int]]:
    """Compute anchor transform and canvas size from all frame corners."""
    if not frames:
        raise ValueError("Frames list is empty; cannot compute global geometry.")
    if not transforms:
        raise ValueError("Transforms list is empty; cannot compute global geometry.")
    height, width = frames[0].shape[:2]
    corners = np.array(
        [[0.0, 0.0, 1.0], [width, 0.0, 1.0], [width, height, 1.0], [0.0, height, 1.0]],
        dtype=np.float32,
    ).T
    all_xs: list[float] = []
    all_ys: list[float] = []
    for transform in transforms:
        pts = (transform @ corners)[:2].T
        all_xs.extend(pts[:, 0].tolist())
        all_ys.extend(pts[:, 1].tolist())
    min_x = math.floor(min(all_xs))
    min_y = math.floor(min(all_ys))
    max_x = math.ceil(max(all_xs))
    max_y = math.ceil(max(all_ys))
    print(f"[panorama] Global min_x={min_x}, min_y={min_y}")
    T_anchor = np.array(
        [[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    canvas_w = int(max_x - min_x)
    canvas_h = int(max_y - min_y)
    return T_anchor, (canvas_w, canvas_h)


def render_mosaic(
    frames: List[np.ndarray],
    transforms: List[np.ndarray],
    strip_idx: float,
    T_anchor: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    dx: np.ndarray,
    dy: np.ndarray,
    strip_width: int,
    strip_width_min: int,
    strip_width_dx_only: bool,
    progress_every: int,
    seam_mode: str,
    seam_expand_ratio: float,
    seam_feather: int,
    seam_blend: str,
    seam_pyr_levels: int,
) -> np.ndarray:
    """Render a strip panorama onto a fixed master canvas."""
    if not frames:
        raise ValueError("Frames list is empty; cannot render mosaic.")
    height, width = frames[0].shape[:2]
    mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    base_center_x = float(strip_idx)

    for count, frame_idx in enumerate(range(len(frames))):
        frame = frames[frame_idx]
        if frame_idx == 0:
            cur_strip_width = strip_width
        else:
            disp = math.hypot(float(dx[frame_idx]), float(dy[frame_idx]))
            if strip_width_dx_only:
                cur_strip_width = max(1, int(round(disp)))
            else:
                cur_strip_width = max(strip_width_min, int(round(disp)))

        M_final = T_anchor @ transforms[frame_idx]
        strip_point = M_final @ np.array([base_center_x, height / 2.0, 1.0], dtype=np.float32)
        strip_center_x = float(strip_point[0])
        warped, warped_mask = warp_with_mask_perspective(frame, M_final, (canvas_w, canvas_h))
        if seam_mode == "graphcut":
            graphcut_strip(
                mosaic,
                weight,
                warped,
                warped_mask,
                strip_center_x,
                cur_strip_width,
                seam_expand_ratio=seam_expand_ratio,
                seam_feather=seam_feather,
                blend_mode=seam_blend,
                pyramid_levels=seam_pyr_levels,
            )
        elif seam_mode == "alpha":
            extra = max(int(round(cur_strip_width * seam_expand_ratio)), 0)
            half = cur_strip_width // 2 + extra
            # Use floor/ceil to keep adjacent strips overlapping and avoid 1px gaps.
            x0 = max(int(math.floor(strip_center_x - half)), 0)
            x1 = min(int(math.ceil(strip_center_x + half)), canvas_w)
            full_mask = np.zeros_like(warped_mask, dtype=np.uint8)
            full_mask[:, x0:x1] = warped_mask[:, x0:x1]
            blend_with_mask(
                mosaic,
                weight,
                warped,
                full_mask,
                seam_feather,
                x0,
                x1,
                blend_mode=seam_blend,
                pyramid_levels=seam_pyr_levels,
            )
        else:
            overwrite_strip(mosaic, weight, warped, warped_mask, strip_center_x, cur_strip_width)

        if progress_every and count % progress_every == 0:
            print(f"[panorama] Pasted frame {count}/{len(frames)-1}")

    norm = np.where(weight > 0, weight, 1.0)[..., None]
    mosaic = (mosaic / norm).clip(0, 255).astype(np.uint8)

    return mosaic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a center-strip panorama from a video.")
    parser.add_argument("--input", required=True, type=Path, help="Input video (use stabilized video here).")
    parser.add_argument("--frames", type=Path, help="Optional video to read frames from (overrides --input).")
    parser.add_argument("--output", required=True, type=Path, help="Output panorama image (e.g., panorama.png).")
    parser.add_argument("--load-pairwise", type=Path, help="Load pairwise transforms (.npy) instead of estimating.")
    parser.add_argument("--save-pairwise", type=Path, help="Save estimated pairwise transforms (.npy).")
    parser.add_argument("--min-matches", type=int, default=2, help="Minimum matches for transform estimation.")
    parser.add_argument("--max-frames", type=int, help="Optional limit on frames.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N frames.")
    parser.add_argument("--strip-width", type=int, default=200, help="Strip width for the first frame (pixels).")
    parser.add_argument("--strip-width-min", type=int, default=10, help="Minimum strip width when using translation length.")
    parser.add_argument(
        "--strip-width-dx-only",
        action="store_true",
        help="Use the raw |dx| for strip width (no minimum clamp).",
    )
    parser.add_argument(
        "--strip-center-ratio",
        type=float,
        default=0.5,
        help="Relative strip center in [0,1]: 0=left, 0.5=center, 1=right.",
    )
    parser.add_argument(
        "--source-strip-ratio",
        type=float,
        help="Relative strip center in [0,1] (overrides --strip-center-ratio).",
    )
    parser.add_argument("--smooth-radius", type=int, default=5, help="Moving average radius for translation smoothing.")
    parser.add_argument("--no-recenter", action="store_true", help="Disable recentring transforms to median center.")
    parser.add_argument(
        "--no-translation-only",
        action="store_true",
        help="Disable translation-only estimation (use full rigid estimation).",
    )
    parser.add_argument("--no-post-crop-black", action="store_true", help="Disable pre-crop of black borders.")
    parser.add_argument("--crop-threshold", type=int, default=30, help="Threshold for black cropping (0-255).")
    parser.add_argument("--pre-crop-margin", type=int, default=0, help="Extra margin for black cropping (pixels).")
    parser.add_argument(
        "--seam",
        choices=["overwrite", "graphcut", "alpha"],
        default="overwrite",
        help="Seam method when pasting strips.",
    )
    parser.add_argument(
        "--seam-expand-ratio",
        type=float,
        default=0.0,
        help="Additional overlap as a fraction of strip width (e.g., 0.2 adds 20% of strip width).",
    )
    parser.add_argument(
        "--seam-feather",
        type=int,
        default=5,
        help="Gaussian feather radius applied to the seam mask.",
    )
    parser.add_argument(
        "--seam-blend",
        choices=["feather", "pyramid"],
        default="feather",
        help="Blending mode for seam mask.",
    )
    parser.add_argument(
        "--seam-pyr-levels",
        type=int,
        default=3,
        help="Number of pyramid levels when seam-blend=pyramid.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["auto", "angle", "ransac", "angle_only"],
        default="auto",
        help="Which candidate to use when estimating the rigid transform.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_path = args.frames or args.input
    pairwise = load_or_estimate_pairwise(
        motion_path=frames_path,
        load_matrices=args.load_pairwise,
        save_matrices=args.save_pairwise,
        min_matches=args.min_matches,
        max_frames=args.max_frames,
        progress_every=args.progress_every,
        translation_only=not args.no_translation_only,
        candidate_mode=args.candidate_mode,
    )

    cap = cv2.VideoCapture(str(frames_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {frames_path}")
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {frames_path}")

    post_crop_box = None
    if not args.no_post_crop_black:
        post_crop_box = compute_crop_box(first_frame, args.crop_threshold, args.pre_crop_margin)
        first_frame = crop_frame(first_frame, post_crop_box)
    height, width = first_frame.shape[:2]

    # Build cumulative transforms.
    dx = np.array([t[0, 2] for t in pairwise], dtype=np.float32)
    dy = np.array([t[1, 2] for t in pairwise], dtype=np.float32)
    if args.smooth_radius > 0:
        dx = moving_average_1d(dx, args.smooth_radius)
        dy = moving_average_1d(dy, args.smooth_radius)
    dx = np.concatenate([[0.0], dx])
    dy = np.concatenate([[0.0], dy])
    transforms = [ensure_homogeneous(t) for t in compose_cumulative_affine(pairwise)]

    if post_crop_box is not None:
        crop_x0, crop_y0, _, _ = post_crop_box
        crop_shift = np.array([[1.0, 0.0, float(crop_x0)], [0.0, 1.0, float(crop_y0)]], dtype=np.float32)
        transforms = [ensure_homogeneous(t) @ affine_to_homogeneous(crop_shift) for t in transforms]

    if not args.no_recenter:
        centers = []
        center_vec = np.array([width / 2.0, height / 2.0, 1.0], dtype=np.float32)
        for transform in transforms:
            warped_center = transform @ center_vec
            centers.append(warped_center[:2])
        centers = np.array(centers, dtype=np.float32)
        med_x = float(np.median(centers[:, 0]))
        med_y = float(np.median(centers[:, 1]))
        recenter_shift = np.array(
            [[1.0, 0.0, width / 2.0 - med_x], [0.0, 1.0, height / 2.0 - med_y], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        transforms = [recenter_shift @ t for t in transforms]
        print("[panorama] Recentering cumulative positions to median center.")

    # Read all frames once to keep indexing consistent.
    frames = [first_frame]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if post_crop_box is not None:
        frames = [crop_frame(f, post_crop_box) for f in frames]
    num_frames = min(len(frames), len(dx), len(transforms))
    frames = frames[:num_frames]
    dx = dx[:num_frames]
    dy = dy[:num_frames]
    transforms = transforms[:num_frames]
    if args.max_frames is not None and args.max_frames < num_frames:
        frames = frames[: args.max_frames]
        dx = dx[: args.max_frames]
        dy = dy[: args.max_frames]
        transforms = transforms[: args.max_frames]

    ratio = args.source_strip_ratio if args.source_strip_ratio is not None else args.strip_center_ratio
    ratio = max(0.0, min(1.0, float(ratio)))
    strip_idx = ratio * (width - 1)

    T_anchor, canvas_size = get_global_geometry(frames, transforms)
    canvas_w, canvas_h = canvas_size
    print(f"[panorama] Fixed Canvas Size: {canvas_w}x{canvas_h}")
    mosaic = render_mosaic(
        frames=frames,
        transforms=transforms,
        strip_idx=strip_idx,
        T_anchor=T_anchor,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        dx=dx,
        dy=dy,
        strip_width=args.strip_width,
        strip_width_min=args.strip_width_min,
        strip_width_dx_only=args.strip_width_dx_only,
        progress_every=args.progress_every,
        seam_mode=args.seam,
        seam_expand_ratio=args.seam_expand_ratio,
        seam_feather=args.seam_feather,
        seam_blend=args.seam_blend,
        seam_pyr_levels=args.seam_pyr_levels,
    )

    cv2.imwrite(str(args.output), mosaic)
    print(f"[panorama] Saved panorama to {args.output}")


if __name__ == "__main__":
    main()
