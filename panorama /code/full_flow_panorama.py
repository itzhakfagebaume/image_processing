"""
End-to-end pipeline: stabilize a video, then build a center-strip panorama.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from panorama_center_strip import (
    affine_to_homogeneous,
    compute_crop_box,
    crop_frame,
    ensure_homogeneous,
    get_global_geometry,
    load_or_estimate_pairwise,
    render_mosaic,
)
from stabilize_video import compose_cumulative, horizontal_trajectory, stabilize_video


def run_pipeline(
    input_path: Path,
    output_path: Path,
    stabilized_path: Path,
    min_matches: int,
    max_frames: Optional[int],
    progress_every: int,
    frame_progress_every: int,
    save_stabilized_matrices: Optional[Path],
    load_stabilized_matrices: Optional[Path],
    save_pairwise: Optional[Path],
    load_pairwise: Optional[Path],
    strip_width: int,
    strip_width_min: int,
    strip_width_dx_only: bool,
    strip_center_ratios: list[float],
    source_strip_ratio: Optional[float],
    seam_mode: str = "graphcut",
    seam_expand_ratio: float = 0.0,
    seam_feather: int = 5,
    seam_blend: str = "feather",
    seam_pyr_levels: int = 3,
    smooth_radius: int = 5,
    recenter: bool = True,
    post_crop_black: bool = True,
    crop_threshold: int = 30,
    pre_crop_margin: int = 0,
    smooth_x: bool = True,
    rotation_zero_thresh_deg: float = 2.0,
    center_reference: bool = False,
    candidate_mode: str = "auto",
) -> None:
    if load_stabilized_matrices is not None or not stabilized_path.exists():
        print(f"[pipeline] Stabilizing {input_path} -> {stabilized_path}")
        stabilize_video(
            input_path=input_path,
            output_path=stabilized_path,
            smoothing_radius=5,
            max_frames=max_frames,
            min_matches=min_matches,
            progress_every=progress_every,
            save_matrices=save_pairwise,
            smooth_x=False,
            load_matrices=load_pairwise if load_pairwise is not None else load_stabilized_matrices,
            save_stabilized=save_stabilized_matrices,
            pairwise_smooth_radius=0,
            pairwise_angle_thresh_deg=2.0,
            pairwise_trans_thresh=4.0,
            rotation_zero_thresh_deg=rotation_zero_thresh_deg,
            post_crop_black=True,
            post_crop_threshold=40,
            post_crop_margin=0,
            candidate_mode=candidate_mode,
        )
    else:
        print(f"[pipeline] Using existing stabilized video: {stabilized_path}")

    motion_path = stabilized_path
    if save_pairwise is not None and save_pairwise.exists():
        load_pairwise = save_pairwise

    if load_pairwise is not None:
        pairwise_for_traj = load_or_estimate_pairwise(
            motion_path=motion_path,
            load_matrices=load_pairwise,
            save_matrices=None,
            min_matches=min_matches,
            max_frames=max_frames,
            progress_every=progress_every,
            translation_only=False,
            candidate_mode=candidate_mode,
        )
    elif save_pairwise is not None and save_pairwise.exists():
        pairwise_for_traj = load_or_estimate_pairwise(
            motion_path=motion_path,
            load_matrices=save_pairwise,
            save_matrices=None,
            min_matches=min_matches,
            max_frames=max_frames,
            progress_every=progress_every,
            translation_only=False,
            candidate_mode=candidate_mode,
        )
    else:
        print("[pipeline] Estimating pairwise transforms for trajectory...")
        pairwise_for_traj = load_or_estimate_pairwise(
            motion_path=motion_path,
            load_matrices=None,
            save_matrices=save_pairwise,
            min_matches=min_matches,
            max_frames=max_frames,
            progress_every=progress_every,
            translation_only=False,
            candidate_mode=candidate_mode,
        )

    cumulative = compose_cumulative(pairwise_for_traj)
    cap = cv2.VideoCapture(str(stabilized_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {stabilized_path}")
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read first frame from {stabilized_path}")
    height, width = first_frame.shape[:2]
    tx = horizontal_trajectory(cumulative, smoothing_radius=smooth_radius, smooth_x=smooth_x, width=width, height=height)
    ty = np.zeros_like(tx)
    tx = -tx
    if center_reference and len(tx) > 0:
        mid = len(tx) // 2
        tx = tx - tx[mid]
        ty = ty - ty[mid]
        print(f"[pipeline] Anchoring trajectory to center frame index {mid}.")

    multi = len(strip_center_ratios) > 1
    decimals = 4

    local_recenter = recenter
    local_post_crop_black = post_crop_black
    if center_reference:
        local_recenter = False
    cap = cv2.VideoCapture(str(stabilized_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {stabilized_path}")
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Failed to read first frame from {stabilized_path}")

    post_crop_box = None
    if local_post_crop_black:
        post_crop_box = compute_crop_box(first_frame, crop_threshold, pre_crop_margin)
        first_frame = crop_frame(first_frame, post_crop_box)
    height, width = first_frame.shape[:2]

    dx = np.diff(tx, prepend=tx[0]).astype(np.float32)
    dy = np.diff(ty, prepend=ty[0]).astype(np.float32)
    transforms = [
        ensure_homogeneous(np.array([[1.0, 0.0, float(x)], [0.0, 1.0, float(y)]], dtype=np.float32))
        for x, y in zip(tx, ty)
    ]
    if post_crop_box is not None:
        crop_x0, crop_y0, _, _ = post_crop_box
        crop_shift = np.array([[1.0, 0.0, float(crop_x0)], [0.0, 1.0, float(crop_y0)]], dtype=np.float32)
        transforms = [ensure_homogeneous(t) @ affine_to_homogeneous(crop_shift) for t in transforms]

    if local_recenter:
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
        print("[pipeline] Recentering cumulative positions to median center.")

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
    if max_frames is not None and max_frames < num_frames:
        frames = frames[:max_frames]
        dx = dx[:max_frames]
        dy = dy[:max_frames]
        transforms = transforms[:max_frames]

    T_anchor, canvas_size = get_global_geometry(frames, transforms)
    canvas_w, canvas_h = canvas_size
    print(f"[pipeline] Fixed Canvas Size: {canvas_w}x{canvas_h}")

    right_idx = max(width - 1, 0)
    center_idx = width // 2 if width > 0 else 0
    test_outputs = [
        ("mosaic_left.png", 0),
        ("mosaic_center.png", center_idx),
        ("mosaic_right.png", right_idx),
    ]
    for name, strip_idx in test_outputs:
        out_path = output_path.with_name(name)
        print(f"[pipeline] Rendering test mosaic: {out_path.name} (strip_idx={strip_idx})")
        mosaic = render_mosaic(
            frames=frames,
            transforms=transforms,
            strip_idx=strip_idx,
            T_anchor=T_anchor,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            dx=dx,
            dy=dy,
            strip_width=strip_width,
            strip_width_min=strip_width_min,
            strip_width_dx_only=strip_width_dx_only,
            progress_every=frame_progress_every,
            seam_mode=seam_mode,
            seam_expand_ratio=seam_expand_ratio,
            seam_feather=seam_feather,
            seam_blend=seam_blend,
            seam_pyr_levels=seam_pyr_levels,
        )
        cv2.imwrite(str(out_path), mosaic)
        print(f"[pipeline] Saved mosaic to {out_path}")

    total = len(strip_center_ratios)
    for idx, ratio in enumerate(strip_center_ratios):
        suffix = ""
        if multi:
            ratio_str = f"{ratio:.{decimals}f}".replace(".", "p")
            suffix = f"_r{ratio_str}"
        out_path = output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")
        print(f"[pipeline] Panorama {idx + 1}/{total} (r={ratio:.3f})")
        use_ratio = source_strip_ratio if source_strip_ratio is not None else ratio
        use_ratio = max(0.0, min(1.0, float(use_ratio)))
        strip_idx = use_ratio * (width - 1)
        mosaic = render_mosaic(
            frames=frames,
            transforms=transforms,
            strip_idx=strip_idx,
            T_anchor=T_anchor,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            dx=dx,
            dy=dy,
            strip_width=strip_width,
            strip_width_min=strip_width_min,
            strip_width_dx_only=strip_width_dx_only,
            progress_every=frame_progress_every,
            seam_mode=seam_mode,
            seam_expand_ratio=seam_expand_ratio,
            seam_feather=seam_feather,
            seam_blend=seam_blend,
            seam_pyr_levels=seam_pyr_levels,
        )
        cv2.imwrite(str(out_path), mosaic)
        print(f"[pipeline] Saved panorama to {out_path}")


def parse_ratio_list(value: str) -> list[float]:
    if not value:
        return []
    ratios = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ratios.append(float(part))
    return ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full pipeline: stabilize video then build panorama.")
    parser.add_argument("--input", required=True, type=Path, help="Input video.")
    parser.add_argument("--output", required=True, type=Path, help="Output panorama image.")
    parser.add_argument(
        "--stabilized",
        type=Path,
        help="Optional stabilized video path (default: <input>_stabilized.mp4).",
    )
    parser.add_argument("--save-stabilized-matrices", type=Path, help="Save stabilized transforms (.npy).")
    parser.add_argument("--load-stabilized-matrices", type=Path, help="Load pairwise transforms for stabilization.")
    parser.add_argument("--save-pairwise", type=Path, help="Save pairwise transforms used for stabilization (.npy).")
    parser.add_argument("--load-pairwise", type=Path, help="Load pairwise transforms used for stabilization (.npy).")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory for cached transforms (default: input video folder).",
    )
    parser.add_argument(
        "--no-cache-transforms",
        action="store_true",
        help="Disable automatic save/reuse of cached transforms.",
    )
    parser.add_argument("--min-matches", type=int, default=2, help="Minimum matches for transform estimation.")
    parser.add_argument("--max-frames", type=int, help="Optional limit on frames.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N frames.")
    parser.add_argument(
        "--frame-progress-every",
        type=int,
        help="Per-panorama frame logging interval (defaults to --progress-every).",
    )
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
        "--strip-center-ratios",
        type=str,
        help="Comma-separated strip center ratios; when set, outputs multiple panoramas.",
    )
    parser.add_argument(
        "--source-strip-ratio",
        type=float,
        help="Relative strip center in [0,1] (overrides --strip-center-ratio).",
    )
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
    parser.add_argument("--smooth-radius", type=int, default=5, help="Moving average radius for translation smoothing.")
    parser.add_argument("--no-recenter", action="store_true", help="Disable recentring to median center.")
    parser.add_argument("--no-post-crop-black", action="store_true", help="Disable pre-crop of black borders.")
    parser.add_argument("--crop-threshold", type=int, default=30, help="Threshold for black cropping (0-255).")
    parser.add_argument("--pre-crop-margin", type=int, default=0, help="Extra margin for black cropping (pixels).")
    parser.add_argument("--no-smooth-x", action="store_true", help="Disable horizontal smoothing (default on).")
    parser.add_argument(
        "--center-reference",
        action="store_true",
        help="Anchor panorama trajectory to the center frame instead of frame 0 (disables recentering).",
    )
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
    stabilized_path = args.stabilized
    if stabilized_path is None:
        stabilized_path = args.input.with_name(f"{args.input.stem}_stabilized.mp4")

    cache_dir = args.cache_dir or args.input.parent
    default_pairwise = cache_dir / f"{args.input.stem}_pairwise.npy"
    if args.save_pairwise is None:
        args.save_pairwise = default_pairwise
        args.save_pairwise.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_cache_transforms:
        cache_dir.mkdir(parents=True, exist_ok=True)
        default_stabilized = cache_dir / f"{args.input.stem}_stabilized.npy"
        if args.load_pairwise is None and default_pairwise.exists():
            args.load_pairwise = default_pairwise
        if args.save_stabilized_matrices is None:
            args.save_stabilized_matrices = default_stabilized

    ratios = parse_ratio_list(args.strip_center_ratios) if args.strip_center_ratios else []
    if not ratios:
        ratios = [args.strip_center_ratio]
    frame_progress_every = args.frame_progress_every if args.frame_progress_every is not None else args.progress_every

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        stabilized_path=stabilized_path,
        min_matches=args.min_matches,
        max_frames=args.max_frames,
        progress_every=args.progress_every,
        frame_progress_every=frame_progress_every,
        save_stabilized_matrices=args.save_stabilized_matrices,
        load_stabilized_matrices=args.load_stabilized_matrices,
        save_pairwise=args.save_pairwise,
        load_pairwise=args.load_pairwise,
        strip_width=args.strip_width,
        strip_width_min=args.strip_width_min,
        strip_width_dx_only=args.strip_width_dx_only,
        strip_center_ratios=ratios,
        source_strip_ratio=args.source_strip_ratio,
        seam_mode=args.seam,
        seam_expand_ratio=args.seam_expand_ratio,
        seam_feather=args.seam_feather,
        seam_blend=args.seam_blend,
        seam_pyr_levels=args.seam_pyr_levels,
        smooth_radius=args.smooth_radius,
        recenter=not args.no_recenter,
        post_crop_black=not args.no_post_crop_black,
        crop_threshold=args.crop_threshold,
        pre_crop_margin=args.pre_crop_margin,
        smooth_x=not args.no_smooth_x,
        rotation_zero_thresh_deg=args.rotation_zero_thresh_deg,
        center_reference=args.center_reference,
        candidate_mode=args.candidate_mode,
    )


if __name__ == "__main__":
    main()
