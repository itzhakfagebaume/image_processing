"""
Generate a stereo sweep video by varying the strip center and applying a convergence shift.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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


@dataclass
class PanoramaState:
    frames: list[np.ndarray]
    transforms: list[np.ndarray]
    dx: np.ndarray
    dy: np.ndarray
    T_anchor: np.ndarray
    canvas_w: int
    canvas_h: int
    frame_w: int
    frame_h: int
    fps: float


def parse_point(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected format x,y")
    try:
        x = int(round(float(parts[0].strip())))
        y = int(round(float(parts[1].strip())))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected numeric coordinates: x,y") from exc
    return x, y


def pick_point(image: np.ndarray, title: str, prompt: str, max_w: int, max_h: int) -> Tuple[int, int]:
    h, w = image.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale < 1.0:
        disp_w = max(int(round(w * scale)), 1)
        disp_h = max(int(round(h * scale)), 1)
        display = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        print(f"[stereo] Displaying {title} at {disp_w}x{disp_h} (scale={scale:.3f})")
    else:
        display = image
        scale = 1.0

    clicked: list[Tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[:] = [(x, y)]

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, display)
    cv2.setMouseCallback(title, on_mouse)
    print(prompt)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if clicked:
            x_disp, y_disp = clicked[0]
            x = int(round(x_disp / scale))
            y = int(round(y_disp / scale))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            preview = display.copy()
            cv2.drawMarker(
                preview,
                (x_disp, y_disp),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
            )
            cv2.imshow(title, preview)
            cv2.waitKey(200)
            break
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("Selection canceled.")
    cv2.destroyWindow(title)
    return x, y


def shift_image(image: np.ndarray, shift: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    shift_x, shift_y = float(shift[0]), float(shift[1])
    if abs(shift_x) < 1e-3 and abs(shift_y) < 1e-3:
        return image
    M = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        M,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def load_mosaic(path: Path, name: str, expected_size: Tuple[int, int]) -> np.ndarray:
    mosaic = cv2.imread(str(path))
    if mosaic is None:
        raise RuntimeError(f"Failed to read {name} mosaic: {path}")
    if (mosaic.shape[1], mosaic.shape[0]) != expected_size:
        print(
            f"[stereo] Warning: {name} mosaic size {mosaic.shape[1]}x{mosaic.shape[0]} "
            f"differs from expected {expected_size[0]}x{expected_size[1]}"
        )
    return mosaic


def build_panorama_state(
    input_path: Path,
    stabilized_path: Path,
    min_matches: int,
    max_frames: Optional[int],
    progress_every: int,
    save_stabilized_matrices: Optional[Path],
    load_stabilized_matrices: Optional[Path],
    save_pairwise: Optional[Path],
    load_pairwise: Optional[Path],
    smooth_radius: int,
    smooth_x: bool,
    recenter: bool,
    post_crop_black: bool,
    crop_threshold: int,
    pre_crop_margin: int,
    rotation_zero_thresh_deg: float,
    center_reference: bool,
) -> PanoramaState:
    if load_stabilized_matrices is not None or not stabilized_path.exists():
        print(f"[stereo] Stabilizing {input_path} -> {stabilized_path}")
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
        )
    else:
        print(f"[stereo] Using existing stabilized video: {stabilized_path}")

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
        )
    else:
        print("[stereo] Estimating pairwise transforms for trajectory...")
        pairwise_for_traj = load_or_estimate_pairwise(
            motion_path=motion_path,
            load_matrices=None,
            save_matrices=save_pairwise,
            min_matches=min_matches,
            max_frames=max_frames,
            progress_every=progress_every,
            translation_only=False,
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
        print(f"[stereo] Anchoring trajectory to center frame index {mid}.")

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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

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
        print("[stereo] Recentering cumulative positions to median center.")

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
    print(f"[stereo] Fixed canvas size: {canvas_w}x{canvas_h}")
    return PanoramaState(
        frames=frames,
        transforms=transforms,
        dx=dx,
        dy=dy,
        T_anchor=T_anchor,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        frame_w=width,
        frame_h=height,
        fps=fps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a stereo sweep video from center-strip mosaics.")
    parser.add_argument("--input", required=True, type=Path, help="Input video.")
    parser.add_argument("--output", required=True, type=Path, help="Output stereo video (e.g., stereo.mp4).")
    parser.add_argument(
        "--stabilized",
        type=Path,
        help="Optional stabilized video path (default: <input>_stabilized.mp4).",
    )
    parser.add_argument("--save-stabilized-matrices", type=Path, help="Save stabilized transforms (.npy).")
    parser.add_argument("--load-stabilized-matrices", type=Path, help="Load pairwise transforms for stabilization.")
    parser.add_argument("--save-pairwise", type=Path, help="Save pairwise transforms used for trajectory (.npy).")
    parser.add_argument("--load-pairwise", type=Path, help="Load pairwise transforms used for trajectory (.npy).")
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
        help="Per-mosaic frame logging interval (defaults to --progress-every).",
    )
    parser.add_argument("--strip-width", type=int, default=200, help="Strip width for the first frame (pixels).")
    parser.add_argument("--strip-width-min", type=int, default=10, help="Minimum strip width when using translation length.")
    parser.add_argument(
        "--strip-width-dx-only",
        action="store_true",
        help="Use the raw |dx| for strip width (no minimum clamp).",
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
        help="Anchor trajectory to the center frame (disables recentering).",
    )
    parser.add_argument("--rotation-zero-thresh-deg", type=float, default=2.0, help="Clamp small rotations to zero (degrees).")
    parser.add_argument("--num-frames", type=int, default=60, help="Number of stereo frames to render.")
    parser.add_argument("--fps", type=float, help="Override output FPS (default: stabilized video FPS).")
    parser.add_argument(
        "--left-mosaic",
        type=Path,
        help="Optional path to pre-rendered left calibration mosaic (matches --calib-left-ratio).",
    )
    parser.add_argument(
        "--right-mosaic",
        type=Path,
        help="Optional path to pre-rendered right calibration mosaic (matches --calib-right-ratio).",
    )
    parser.add_argument("--left-point", type=parse_point, help="Optional calibration point in left mosaic: x,y")
    parser.add_argument("--right-point", type=parse_point, help="Optional calibration point in right mosaic: x,y")
    parser.add_argument("--calib-left-ratio", type=float, default=0.0, help="Strip ratio for left calibration mosaic.")
    parser.add_argument("--calib-right-ratio", type=float, default=1.0, help="Strip ratio for right calibration mosaic.")
    parser.add_argument("--display-max-w", type=int, default=1200, help="Max display width for selection.")
    parser.add_argument("--display-max-h", type=int, default=700, help="Max display height for selection.")
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

    frame_progress_every = args.frame_progress_every if args.frame_progress_every is not None else args.progress_every
    state = build_panorama_state(
        input_path=args.input,
        stabilized_path=stabilized_path,
        min_matches=args.min_matches,
        max_frames=args.max_frames,
        progress_every=args.progress_every,
        save_stabilized_matrices=args.save_stabilized_matrices,
        load_stabilized_matrices=args.load_stabilized_matrices,
        save_pairwise=args.save_pairwise,
        load_pairwise=args.load_pairwise,
        smooth_radius=args.smooth_radius,
        smooth_x=not args.no_smooth_x,
        recenter=not args.no_recenter,
        post_crop_black=not args.no_post_crop_black,
        crop_threshold=args.crop_threshold,
        pre_crop_margin=args.pre_crop_margin,
        rotation_zero_thresh_deg=args.rotation_zero_thresh_deg,
        center_reference=args.center_reference,
    )

    if (args.left_point is None) != (args.right_point is None):
        raise SystemExit("Provide both --left-point and --right-point or neither.")

    calib_left_ratio = max(0.0, min(1.0, float(args.calib_left_ratio)))
    calib_right_ratio = max(0.0, min(1.0, float(args.calib_right_ratio)))
    if abs(calib_right_ratio - calib_left_ratio) < 1e-6:
        raise SystemExit("Calibration ratios must be different.")
    left_strip_idx = calib_left_ratio * max(state.frame_w - 1, 0)
    right_strip_idx = calib_right_ratio * max(state.frame_w - 1, 0)
    if args.left_mosaic is not None:
        left_mosaic = load_mosaic(args.left_mosaic, "left", (state.canvas_w, state.canvas_h))
    else:
        print("[stereo] Rendering left mosaic for calibration...")
        left_mosaic = render_mosaic(
            frames=state.frames,
            transforms=state.transforms,
            strip_idx=left_strip_idx,
            T_anchor=state.T_anchor,
            canvas_w=state.canvas_w,
            canvas_h=state.canvas_h,
            dx=state.dx,
            dy=state.dy,
            strip_width=args.strip_width,
            strip_width_min=args.strip_width_min,
            strip_width_dx_only=args.strip_width_dx_only,
            progress_every=frame_progress_every,
            seam_mode=args.seam,
            seam_expand_ratio=args.seam_expand_ratio,
            seam_feather=args.seam_feather,
            seam_blend=args.seam_blend,
            seam_pyr_levels=args.seam_pyr_levels,
        )
    if args.right_mosaic is not None:
        right_mosaic = load_mosaic(args.right_mosaic, "right", (state.canvas_w, state.canvas_h))
    else:
        print("[stereo] Rendering right mosaic for calibration...")
        right_mosaic = render_mosaic(
            frames=state.frames,
            transforms=state.transforms,
            strip_idx=right_strip_idx,
            T_anchor=state.T_anchor,
            canvas_w=state.canvas_w,
            canvas_h=state.canvas_h,
            dx=state.dx,
            dy=state.dy,
            strip_width=args.strip_width,
            strip_width_min=args.strip_width_min,
            strip_width_dx_only=args.strip_width_dx_only,
            progress_every=frame_progress_every,
            seam_mode=args.seam,
            seam_expand_ratio=args.seam_expand_ratio,
            seam_feather=args.seam_feather,
            seam_blend=args.seam_blend,
            seam_pyr_levels=args.seam_pyr_levels,
        )

    left_point = args.left_point
    right_point = args.right_point
    if left_point is None:
        left_point = pick_point(
            left_mosaic,
            "Left mosaic",
            "Click the convergence object in the LEFT mosaic (ESC to cancel).",
            args.display_max_w,
            args.display_max_h,
        )
    if right_point is None:
        right_point = pick_point(
            right_mosaic,
            "Right mosaic",
            "Click the same object in the RIGHT mosaic (ESC to cancel).",
            args.display_max_w,
            args.display_max_h,
        )

    shift = np.array([left_point[0] - right_point[0], left_point[1] - right_point[1]], dtype=np.float32)
    print(
        f"[stereo] Convergence shift dx={shift[0]:.2f}, dy={shift[1]:.2f} "
        f"(calib ratios {calib_left_ratio:.3f}->{calib_right_ratio:.3f})"
    )

    fps = args.fps if args.fps is not None else state.fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (state.canvas_w, state.canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {args.output}")

    total = max(args.num_frames, 1)
    for idx in range(total):
        t = 0.0 if total == 1 else idx / (total - 1)
        strip_idx = t * max(state.frame_w - 1, 0)
        mosaic = render_mosaic(
            frames=state.frames,
            transforms=state.transforms,
            strip_idx=strip_idx,
            T_anchor=state.T_anchor,
            canvas_w=state.canvas_w,
            canvas_h=state.canvas_h,
            dx=state.dx,
            dy=state.dy,
            strip_width=args.strip_width,
            strip_width_min=args.strip_width_min,
            strip_width_dx_only=args.strip_width_dx_only,
            progress_every=frame_progress_every,
            seam_mode=args.seam,
            seam_expand_ratio=args.seam_expand_ratio,
            seam_feather=args.seam_feather,
            seam_blend=args.seam_blend,
            seam_pyr_levels=args.seam_pyr_levels,
        )
        # Linearly interpolate the convergence shift relative to the calibration strip ratios.
        alpha = (t - calib_left_ratio) / (calib_right_ratio - calib_left_ratio)
        shift_t = shift * alpha
        shifted = shift_image(mosaic, shift_t, state.canvas_w, state.canvas_h)
        writer.write(shifted)

        if args.progress_every and idx % args.progress_every == 0:
            print(
                f"[stereo] Wrote frame {idx}/{total - 1} "
                f"(strip_idx={strip_idx:.1f}, shift=({shift_t[0]:.1f},{shift_t[1]:.1f}))"
            )

    writer.release()
    cv2.destroyAllWindows()
    print(f"[stereo] Saved stereo sweep to {args.output}")


if __name__ == "__main__":
    main()
