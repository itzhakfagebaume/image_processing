

from __future__ import annotations


from pathlib import Path
from typing import List, Optional, Tuple
import os, glob

import cv2
import numpy as np
from PIL import Image

# ---- Import de TES fonctions depuis ex_4_video_stable.py
import sys
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ex_4_video_stable import estimate_pair_transform, to_homog  # réutilise tes fonctions


# ============================================================
# 1) I/O vidéo + utils
# ============================================================

def read_video_frames(video_path: str, max_frames: Optional[int] = None, step: int = 1) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo: {video_path}")

    frames: List[np.ndarray] = []
    idx, kept = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step == 0:
            frames.append(frame)
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break

        idx += 1

    cap.release()

    if len(frames) < 2:
        raise ValueError("Il faut au moins 2 frames")

    return frames


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# ============================================================
# 2) Pairwise (curr->prev) + Cumulative (frame i -> frame 0)
# ============================================================

def compute_cumulative(frames: List[np.ndarray], show_progress: bool = True) -> List[np.ndarray]:
    prev_gray = to_gray(frames[0])

    cumulative: List[np.ndarray] = [np.eye(3, dtype=np.float32)]

    for i in range(1, len(frames)):
        curr_gray = to_gray(frames[i])

        # estimate_pair_transform(prev, curr) -> T(2x3) mappe curr -> prev
        T = estimate_pair_transform(prev_gray, curr_gray)
        H = to_homog(T).astype(np.float32)

        # frame i -> frame 0
        cumulative.append(cumulative[-1] @ H)

        prev_gray = curr_gray



    return cumulative


# ============================================================
# 3) Conserver UNIQUEMENT le mouvement horizontal (tx)
# ============================================================

def compute_tx_from_cumulative(cumulative: List[np.ndarray], width: int, height: int) -> np.ndarray:
    center = np.array([width / 2.0, height / 2.0, 1.0], dtype=np.float32)

    dx = []
    for C in cumulative:
        A = np.linalg.inv(C)
        p = A @ center
        dx.append(float(p[0] - center[0]))

    tx = np.array(dx, dtype=np.float32)
    tx = tx - tx[0]
    return tx


def build_stabilized_affines(cumulative: List[np.ndarray], tx: np.ndarray) -> List[np.ndarray]:
    stabilized_2x3: List[np.ndarray] = []

    for C, x in zip(cumulative, tx):
        shift = np.array(
            [[1.0, 0.0, float(x)],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        M = (shift @ C).astype(np.float32)  # 3x3
        stabilized_2x3.append(M[:2, :])     # 2x3

    return stabilized_2x3


# ============================================================
# 4) Warp + crop une seule fois (on réutilise les crops pour toutes les views)
# ============================================================

def warp_and_crop_all(
    frames: List[np.ndarray],
    stabilized_2x3: List[np.ndarray],
    crop_percent: float,
    border_replicate: bool,
) -> Tuple[List[np.ndarray], int, int]:
    h, w = frames[0].shape[:2]

    crop_x = int(w * crop_percent)
    crop_y = int(h * crop_percent)

    out_w = w - 2 * crop_x
    out_h = h - 2 * crop_y

    if out_w <= 10 or out_h <= 10:
        raise ValueError("crop_percent trop grand: image trop petite")

    borderMode = cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT
    borderValue = (0, 0, 0)

    crops: List[np.ndarray] = []
    for frame, A in zip(frames, stabilized_2x3):
        warped = cv2.warpAffine(
            frame,
            A,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=borderValue,
        )
        cropped = warped[crop_y:crop_y + out_h, crop_x:crop_x + out_w]
        crops.append(cropped)

    return crops, out_w, out_h


# ============================================================
# 5) Construire UN panorama pour une view (colonne) donnée
# ============================================================

def compute_strip_widths(tx: np.ndarray, use_auto: bool, fixed_width: int, max_strip: int) -> List[int]:
    """Largeur du strip pour chaque frame.
    - auto: basé sur dx = |tx[i+1]-tx[i]| (comme ex4)
    - sinon: largeur fixe
    """
    N = len(tx)
    if not use_auto:
        w = max(1, int(fixed_width))
        return [w] * N

    # dx sur N-1 transitions
    dx = np.abs(np.diff(tx))
    dx = np.maximum(dx, 1.0)
    dx = np.minimum(dx, float(max_strip))

    widths = dx.astype(int).tolist()
    # pour avoir N widths, on répète la dernière
    if len(widths) < N:
        widths.append(widths[-1] if widths else max(1, int(fixed_width)))
    return widths


def build_panorama_for_view(
    crops: List[np.ndarray],
    view_frac: float,
    strip_widths: List[int],
) -> np.ndarray:
    """Colle un strip vertical de chaque frame (crop) autour d'une colonne donnée par view_frac."""

    out_h, out_w = crops[0].shape[:2]
    view_frac = float(np.clip(view_frac, 0.0, 1.0))
    col = int(view_frac * (out_w - 1))

    # largeur totale = somme des widths
    pano_w = int(np.sum(strip_widths))
    pano = np.zeros((out_h, pano_w, 3), dtype=np.uint8)

    x_cursor = 0
    for img, sw in zip(crops, strip_widths):
        sw = max(1, int(sw))
        half = sw // 2
        x0 = max(0, col - half)
        x1 = x0 + sw
        if x1 > out_w:
            x1 = out_w
            x0 = out_w - sw

        strip = img[:, x0:x1]
        pano[:, x_cursor:x_cursor + sw] = strip
        x_cursor += sw

    return pano[:, :x_cursor]


def autocrop_black(pano: np.ndarray) -> np.ndarray:
    """Crop simple: enlève bordures noires autour (bbox des pixels non noirs)."""
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    mask = gray > 2
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return pano
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad = 5
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(pano.shape[1] - 1, x1 + pad)
    y1 = min(pano.shape[0] - 1, y1 + pad)
    return pano[y0:y1 + 1, x0:x1 + 1]


# ============================================================
# 6) MULTI-VIEW: génère plusieurs panoramas (sweep)
# ============================================================

def generate_multiview_panoramas(
    input_path: str,
    out_dir: str,
    max_frames: int,
    step: int,
    crop_percent: float,
    border_replicate: bool,
    n_views: int,
    view_min: float,
    view_max: float,
    strip_width: int,
    auto_strip: bool,
    max_strip: int,
    autocrop: bool,
) -> List[Path]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    frames = read_video_frames(input_path, max_frames=max_frames, step=step)
    h, w = frames[0].shape[:2]

    cumulative = compute_cumulative(frames, show_progress=True)
    tx = compute_tx_from_cumulative(cumulative, w, h)
    stabilized_2x3 = build_stabilized_affines(cumulative, tx)

    crops, out_w, out_h = warp_and_crop_all(frames, stabilized_2x3, crop_percent, border_replicate)

    strip_widths = compute_strip_widths(tx, use_auto=auto_strip, fixed_width=strip_width, max_strip=max_strip)

    views = np.linspace(view_min, view_max, max(1, int(n_views))).astype(np.float32)
    views = np.clip(views, 0.0, 1.0)
    views = np.unique(views)
    views.sort()


    saved: List[Path] = []
    for k, vf in enumerate(views):
        pano = build_panorama_for_view(crops, view_frac=float(vf), strip_widths=strip_widths)
        if autocrop:
            pano = autocrop_black(pano)

        out_path = out_dir_p / f"panorama_view_{k:02d}_x{vf:.2f}.png"
        ok = cv2.imwrite(str(out_path), pano)
        if not ok:
            raise RuntimeError(f"Impossible d'écrire: {out_path}")

        saved.append(out_path)

    return saved


# ============================================================
# 7) MAIN
# ============================================================

def main():
    # Hardcoded parameters - modify these values directly in the code
    input_path = "video/Shinkansen.mp4"  # Change to your video path
    out_dir = "panoramas"
    max_frames = None
    step = 1
    crop_percent = 0.06
    border_constant = False  # True for black border, False for replicate
    n_views = 10
    view_min = 0.2
    view_max = 0.8
    strip_width = 8
    auto_strip = True  # Set to True to use automatic strip width based on dx
    max_strip = 30
    autocrop = False  # Set to True to automatically crop black borders

    generate_multiview_panoramas(
        input_path=input_path,
        out_dir=out_dir,
        max_frames=max_frames,
        step=step,
        crop_percent=crop_percent,
        border_replicate=(not border_constant),
        n_views=n_views,
        view_min=view_min,
        view_max=view_max,
        strip_width=strip_width,
        auto_strip=auto_strip,
        max_strip=max_strip,
        autocrop=autocrop,
    )


def _read_frames_dir(input_frames_path):
    files = sorted(glob.glob(os.path.join(input_frames_path, "frame_*.jpg")))
    if len(files) == 0:
        raise FileNotFoundError(f"No frames found in {input_frames_path}")

    frames = []
    for f in files:
        img = cv2.imread(f)  # BGR
        if img is None:
            continue
        frames.append(img)
    if len(frames) < 2:
        raise RuntimeError("Need at least 2 readable frames")
    return frames


def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point for ex4
    :param input_frames_path: dir with frame_%05d.jpg
    :param n_out_frames: number of panoramas to output
    :return: list of PIL Images (length n_out_frames)
    """
    if n_out_frames <= 0:
        return []

    # 1) Lire frames (BGR OpenCV)
    frames = _read_frames_dir(input_frames_path)
    h, w = frames[0].shape[:2]

    # 2) Reprendre ton pipeline existant (celui du multiview strip)
    cumulative = compute_cumulative(frames, show_progress=False)         # frame i -> frame 0
    tx = compute_tx_from_cumulative(cumulative, w, h)                    # horizontal only
    stabilized_2x3 = build_stabilized_affines(cumulative, tx)            # 2x3 for warpAffine

    crops, out_w, out_h = warp_and_crop_all(
        frames, stabilized_2x3,
        crop_percent=0.06,
        border_replicate=True
    )

    # largeur strip (comme ex4: auto via dx) ou fixe
    strip_widths = compute_strip_widths(
        tx,
        use_auto=True,      # mets False si tu veux fixe
        fixed_width=8,
        max_strip=30
    )

    # 3) Views TRIÉES (monotones) : n_out_frames panoramas
    #    Exemple: du 20% au 80% de la largeur
    views = np.linspace(0.2, 0.8, int(n_out_frames)).astype(np.float32)
    views = np.clip(views, 0.0, 1.0)
    views = np.unique(views)
    views.sort()

    # 4) Construire la liste de PIL images
    panoramas = []
    for vf in views[:n_out_frames]:
        pano_bgr = build_panorama_for_view(crops, view_frac=float(vf), strip_widths=strip_widths)

        # optionnel
        pano_bgr = autocrop_black(pano_bgr)

        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
        panoramas.append(Image.fromarray(pano_rgb))

    # 5) Assurer exactement n_out_frames (si unique a réduit)
    while len(panoramas) < n_out_frames:
        panoramas.append(panoramas[-1].copy())

    return panoramas

if __name__ == "__main__":
    # p = generate_panorama("frames_test", 5)
    # for i, im in enumerate(p):
    #     im.save(f"out_{i:02d}.png")
    main()
