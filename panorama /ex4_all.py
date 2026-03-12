from __future__ import annotations
import numpy as np

import cv2


# ============================================================
# 1) Transformation rigide robuste
# ============================================================
def compute_rigid_transform_svd(src: np.ndarray, dst: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Calcule une transformation rigide (rotation + translation) de src vers dst.
    Utilise la décomposition SVD de la matrice de covariance (méthode Kabsch).

    """
    if mask is not None:
        mask = mask.reshape(-1).astype(bool)
        src = src[mask]
        dst = dst[mask]

    if len(src) < 3:
        return np.eye(2, 3, dtype=np.float32)

    # 1. Centrer les points (enlever la translation)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # 2. Construire la matrice de covariance H = src^T * dst
    H = src_centered.T @ dst_centered

    # 3. Décomposition SVD de la covariance
    U, S, Vt = np.linalg.svd(H)

    # 4. Calculer la rotation R = V * U^T
    R = Vt.T @ U.T

    # 5. Correction si on obtient une réflexion (det < 0)
    if np.linalg.det(R) < 0:
        # Inverser la dernière colonne de V
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 6. Calculer la translation t = dst_mean - R * src_mean
    t = dst_mean - R @ src_mean

    # 7. Construire la matrice 2x3
    M = np.hstack([R, t.reshape(2, 1)])
    return M.astype(np.float32)



def match_sift_features(img1_gray, img2_gray, max_features=2000, ratio=0.6):
    """
    Trouve les correspondances SIFT entre deux images.
    Retourne (points_img1, points_img2) ou (None, None) si échec.
    """
    sift = cv2.SIFT_create(nfeatures=max_features)

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None

    # Matching avec ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Filtre de Lowe
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio * n.distance:
                good.append(m)

    if len(good) < 10:
        print("Pas assez de bonnes correspondances SIFT")
        return None, None
    good = sorted(good, key=lambda x: x.distance)
    # Extraire les coordonnées
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    return pts1, pts2







# ============================================================
# 3) Stabilisation complète
# ============================================================

def to_homog(M2x3: np.ndarray) -> np.ndarray:
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = M2x3
    return H

def rigid_from_angle(angle_rad: float, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Rigid 2x3 that maps src -> dst, given angle."""
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)
    t = np.mean(dst_pts - (src_pts @ R.T), axis=0)  # (2,)
    return np.hstack([R, t.reshape(2, 1)]).astype(np.float32)

def estimate_pair_transform(prev_gray, curr_gray,
                            max_features=3000, ratio=0.2,
                            down_fx=0.5,
                            ransac_thresh=2.5,
                            ransac_max_iters=3000,
                            rotation_zero_thresh_deg=2.0,
                            max_angle_deg=8.0):
    """
    Estime T = (curr -> prev)
    Downsample utilisé UNIQUEMENT pour estimer l'angle (plus stable), mais la translation
    est recalculée en pleine résolution (donc pas de scaling).
    """


    angle_hint = None
    # -------- (B) Inliers RANSAC sur pleine résolution --------
    curr_pts, prev_pts = match_sift_features(curr_gray, prev_gray, max_features=max_features, ratio=ratio)
    if curr_pts is None:
        return np.eye(2, 3, dtype=np.float32)

    M_r, inliers = cv2.estimateAffinePartial2D(
        curr_pts, prev_pts,                      # curr -> prev (IMPORTANT)
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=ransac_max_iters,
        confidence=0.995,
        refineIters=10
    )
    if M_r is None or inliers is None:
        return np.eye(2, 3, dtype=np.float32)

    mask = inliers.reshape(-1).astype(bool)
    src_in = curr_pts[mask]
    dst_in = prev_pts[mask]
    if len(src_in) < 15:
        return np.eye(2, 3, dtype=np.float32)

    # -------- (C) Construire la rigid transform finale --------
    if angle_hint is None:
        # fallback: SVD sur inliers
        M_ref = compute_rigid_transform_svd(src_in, dst_in, mask=None)
        angle = float(np.arctan2(M_ref[1, 0], M_ref[0, 0]))
    else:
        angle = float(angle_hint)

    angle_deg = abs(float(np.degrees(angle)))
    if angle_deg > max_angle_deg:
        return np.eye(2, 3, dtype=np.float32)

    # Si rotation très petite, forcer rotation=0 et estimer translation pure (stable)
    if angle_deg <= rotation_zero_thresh_deg:
        trans = dst_in - src_in
        mean = np.mean(trans, axis=0)
        return np.array([[1.0, 0.0, float(mean[0])],
                         [0.0, 1.0, float(mean[1])]], dtype=np.float32)

    # Sinon: angle + translation sur inliers pleine rés
    return rigid_from_angle(angle, src_in, dst_in)


def stabilize_video(
        input_path: str,
        output_path: str,
        smoothing_radius: int = 40,
        max_features: int = 3000,
        match_ratio: float = 0.2,
        crop_percent: float = 0.06,
        show_progress: bool = True,
        smooth_x: bool = False,          # <-- tu veux garder le mouvement horizontal, donc False
        border_replicate: bool = False,  # True si tu veux éviter le noir (au prix d'un "stretch")
):
    """
    Stabilise une vidéo en supprimant les mouvements verticaux et rotationnels.
    Garde le mouvement horizontal pour un effet de "suivi".

    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Info] Vidéo: {width}x{height}, {fps:.2f} fps, {n_frames} frames")

    # -------------------------
    # (1) Pairwise transforms: T_i = (frame i -> frame i-1)
    # -------------------------
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Impossible de lire la première frame")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    transforms = []
    idx = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        T = estimate_pair_transform(prev_gray, curr_gray,
                                    max_features=max_features,
                                    ratio=match_ratio)
        transforms.append(T)

        prev_gray = curr_gray
        idx += 1
        if show_progress and idx % 30 == 0:
            print(f"  Pairwise {idx}/{n_frames}")

    print(f"[1/3] {len(transforms)} transformations (curr->prev) calculées")

    # -------------------------
    # (2) Cumulative: C_i maps frame i -> frame 0
    # C_0 = I ; C_i = C_{i-1} @ T_i
    # -------------------------
    cumulative = [np.eye(3, dtype=np.float32)]
    for T in transforms:
        cumulative.append(cumulative[-1] @ to_homog(T))

    # Trajectoire horizontale du centre via inv(C)
    center = np.array([width / 2.0, height / 2.0, 1.0], dtype=np.float32)
    dx = []
    for C in cumulative:
        A = np.linalg.inv(C)
        p = A @ center
        dx.append(float(p[0] - center[0]))
    tx = np.array(dx, dtype=np.float32)
    tx = tx - tx[0]



    # Build stabilized transforms: M_i = shift_x(tx_i) @ C_i
    stabilized = []
    for C, x in zip(cumulative, tx):
        shift = np.array([[1.0, 0.0, float(x)],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
        M = (shift @ C).astype(np.float32)
        stabilized.append(M[:2, :])

    print("[2/3] Matrices stabilisées construites (rotation + vertical supprimés, horizontal conservé)")

    # -------------------------
    # (3) Warp + crop + write
    # -------------------------
    crop_x = int(width * crop_percent)
    crop_y = int(height * crop_percent)
    out_width = width - 2 * crop_x
    out_height = height - 2 * crop_y

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    if not out.isOpened():
        raise RuntimeError("Impossible d'ouvrir le writer (frameSize manquant ou codec)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    borderMode = cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT
    borderValue = (0, 0, 0)

    for i in range(len(stabilized)):
        ret, frame = cap.read()
        if not ret:
            break

        warped = cv2.warpAffine(
            frame,
            stabilized[i],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=borderValue
        )

        # crop (maintenant crop_x/crop_y existent, et out a la bonne taille)
        cropped = warped[crop_y:height - crop_y, crop_x:width - crop_x]
        out.write(cropped)

        if show_progress and (i + 1) % 30 == 0:
            print(f"  Write {i + 1}/{len(stabilized)}")

    cap.release()
    out.release()
    print(f"[3/3] Vidéo sauvegardée: {output_path}")







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

def warp_all(
    frames: List[np.ndarray],
    stabilized_2x3: List[np.ndarray],
    border_replicate: bool,
) -> Tuple[List[np.ndarray], int, int]:
    """
    Warp toutes les frames en taille FULL (w,h), sans aucun crop.
    => gros canvas, et les zones noires (haut/bas) restent visibles.
    """
    h, w = frames[0].shape[:2]

    borderMode = cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT
    borderValue = (0, 0, 0)

    warps: List[np.ndarray] = []
    for frame, A in zip(frames, stabilized_2x3):
        warped = cv2.warpAffine(
            frame,
            A,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=borderValue,
        )
        warps.append(warped)

    return warps, w, h



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

    crops, out_w, out_h = warp_all(frames, stabilized_2x3, border_replicate)

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
    input_path = "bad.mp4"  # Change to your video path
    out_dir = "panoramas_bad"
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
    d = np.median(np.diff(tx))  # déplacement typique
    move_right = (d > 0)  # True: gauche->droite, False: droite->gauchevf

    stabilized_2x3 = build_stabilized_affines(cumulative, tx)
    # 2x3 for warpAffine
    border_replicate= False
    crops, out_w, out_h = warp_all(frames, stabilized_2x3, border_replicate)

    # largeur strip (comme ex4: auto via dx) ou fixe
    strip_widths = compute_strip_widths(
        tx,
        use_auto=True,      # mets False si tu veux fixe
        fixed_width=8,
        max_strip=30
    )
    if not move_right:
        crops = crops[::-1]
        strip_widths = strip_widths[::-1]

    # 3) Views TRIÉES (monotones) : n_out_frames panoramas
    #    Exemple: du 20% au 80% de la largeur
    views = np.linspace(0.2, 0.8, int(n_out_frames)).astype(np.float32)
    views = np.clip(views, 0.0, 1.0)
    views = np.unique(views)
    views.sort()

    # 4) Construire la liste de PIL images
    panoramas = []
    for vf in views[:n_out_frames]:
        if not move_right:
            vf = 1.0 - vf
        pano_bgr = build_panorama_for_view(crops, view_frac=float(vf), strip_widths=strip_widths)

        # optionnel
        pano_bgr = autocrop_black(pano_bgr)

        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
        panoramas.append(Image.fromarray(pano_rgb))

    # 5) Assurer exactement n_out_frames (si unique a réduit)
    while len(panoramas) < n_out_frames:
        panoramas.append(panoramas[-1].copy())

    return panoramas

def _resize_and_pad_bgr(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    scale = min(out_w / w, out_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas
def detect_motion_direction(tx: np.ndarray) -> bool:
    if len(tx) < 3:
        return True
    d = float(np.median(np.diff(tx)))
    return d > 0
def generate_view_sweep_video_smooth(
frames: List[np.ndarray],
out_path: str,
view_start: float = 0.2,
view_end: float = 0.8,
n_video_frames: int = 180,
fps: int = 30,
out_width: int = 1280,
out_height: int = 720,
# panorama params
crop_percent: float = 0.0, # 0.0 => pas de crop (canvas full)
border_replicate: bool = True,
strip_width: int = 8,
auto_strip: bool = True,
max_strip: int = 30,
autocrop: bool = False,
auto_fix_direction: bool = True,
border_constant_video: bool = False,
) -> Path:
    """Crée une vidéo mp4 où le viewpoint bouge LISSEMENT de view_start -> view_end.

    - Lisse = easing cosinus (ease-in-out)
    - crop_percent=0.0 => gros canvas (pas de crop)
    - border_constant_video=True => zones noires visibles (hauteurs variables des strips)

    Sortie: out_path (mp4)
    """

    h, w = frames[0].shape[:2]

    # 1) Stabilisation (comme ton ex_4): cumulative + tx + affines
    cumulative = compute_cumulative(frames, show_progress=True)
    tx = compute_tx_from_cumulative(cumulative, w, h)
    stabilized_2x3 = build_stabilized_affines(cumulative, tx)

    # 2) Warp + crop (crop_percent=0 => full)
    # Si tu veux VOIR les variations haut/bas: border_constant_video=True
    border_rep = border_replicate and (not border_constant_video)
    crops, out_w, out_h = warp_all(frames, stabilized_2x3, border_rep)

    d = np.median(np.diff(tx))  # déplacement typique
    move_right = (d > 0)

    # Fix droite->gauche
    # if auto_fix_direction and (not move_right):
    #     crops = crops[::-1]
    #     tx = tx[::-1]

    strip_widths = compute_strip_widths(tx, use_auto=auto_strip, fixed_width=strip_width, max_strip=max_strip)

    # 3) Courbe de viewpoint lisse (ease-in-out)
    n_video_frames = max(2, int(n_video_frames))
    v0 = float(np.clip(view_start, 0.0, 1.0))
    v1 = float(np.clip(view_end, 0.0, 1.0))

    t = np.linspace(0.0, 1.0, n_video_frames, dtype=np.float32)
    ease = 0.5 - 0.5 * np.cos(np.pi * t) # 0->1 smooth
    views = (1.0 - ease) * v0 + ease * v1

    # 4) Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (int(out_width), int(out_height)))
    if not writer.isOpened():
        raise RuntimeError("Impossible d'ouvrir VideoWriter (codec/frameSize)")

    for i, vf in enumerate(views):
        vf_eff = float(vf)

        pano = build_panorama_for_view(crops, view_frac=vf_eff, strip_widths=strip_widths)
        if autocrop:
            pano = autocrop_black(pano)

        anchor_idx = len(strip_widths) // 2
        pano = center_panorama_on_anchor(pano, strip_widths, anchor_idx)

        frame_out = _resize_and_pad_bgr(pano, int(out_width), int(out_height))
        writer.write(frame_out)



    writer.release()
    return Path(out_path)

def center_panorama_on_anchor(pano: np.ndarray, strip_widths: List[int], anchor_idx: int) -> np.ndarray:
    """
    Renvoie un pano avec padding noir gauche/droite de sorte que le strip anchor_idx
    soit au centre horizontal.
    """
    H, W = pano.shape[:2]
    anchor_idx = int(np.clip(anchor_idx, 0, len(strip_widths) - 1))

    # position x du centre du strip anchor dans le panorama
    x_anchor = float(np.sum(strip_widths[:anchor_idx]) + 0.5 * strip_widths[anchor_idx])
    x_center = 0.5 * W
    shift = int(round(x_center - x_anchor))  # shift>0 => déplacer pano vers la droite

    pad = abs(shift)
    canvas = np.zeros((H, W + 2 * pad, 3), dtype=pano.dtype)

    # place pano à x = pad + shift (ça “recentre” le point d’ancrage)
    x_place = pad + shift
    canvas[:, x_place:x_place + W] = pano
    return canvas


if __name__ == "__main__":

     main()


