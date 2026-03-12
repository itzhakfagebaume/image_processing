from __future__ import annotations
import numpy as np

from pathlib import Path
from typing import List, Optional, Tuple
import os, glob

import cv2
import numpy as np
from PIL import Image


_USE_SIFT = hasattr(cv2, "SIFT_create")
if _USE_SIFT:
    _FE = cv2.SIFT_create(nfeatures=2000)          # ↓ moins de features = plus rapide
    _MATCHER = cv2.BFMatcher()





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



def match_features_downsample(gray_curr, gray_prev, down_fx=1.0, ratio=0.75, max_keep=400):
    """
    Matching rapide (SIFT ou ORB selon dispo) sur images downsampled.
    Retourne pts_curr, pts_prev en COORDONNÉES PLEINE RÉSOLUTION.
    """
    if down_fx < 1.0:
        curr_small = cv2.resize(gray_curr, None, fx=down_fx, fy=down_fx, interpolation=cv2.INTER_AREA)
        prev_small = cv2.resize(gray_prev, None, fx=down_fx, fy=down_fx, interpolation=cv2.INTER_AREA)
    else:
        curr_small, prev_small = gray_curr, gray_prev

    kp1, des1 = _FE.detectAndCompute(curr_small, None)  # curr
    kp2, des2 = _FE.detectAndCompute(prev_small, None)  # prev

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, None

    knn = _MATCHER.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        return None, None

    # garder seulement les meilleurs (évite gros tri)
    if len(good) > max_keep:
        good = sorted(good, key=lambda x: x.distance)[:max_keep]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])  # curr_small
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])  # prev_small

    # rescale vers pleine résolution
    if down_fx < 1.0:
        scale = 1.0 / down_fx
        pts1 *= scale
        pts2 *= scale

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
    curr_pts, prev_pts = match_features_downsample(curr_gray, prev_gray,  ratio=ratio)
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





# def estimate_pair_transform(prev_gray, curr_gray,
#                             down_fx=0.5,
#                             ratio=0.75,
#                             ransac_thresh=3.0,
#                             ransac_max_iters=800,
#                             rotation_zero_thresh_deg=2.0,
#                             max_angle_deg=10.0):
#     """
#     Estime T = (curr -> prev) en rigide (rotation+translation).
#     - matching sur downsample
#     - RANSAC moins lourd
#     """
#     curr_pts, prev_pts = match_features_downsample(curr_gray, prev_gray,
#                                                    down_fx=down_fx, ratio=ratio)
#
#     if curr_pts is None:
#         return np.eye(2, 3, dtype=np.float32)
#
#     M_r, inliers = cv2.estimateAffinePartial2D(
#         curr_pts, prev_pts,  # curr -> prev
#         method=cv2.RANSAC,
#         ransacReprojThreshold=ransac_thresh,
#         maxIters=ransac_max_iters,
#         confidence=0.99,
#         refineIters=5
#     )
#     if M_r is None or inliers is None:
#         return np.eye(2, 3, dtype=np.float32)
#
#     mask = inliers.reshape(-1).astype(bool)
#     src_in = curr_pts[mask]
#     dst_in = prev_pts[mask]
#     if len(src_in) < 10:
#         return np.eye(2, 3, dtype=np.float32)
#
#     # rigid via SVD (Kabsch)
#     M_ref = compute_rigid_transform_svd(src_in, dst_in, mask=None)
#     angle = float(np.arctan2(M_ref[1, 0], M_ref[0, 0]))
#     angle_deg = abs(float(np.degrees(angle)))
#
#     if angle_deg > max_angle_deg:
#         return np.eye(2, 3, dtype=np.float32)
#
#     if angle_deg <= rotation_zero_thresh_deg:
#         trans = np.mean(dst_in - src_in, axis=0)
#         return np.array([[1.0, 0.0, float(trans[0])],
#                          [0.0, 1.0, float(trans[1])]], dtype=np.float32)
#
#     return rigid_from_angle(angle, src_in, dst_in)
#


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
    move_right = (tx[-1] - tx[0]) > 0

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
    if move_right:
        crops = crops[::-1]
        tx = tx[::-1]

    # 3) Views TRIÉES (monotones) : n_out_frames panoramas
    #    Exemple: du 20% au 80% de la largeur
    views = np.linspace(0.2, 0.8, int(n_out_frames)).astype(np.float32)
    views = np.clip(views, 0.0, 1.0)
    views = np.unique(views)
    views.sort()

    # 4) Construire la liste de PIL images
    panoramas = []
    for vf in views[:n_out_frames]:
        if move_right:
            vf = 1.0 - vf
        pano_bgr = build_panorama_for_view(crops, view_frac=float(vf), strip_widths=strip_widths)



        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
        panoramas.append(Image.fromarray(pano_rgb))

    # 5) Assurer exactement n_out_frames (si unique a réduit)
    while len(panoramas) < n_out_frames:
        panoramas.append(panoramas[-1].copy())

    return panoramas



if __name__ == "__main__":

    panoramas = generate_panorama("frames_test", 10)
    for i, pano in enumerate(panoramas):
        pano.save(f"panorama_{i:02d}.png")

