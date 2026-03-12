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



# ============================================================
# 4) MAIN
# ============================================================

if __name__ == "__main__":
    input_video = "video/Kessaria.mp4"
    output_video = "boat_stabilized.mp4"

    stabilize_video(
        input_path=input_video,
        output_path=output_video,
        smoothing_radius=40,  # 40 = lissage fort pour éliminer oscillations verticales
        max_features=3000,  # Plus de features = meilleure détection verticale
        match_ratio=0.2,  # Plus strict = moins de faux matches = moins de bruit vertical
        crop_percent=0.06  # 6% de crop pour compenser les corrections
    )

