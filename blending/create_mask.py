"""
Face Alignment and Mask Creation Helper
Aligne deux visages et crée un masque pour le blending pyramidal
"""

import cv2
import mediapipe as mp
import numpy as np


# ============================================================================
# HYPERPARAMÈTRES - AJUSTER ICI POUR DIFFÉRENTS TYPES D'IMAGES
# ============================================================================

class BlendConfig:
    """Configuration pour l'alignement et le masque"""

    # --- ALIGNEMENT DES VISAGES ---
    PADDING_RATIO = 0.35  # Marge autour du visage (0.3 = 30%)
    SCALE_ADJUSTMENT = 1.0  # Ajustement de la taille (1.0 = taille identique, 1.1 = 10% plus grand)
    USE_PROCRUSTES = True  # Aligner orientation avec Procrustes (rotation + translation)

    # --- DÉFINITION DE LA ZONE À EXTRAIRE ---
    USE_INTERIOR_ONLY = True  # True = sans cheveux/oreilles, False = visage complet
    INCLUDE_FOREHEAD = True # Inclure le front dans le masque
    EXCLUDE_NECK = True  # Exclure le cou du masque

    # --- LISSAGE DU MASQUE ---
    MASK_DILATION = 15  # Taille de dilatation du masque (pixels)
    MASK_BLUR = 31  # Taille du flou gaussien pour adoucir les bords (impair)
    FEATHER_AMOUNT = 0.15  # Pourcentage de feathering sur les bords (0-1)

    # --- POINTS DE RÉFÉRENCE POUR L'ALIGNEMENT ---
    # Points clés pour l'alignement (yeux, nez, bouche)
    ALIGNMENT_LANDMARKS = [
        33, 133, 362, 263,  # Coins des yeux
        1, 4,  # Nez
        61, 291  # Coins de la bouche
    ]


# ============================================================================
# DÉFINITION DES ZONES DU VISAGE
# ============================================================================

mp_face = mp.solutions.face_mesh

# Contour ovale complet du visage
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Zone intérieure du visage (sans cheveux, oreilles)
FACE_INTERIOR_IDX = [
    # Front (partie supérieure)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    # Sourcils
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # Contour des yeux
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144,
    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380,
    # Nez
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13,
    # Bouche
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    # Joues (partie centrale uniquement)
    206, 203, 142, 126, 217, 174, 196, 419, 426, 423, 371, 355, 437, 399, 420,
    # Menton (sans le cou)
    175, 171, 140, 170, 169, 135, 138, 215, 177, 396, 395, 369, 400, 377, 152, 148, 176
]

# Zone du cou à exclure
NECK_IDX = [
    152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338
]

# Zone du front (partie haute)
FOREHEAD_EXTENSION = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    71, 68, 54, 103, 67, 109, 10
]


# ============================================================================
# FONCTIONS DE DÉTECTION ET ALIGNEMENT
# ============================================================================

def get_face_landmarks(img):
    """
    Détecte les landmarks du visage dans une image.

    Args:
        img: Image BGR

    Returns:
        numpy array (N, 2) des coordonnées (x, y) ou None
    """
    h, w = img.shape[:2]
    with mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
    ) as face_mesh:
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in range(len(lm))], dtype=np.float32)
    return pts


def procrustes_alignment(src_pts, tgt_pts):
    """
    Calcule la transformation Procrustes pour aligner src sur tgt.
    Utilise seulement translation, rotation et échelle uniforme.

    Args:
        src_pts: Points source (N, 2)
        tgt_pts: Points cible (N, 2)

    Returns:
        scale, rotation, translation
    """
    # Centrer les points
    src_mean = src_pts.mean(axis=0)
    tgt_mean = tgt_pts.mean(axis=0)
    src_centered = src_pts - src_mean
    tgt_centered = tgt_pts - tgt_mean

    # Calculer l'échelle
    src_scale = np.sqrt((src_centered ** 2).sum() / len(src_pts))
    tgt_scale = np.sqrt((tgt_centered ** 2).sum() / len(tgt_pts))
    scale = tgt_scale / src_scale

    # Normaliser pour trouver la rotation
    src_norm = src_centered / src_scale
    tgt_norm = tgt_centered / tgt_scale

    # SVD pour trouver la rotation optimale
    H = src_norm.T @ tgt_norm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Gérer la réflexion
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Calculer la translation
    translation = tgt_mean - scale * (R @ src_mean)

    return scale, R, translation


def apply_procrustes_transform(img, scale, R, translation):
    """
    Applique une transformation Procrustes à une image.

    Args:
        img: Image source
        scale: Facteur d'échelle
        R: Matrice de rotation (2x2)
        translation: Vecteur de translation (2,)

    Returns:
        Image transformée
    """
    h, w = img.shape[:2]

    # Créer la matrice de transformation affine
    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:, 2] = translation

    # Appliquer la transformation
    result = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return result


def align_faces(img_source, img_target, config=BlendConfig()):
    """
    Aligne deux images de visages pour qu'elles aient la même taille, position et orientation.

    Args:
        img_source: Image source (visage à transplanter)
        img_target: Image cible (arrière-plan de destination)
        config: Configuration d'alignement

    Returns:
        tuple: (img_source_aligned, img_target, transform_info)
    """

    # Détecter les landmarks
    landmarks_source = get_face_landmarks(img_source)
    landmarks_target = get_face_landmarks(img_target)

    if landmarks_source is None:
        raise RuntimeError("❌ Aucun visage détecté dans l'image source")
    if landmarks_target is None:
        raise RuntimeError("❌ Aucun visage détecté dans l'image cible")


    # Utiliser l'alignement Procrustes si activé
    if config.USE_PROCRUSTES:
        print("🔄 Alignement avec Procrustes (rotation + échelle)...")

        # Extraire les points clés pour l'alignement
        src_key_pts = landmarks_source[config.ALIGNMENT_LANDMARKS]
        tgt_key_pts = landmarks_target[config.ALIGNMENT_LANDMARKS]

        # Calculer la transformation Procrustes
        scale, R, translation = procrustes_alignment(src_key_pts, tgt_key_pts)

        # Ajuster l'échelle si demandé
        scale *= config.SCALE_ADJUSTMENT

        # Créer une image de la taille de la cible
        h_tgt, w_tgt = img_target.shape[:2]

        # Appliquer la transformation à l'image source
        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = scale * R
        M[:, 2] = translation

        img_source_aligned = cv2.warpAffine(
            img_source, M, (w_tgt, h_tgt),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        transform_info = {'scale': scale, 'rotation': R, 'translation': translation}
        print(f"✓ Alignement terminé (échelle: {scale:.2f})")

    else:
        # Alignement simple par bounding box
        print("📐 Alignement par bounding box...")

        # Calculer les bounding boxes
        face_pts_src = landmarks_source[FACE_OVAL_IDX]
        face_pts_tgt = landmarks_target[FACE_OVAL_IDX]

        x1_src, y1_src = face_pts_src.min(axis=0)
        x2_src, y2_src = face_pts_src.max(axis=0)
        x1_tgt, y1_tgt = face_pts_tgt.min(axis=0)
        x2_tgt, y2_tgt = face_pts_tgt.max(axis=0)

        # Ajouter du padding
        w_src, h_src = x2_src - x1_src, y2_src - y1_src
        w_tgt, h_tgt = x2_tgt - x1_tgt, y2_tgt - y1_tgt

        pad_src = max(w_src, h_src) * config.PADDING_RATIO
        pad_tgt = max(w_tgt, h_tgt) * config.PADDING_RATIO

        x1_src -= pad_src;
        y1_src -= pad_src
        x2_src += pad_src;
        y2_src += pad_src
        x1_tgt -= pad_tgt;
        y1_tgt -= pad_tgt
        x2_tgt += pad_tgt;
        y2_tgt += pad_tgt

        # Calculer l'échelle et le décalage
        size_src = max(x2_src - x1_src, y2_src - y1_src)
        size_tgt = max(x2_tgt - x1_tgt, y2_tgt - y1_tgt)
        scale = (size_tgt / size_src) * config.SCALE_ADJUSTMENT

        center_src = np.array([(x1_src + x2_src) / 2, (y1_src + y2_src) / 2])
        center_tgt = np.array([(x1_tgt + x2_tgt) / 2, (y1_tgt + y2_tgt) / 2])

        # Redimensionner et repositionner
        h_src_orig, w_src_orig = img_source.shape[:2]
        new_w, new_h = int(w_src_orig * scale), int(h_src_orig * scale)
        img_source_scaled = cv2.resize(img_source, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        center_src_scaled = center_src * scale
        offset = center_tgt - center_src_scaled

        # Créer l'image alignée
        h_tgt_img, w_tgt_img = img_target.shape[:2]
        img_source_aligned = np.zeros((h_tgt_img, w_tgt_img, 3), dtype=np.uint8)

        x_offset, y_offset = int(offset[0]), int(offset[1])

        src_x1 = max(0, -x_offset)
        src_y1 = max(0, -y_offset)
        src_x2 = min(new_w, w_tgt_img - x_offset)
        src_y2 = min(new_h, h_tgt_img - y_offset)

        dst_x1 = max(0, x_offset)
        dst_y1 = max(0, y_offset)
        dst_x2 = min(w_tgt_img, x_offset + new_w)
        dst_y2 = min(h_tgt_img, y_offset + new_h)

        if src_x2 > src_x1 and src_y2 > src_y1:
            img_source_aligned[dst_y1:dst_y2, dst_x1:dst_x2] = \
                img_source_scaled[src_y1:src_y2, src_x1:src_x2]

        transform_info = {'scale': scale, 'offset': (x_offset, y_offset)}

    return img_source_aligned, img_target, transform_info


# ============================================================================
# CRÉATION DU MASQUE
# ============================================================================

def build_face_mask(img, config=BlendConfig()):
    """
    Construit un masque lisse pour le visage avec feathering.

    Args:
        img: Image d'entrée (BGR)
        config: Configuration du masque

    Returns:
        Masque float32 (0-1), où 0 = utiliser source, 1 = utiliser target
    """
    h, w = img.shape[:2]

    # Détecter les landmarks
    landmarks = get_face_landmarks(img)
    if landmarks is None:
        raise RuntimeError("❌ Aucun visage détecté pour créer le masque")

    # Choisir les indices selon la configuration
    if config.USE_INTERIOR_ONLY:
        print("🎭 Création du masque pour l'INTÉRIEUR du visage...")
        mask_indices = FACE_INTERIOR_IDX.copy()

        # Étendre avec le front si demandé
        if config.INCLUDE_FOREHEAD:
            mask_indices.extend(FOREHEAD_EXTENSION)
            print("  + Front inclus")

        # Exclure le cou si demandé
        if config.EXCLUDE_NECK:
            print("  + Cou exclu")
    else:
        print("🎭 Création du masque pour le visage COMPLET...")
        mask_indices = FACE_OVAL_IDX

    # Extraire les points et créer le masque de base
    pts = landmarks[mask_indices].astype(np.int32)

    # Créer un masque binaire
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Dilatation pour étendre légèrement le masque
    if config.MASK_DILATION > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (config.MASK_DILATION, config.MASK_DILATION))
        mask = cv2.dilate(mask, kernel)

    # Appliquer un flou gaussien pour lisser les bords
    if config.MASK_BLUR > 0:
        blur_size = config.MASK_BLUR if config.MASK_BLUR % 2 == 1 else config.MASK_BLUR + 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # Normaliser en float [0, 1]
    mask = mask.astype(np.float32) / 255.0

    # Appliquer un feathering sur les bords
    if config.FEATHER_AMOUNT > 0:
        # Distance transform pour créer un gradient sur les bords
        mask_binary = (mask > 0.5).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)

        # Normaliser la distance
        max_dist = dist_transform.max()
        if max_dist > 0:
            feather_width = max_dist * config.FEATHER_AMOUNT
            dist_norm = np.clip(dist_transform / feather_width, 0, 1)

            # Appliquer une courbe sigmoïde pour un transition douce
            mask = 1 / (1 + np.exp(-10 * (dist_norm - 0.5)))

    # Inverser le masque (0 = source, 1 = target)
    mask = 1 - mask

    print(f"✓ Masque créé (feathering: {config.FEATHER_AMOUNT:.0%})")

    return mask


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def prepare_images_for_blending(img_source_path, img_target_path, config=None):
    """
    Prépare deux images pour le blending pyramidal.

    Cette fonction:
    1. Charge les images
    2. Aligne les visages (taille, position, rotation)
    3. Crée un masque lisse pour le blending

    Args:
        img_source_path: Chemin vers l'image source (visage à transplanter)
        img_target_path: Chemin vers l'image cible (arrière-plan)
        config: Configuration (BlendConfig) ou None pour défaut

    Returns:
        tuple: (img_source_aligned, img_target, mask)
            - img_source_aligned: Image source alignée
            - img_target: Image cible
            - mask: Masque float32 (0-1) pour le blending
    """
    if config is None:
        config = BlendConfig()



    # Charger les images
    img_source = cv2.imread(img_source_path)
    if img_source is None:
        raise FileNotFoundError(f"❌ Impossible de charger: {img_source_path}")

    img_target = cv2.imread(img_target_path)
    if img_target is None:
        raise FileNotFoundError(f"❌ Impossible de charger: {img_target_path}")



    # Aligner les visages
    img_source_aligned, img_target, transform_info = align_faces(
        img_source, img_target, config
    )

    # Créer le masque sur l'image source alignée
    print()
    mask = build_face_mask(img_source_aligned, config)



    return img_source_aligned, img_target, mask


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Configuration personnalisée
    config = BlendConfig()
    config.USE_PROCRUSTES = True  # Aligner avec rotation
    config.SCALE_ADJUSTMENT = 1.0  # Taille identique
    config.USE_INTERIOR_ONLY = True  # Seulement l'intérieur du visage
    config.INCLUDE_FOREHEAD = False  # Garder le front
    config.EXCLUDE_NECK = True  # Pas de cou
    config.MASK_DILATION = 15  # Dilatation modérée
    config.MASK_BLUR = 31  # Flou important pour transition douce
    config.FEATHER_AMOUNT = 0.2  # Feathering sur 20% des bords

    try:
        # Préparer les images
        img_source_aligned, img_target, mask = prepare_images_for_blending(
            'images/source.jpg',
            'images/target.jpg',
            config=config
        )

        # Sauvegarder les résultats
        print("💾 Sauvegarde des résultats...")
        cv2.imwrite('images/source_aligned.jpg', img_source_aligned)
        cv2.imwrite('images/target_processed.jpg', img_target)
        cv2.imwrite('images/mask.png', (mask * 255).astype(np.uint8))

        # Visualisation du masque en couleur
        mask_viz = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        mask_viz[:, :, 2] = 255 - mask_viz[:, :, 2]  # Rouge = zone source
        cv2.imwrite('images/mask_visualization.png', mask_viz)

        # Aperçu rapide (sans blending)
        preview = (img_source_aligned * (1 - mask[:, :, None]) +
                   img_target * mask[:, :, None]).astype(np.uint8)
        cv2.imwrite('images/preview_simple.jpg', preview)



    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback

        traceback.print_exc()