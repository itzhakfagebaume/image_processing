import numpy as np
import cv2
from typing import List, Tuple
from create_mask import *
import matplotlib.pyplot as plt


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Create a Gaussian kernel for smoothing.

    Args:
        size: Kernel size (should be odd)
        sigma: Standard deviation of Gaussian

    Returns:
        2D Gaussian kernel normalized to sum to 1
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))

    return kernel / np.sum(kernel)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply 2D convolution to an image.

    Args:
        image: Input image (can be multi-channel)
        kernel: Convolution kernel

    Returns:
        Convolved image with same dimensions as input
    """
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2

    if len(image.shape) == 3:
        height, width, channels = image.shape
        result = np.zeros_like(image)

        for c in range(channels):
            # Pad image to handle borders
            padded = np.pad(image[:, :, c], ((pad_h, pad_h), (pad_w, pad_w)),
                            mode='reflect')

            # Perform convolution
            for i in range(height):
                for j in range(width):
                    region = padded[i:i + k_height, j:j + k_width]
                    result[i, j, c] = np.sum(region * kernel)
    else:
        height, width = image.shape
        result = np.zeros_like(image)
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        for i in range(height):
            for j in range(width):
                region = padded[i:i + k_height, j:j + k_width]
                result[i, j] = np.sum(region * kernel)

    return result


def reduce_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Reduce image size by smoothing and downsampling (Gaussian pyramid level down).

    Args:
        image: Input image
        kernel: Gaussian kernel for smoothing

    Returns:
        Image reduced to half size in each dimension
    """
    smoothed = convolve2d(image, kernel)

    return smoothed[::2, ::2]


def expand_image(image: np.ndarray, kernel: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Expand image size by upsampling and smoothing (Gaussian pyramid level up).

    Args:
        image: Input image to expand
        kernel: Gaussian kernel for smoothing
        target_shape: Desired output shape (height, width)

    Returns:
        Expanded image
    """
    if len(image.shape) == 3:
        expanded = np.zeros((target_shape[0], target_shape[1], image.shape[2]),
                            dtype=image.dtype)
    else:
        expanded = np.zeros(target_shape, dtype=image.dtype)

    expanded[::2, ::2] = image

    expanded = expanded * 4.0

    return convolve2d(expanded, kernel)


def build_gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Build a Gaussian pyramid by repeatedly smoothing and downsampling.

    Args:
        image: Input image
        levels: Number of pyramid levels

    Returns:
        List of images from fine to coarse (level 0 is original)
    """
    kernel = gaussian_kernel(5, 1.0)
    pyramid = [image.astype(np.float32)]

    for i in range(levels - 1):
        reduced = reduce_image(pyramid[-1], kernel)
        pyramid.append(reduced)

    return pyramid


def build_laplacian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Build a Laplacian pyramid by storing differences between Gaussian levels.
    The Laplacian pyramid captures details at each scale.

    Args:
        image: Input image
        levels: Number of pyramid levels

    Returns:
        List of Laplacian images (band-pass filtered)
    """
    gaussian_pyr = build_gaussian_pyramid(image, levels)
    kernel = gaussian_kernel(5, 1.0)

    laplacian_pyr = []

    for i in range(levels - 1):
        current = gaussian_pyr[i]

        next_expanded = expand_image(gaussian_pyr[i + 1], kernel,
                                     (current.shape[0], current.shape[1]))

        laplacian = current - next_expanded
        laplacian_pyr.append(laplacian)

    laplacian_pyr.append(gaussian_pyr[-1])

    return laplacian_pyr


def collapse_laplacian_pyramid(laplacian_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Reconstruct image from Laplacian pyramid by expanding and summing.

    Args:
        laplacian_pyr: List of Laplacian pyramid levels

    Returns:
        Reconstructed image
    """
    kernel = gaussian_kernel(5, 1.0)

    result = laplacian_pyr[-1]

    for i in range(len(laplacian_pyr) - 2, -1, -1):
        result = expand_image(result, kernel,
                              (laplacian_pyr[i].shape[0], laplacian_pyr[i].shape[1]))

        result = result + laplacian_pyr[i]

    return result


def pyramid_blend(image_a: np.ndarray, image_b: np.ndarray,
                  mask: np.ndarray, levels: int = 6) -> np.ndarray:
    """
    Blend two images using pyramid blending for seamless composition.

    The algorithm:
    1. Build Laplacian pyramids L_a and L_b for images A and B
    2. Build Gaussian pyramid G_m for the mask M
    3. Blend at each level: L_c(i,j) = G_m(i,j)*L_a(i,j) + (1-G_m(i,j))*L_b(i,j)
    4. Reconstruct the final image by collapsing the blended pyramid

    Args:
        image_a: First input image (numpy array)
        image_b: Second input image (same size as image_a)
        mask: Binary mask (1 for image_a, 0 for image_b)
        levels: Number of pyramid levels

    Returns:
        Blended image
    """
    image_a = image_a.astype(np.float32)
    image_b = image_b.astype(np.float32)
    mask = mask.astype(np.float32)

    if mask.max() > 1.0:
        mask = mask / 255.0

    laplacian_a = build_laplacian_pyramid(image_a, levels)

    laplacian_b = build_laplacian_pyramid(image_b, levels)

    gaussian_mask = build_gaussian_pyramid(mask, levels)

    blended_pyramid = []

    for i in range(levels):
        mask_level = gaussian_mask[i]

        if len(laplacian_a[i].shape) == 3:
            mask_level = np.expand_dims(mask_level, axis=2)

        # Blend: L_c = G_m * L_a + (1 - G_m) * L_b
        blended_level = (mask_level * laplacian_a[i] +
                         (1 - mask_level) * laplacian_b[i])
        blended_pyramid.append(blended_level)

    # Reconstruct the final image from the blended pyramid
    result = collapse_laplacian_pyramid(blended_pyramid)

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result





def hybrid_blend(image_a: np.ndarray, image_b: np.ndarray,
                 levels: int = 7) -> np.ndarray:
    """
    Create a hybrid image by blending low frequencies from image_a
    with high frequencies from image_b.

    Args:
        image_a: First input image (low frequencies visible from distance)
        image_b: Second input image (high frequencies visible up close)
        levels: Number of pyramid levels

    Returns:
        Hybrid image
    """
    image_a = image_a.astype(np.float32)
    image_b = image_b.astype(np.float32)

    # Build Laplacian pyramids for both images
    laplacian_a = build_laplacian_pyramid(image_a, levels)
    laplacian_b = build_laplacian_pyramid(image_b, levels)

    # Blend pyramids: low frequencies from A, high frequencies from B
    hybrid_pyramid = []

    for i in range(levels):
        if i < 2:
            # Lower levels (coarser) - use image_a (low frequencies)
            hybrid_level = laplacian_a[i]
        else:
            # Higher levels (finer) - use image_b (high frequencies)
            hybrid_level = laplacian_b[i]

        hybrid_pyramid.append(hybrid_level)

    result = collapse_laplacian_pyramid(hybrid_pyramid)

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result



def task1():

    img_source_aligned, img_target, mask_img = prepare_images_for_blending(
        'images/bibi.png',
        'images/beny.png',
    )


    blended = pyramid_blend(img_source_aligned, img_target, mask_img)


    cv2.imwrite("images/blended.png", blended)
    # cv2.imwrite("images/mask.png", mask_img*255)



def task2():
    # img_source, img_target, mask_img = prepare_images_for_blending(
    #     'images/margot.png',
    #     'images/peche.png',
    # )

    img_source = cv2.imread("images/bibi.png")

    img_target = cv2.imread("images/amir.png")


    if img_source.shape != img_target.shape:
        img_target = cv2.resize(img_target, (img_source.shape[1], img_source.shape[0]))
    print("Creating hybrid image...")

    blended = hybrid_blend(img_source, img_target)

    cv2.imwrite("images/hybrid_image_peche2.png", blended)

def visualize_gaussian_pyramid(image: np.ndarray, levels: int = 5) -> None:
    """
    Visualize Gaussian pyramid levels by displaying them side-by-side.

    Args:
        image: Input image
        levels: Number of pyramid levels
    """
    pyramid = build_gaussian_pyramid(image, levels)

    fig, axes = plt.subplots(1, levels, figsize=(15, 3))

    for i, level in enumerate(pyramid):
        display_img = np.clip(level, 0, 255).astype(np.uint8)
        axes[i].imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Level {i}\n{display_img.shape}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("images/gaussian_pyramid_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()


def visualize_laplacian_pyramid(image: np.ndarray, levels: int = 5) -> None:
    """
    Visualize Laplacian pyramid levels by displaying them side-by-side.

    Args:
        image: Input image
        levels: Number of pyramid levels
    """
    pyramid = build_laplacian_pyramid(image, levels)

    fig, axes = plt.subplots(1, levels, figsize=(15, 3))

    for i, level in enumerate(pyramid):
        display_img = np.clip((level + 128), 0, 255).astype(np.uint8)
        axes[i].imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Level {i}\n{display_img.shape}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("images/laplacian_pyramid_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Normalise une image (Laplacien) pour l'affichage avec imshow.
    On remappe les valeurs dans [0,1] pour que le contraste soit visible.
    """
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img[:] = 0.5  # image constante: gris moyen
    return img


def visualize_blend_levels(levels_to_show=(0, 2,4, 6), levels=6):
    """
    Affiche, pour quelques niveaux de la pyramide :
    - Laplacien de l'image A
    - Laplacien de l'image B
    - Masque à ce niveau
    - Laplacien blendé à ce niveau
    """


    img_source_aligned, img_target, mask_img = prepare_images_for_blending(
        'images/bibi.png',
        'images/trump.png',
    )

    img_source_aligned = img_source_aligned.astype(np.float32)
    img_target = img_target.astype(np.float32)
    mask = mask_img.astype(np.float32)


    if mask.max() > 1.0:
        mask = mask / 255.0

    # 2) Construire les pyramides
    laplacian_a = build_laplacian_pyramid(img_source_aligned, levels)
    laplacian_b = build_laplacian_pyramid(img_target, levels)
    gaussian_mask = build_gaussian_pyramid(mask, levels)

    # 3) Visualiser pour les niveaux demandés
    for lvl in levels_to_show:
        if lvl < 0 or lvl >= levels:
            continue  # ignorer si niveau invalide

        L_a = laplacian_a[lvl]
        L_b = laplacian_b[lvl]
        M = gaussian_mask[lvl]

        # Si images couleur, étendre la dimension du masque
        if L_a.ndim == 3:
            M_expanded = np.expand_dims(M, axis=2)
        else:
            M_expanded = M

        # Laplacien blendé à ce niveau
        L_blend = M_expanded * L_a + (1.0 - M_expanded) * L_b

        # Normalisation pour affichage
        disp_L_a = normalize_for_display(L_a)
        disp_L_b = normalize_for_display(L_b)
        disp_L_blend = normalize_for_display(L_blend)
        disp_M = normalize_for_display(M)

        # Figure : 1 ligne, 4 colonnes
        plt.figure(figsize=(12, 3))
        plt.suptitle(f"Pyramid blending – level {lvl}", fontsize=14)

        # Laplacian A
        plt.subplot(1, 3, 1)
        if disp_L_a.ndim == 2:
            plt.imshow(disp_L_a, cmap='gray')
        else:
            plt.imshow(disp_L_a)
        plt.title("Laplacian A")
        plt.axis('off')

        # Laplacian B
        plt.subplot(1, 3, 2)
        if disp_L_b.ndim == 2:
            plt.imshow(disp_L_b, cmap='gray')
        else:
            plt.imshow(disp_L_b)
        plt.title("Laplacian B")
        plt.axis('off')



        # Blended Laplacian
        plt.subplot(1, 3, 3)
        if disp_L_blend.ndim == 2:
            plt.imshow(disp_L_blend, cmap='gray')
        else:
            plt.imshow(disp_L_blend)
        plt.title("Blended Laplacian")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def visualize_gaussian_levels(levels_to_show=(0, 2, 4), levels=6):
    """
    Affiche, pour quelques niveaux de la pyramide gaussienne :
    - Gaussian de l'image A
    - Gaussian de l'image B
    - Gaussian du masque
    """

    # 1) Charger / préparer les images et le masque
    img_source_aligned, img_target, mask_img = prepare_images_for_blending(
        'images/bibi.png',
        'images/trump.png',
    )

    img_source_aligned = img_source_aligned.astype(np.float32)
    img_target = img_target.astype(np.float32)
    mask = mask_img.astype(np.float32)

    # Normaliser le masque dans [0,1] si besoin
    if mask.max() > 1.0:
        mask = mask / 255.0

    # 2) Construire les pyramides gaussiennes
    gaussian_a = build_gaussian_pyramid(img_source_aligned, levels)
    gaussian_b = build_gaussian_pyramid(img_target, levels)
    gaussian_mask = build_gaussian_pyramid(mask, levels)

    # 3) Visualiser pour les niveaux demandés
    for lvl in levels_to_show:
        if lvl < 0 or lvl >= levels:
            continue

        G_a = gaussian_a[lvl]
        G_b = gaussian_b[lvl]
        M = gaussian_mask[lvl]

        disp_G_a = normalize_for_display(G_a)
        disp_G_b = normalize_for_display(G_b)
        disp_M = normalize_for_display(M)

        plt.figure(figsize=(12, 3))
        plt.suptitle(f"Gaussian pyramids – level {lvl}", fontsize=14)

        # Gaussian A
        plt.subplot(1, 2, 1)
        if disp_G_a.ndim == 2:
            plt.imshow(disp_G_a, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(np.clip(disp_G_a * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title("Gaussian A")
        plt.axis('off')

        # Gaussian B
        plt.subplot(1, 2, 2)
        if disp_G_b.ndim == 2:
            plt.imshow(disp_G_b, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(np.clip(disp_G_b * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title("Gaussian B")
        plt.axis('off')



        plt.tight_layout()
        plt.show()


def compute_fft_magnitude(image_gray: np.ndarray) -> np.ndarray:
    """
    Calcule le spectre de magnitude 2D (centré) d'une image en niveaux de gris.
    On renvoie log(1 + |FFT|) pour mieux visualiser les fréquences.
    """
    # Transformée de Fourier 2D
    f = np.fft.fft2(image_gray.astype(np.float32))
    # Shift pour mettre les basses fréquences au centre
    fshift = np.fft.fftshift(f)
    # Magnitude
    magnitude = np.abs(fshift)
    # Échelle log pour visualiser (sinon tout est saturé par les basses fréquences)
    magnitude_log = np.log(1 + magnitude)
    return magnitude_log


def visualize_fft_blends(path_good: str, path_bad: str):
    """
    Visualise et compare les spectres de Fourier 2D (magnitude) du
    bon blend et du mauvais blend.
    """

    # Charger les images en couleur
    img_good = cv2.imread(path_good)
    img_bad = cv2.imread(path_bad)

    if img_good is None or img_bad is None:
        raise FileNotFoundError("Impossible de charger les images de blend. Vérifie les chemins.")

    # Convertir en niveaux de gris pour simplifier l'analyse de fréquence
    img_good_gray = cv2.cvtColor(img_good, cv2.COLOR_BGR2GRAY)
    img_bad_gray = cv2.cvtColor(img_bad, cv2.COLOR_BGR2GRAY)

    # Calculer les spectres de magnitude
    mag_good = compute_fft_magnitude(img_good_gray)
    mag_bad = compute_fft_magnitude(img_bad_gray)

    # Normalisation pour affichage (optionnel mais plus propre)
    def normalize(img):
        img = img.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img[:] = 0.5
        return img

    mag_good_disp = normalize(mag_good)
    mag_bad_disp = normalize(mag_bad)

    # Affichage côte à côte
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(mag_good_disp, cmap='gray')
    plt.title("Good blend – FFT magnitude")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mag_bad_disp, cmap='gray')
    plt.title("Bad blend – FFT magnitude")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR or RGB image to grayscale.

    Args:
        image: image couleur (numpy array, shape H×W×3)

    Returns:
        image en niveaux de gris (numpy array, shape H×W)
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray









"""
Face Alignment and Mask Creation Helper coded with gpt 
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








if __name__ == "__main__":


   task1()
   task2()
   # visualize_blend_levels(levels_to_show=(0, 2, 4), levels=6)
   # visualize_gaussian_levels(levels_to_show=(0, 2, 4), levels=6)
   # visualize_fft_blends("images/blended_result5.png", "images/blended_result4.png")
   # img = cv2.imread("images/hybrid_image.png")
   # gray = to_grayscale(img)
   # cv2.imwrite("images/blend_good_gray.png", gray)


   # Visualize pyramids
   # img = cv2.imread("images/bibi.png")
   # if img is not None:
   #     visualize_gaussian_pyramid(img, levels=4)
   #     visualize_laplacian_pyramid(img, levels=4)