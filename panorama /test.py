import numpy as np
import cv2
import pytest

# >>> adapte le module ici
from ex4 import warp_image, warp_image_rotation, lucas_kanade_pyramid


# -------------------------
# Helpers
# -------------------------
def make_synthetic_rgb(h=240, w=320):
    """
    Image synthétique riche en gradients + coins => stable pour LK.
    Déterministe (pas de random).
    """
    y, x = np.mgrid[0:h, 0:w]

    base = (
        127
        + 40 * np.sin(2 * np.pi * x / 23.0)
        + 35 * np.cos(2 * np.pi * y / 31.0)
        + 25 * np.sin(2 * np.pi * (x + y) / 47.0)
        + 15 * np.sin(2 * np.pi * (x - 2 * y) / 61.0)
    )
    gray = np.clip(base, 0, 255).astype(np.uint8)
    rgb = np.dstack([gray, gray, gray]).copy()

    # Formes avec coins / contrastes
    cv2.rectangle(rgb, (60, 50), (95, 85), (255, 255, 255), -1)
    cv2.rectangle(rgb, (200, 140), (235, 175), (30, 30, 30), -1)
    cv2.circle(rgb, (160, 80), 18, (220, 220, 220), -1)
    cv2.line(rgb, (30, 200), (290, 210), (180, 180, 180), 2)

    return rgb


def rgb_to_gray_float(img_rgb):
    return (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]).astype(np.float64)


def make_mask(h, w, border):
    mask = np.ones((h, w), dtype=bool)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False
    return mask


def mse(a, b, mask=None):
    d = (a - b)
    if mask is not None:
        d = d[mask]
    return float(np.mean(d * d))







# -----------------------------------------
# 2) LK avec rotation : récupération des params
# -----------------------------------------
def test_lk_recovers_translation_only_theta_near_zero():
    img1 = make_synthetic_rgb()
    dx_gt, dy_gt, theta_gt = 7.0, 4.0, 0.0

    img2 = warp_image_rotation(img1, dx_gt, dy_gt, theta_gt)

    dx, dy = lucas_kanade_pyramid(
        img1, img2,
        num_levels=4,
        num_iterations=40,
        border=8,
        convergence_threshold=1e-3,
        use_rotation=False
    )
    theta =0

    assert dx == pytest.approx(dx_gt, abs=0.5)
    assert dy == pytest.approx(dy_gt, abs=0.5)
    assert theta == pytest.approx(0.0, abs=0.02)  # rad ~ 1.1°


def test_lk_recovers_rotation_only():
    img1 = make_synthetic_rgb()
    dx_gt, dy_gt, theta_gt = 0.0, 0.0, 0.06  # ~3.4°

    img2 = warp_image_rotation(img1, dx_gt, dy_gt, theta_gt)

    dx, dy, theta = lucas_kanade_pyramid(
        img1, img2,
        num_levels=4,
        num_iterations=50,
        border=8,
        convergence_threshold=1e-3,
        use_rotation=True
    )

    assert dx == pytest.approx(0.0, abs=0.8)
    assert dy == pytest.approx(0.0, abs=0.8)
    assert theta == pytest.approx(theta_gt, abs=0.02)


@pytest.mark.parametrize("dx_gt,dy_gt,theta_gt", [
    (5.0, 3.0, 0.03),   # ~1.7°
    (-6.0, 2.5, -0.04), # ~-2.3°
    (4.0, -5.0, 0.05),  # ~2.9°
])
def test_lk_recovers_rotation_and_translation(dx_gt, dy_gt, theta_gt):
    img1 = make_synthetic_rgb()
    img2 = warp_image_rotation(img1, dx_gt, dy_gt, theta_gt)

    dx, dy, theta = lucas_kanade_pyramid(
        img1, img2,
        num_levels=4,
        num_iterations=60,
        border=8,
        convergence_threshold=1e-3,
        use_rotation=True
    )

    assert dx == pytest.approx(dx_gt, abs=0.7)
    assert dy == pytest.approx(dy_gt, abs=0.7)
    assert theta == pytest.approx(theta_gt, abs=0.025)

    # Vérif alignement final (résidu)
    g1 = rgb_to_gray_float(img1)
    g2 = rgb_to_gray_float(img2)
    g1_aligned = warp_image_rotation(g1, dx, dy, theta)

    mask = make_mask(g1.shape[0], g1.shape[1], 20)
    assert mse(g2, g1_aligned, mask=mask) < 60.0



from ex4 import smooth_motions, apply_drift_correction, stabilize_sequence_optimized


# -------------------------
# Helpers
# -------------------------
def almost_tuple(t1, t2, atol=1e-9):
    return all(abs(a - b) <= atol for a, b in zip(t1, t2))


# =========================
# 1) Tests smooth_motions
# =========================

def test_smooth_motions_shorter_than_window_returns_same():
    motions = [(1.0, 2.0, 0.1), (2.0, 3.0, 0.2)]
    out = smooth_motions(motions, window_size=5)
    assert out == motions


def test_smooth_motions_constant_sequence_unchanged():
    motions = [(5.0, -2.0, 0.03)] * 10
    out = smooth_motions(motions, window_size=5)
    assert out == motions


def test_smooth_motions_moving_average_edges_are_truncated():
    # dx = [0, 10, 0], window=3
    motions = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    out = smooth_motions(motions, window_size=3)

    # i=0 -> mean over [0,10] => 5
    # i=1 -> mean over [0,10,0] => 3.333...
    # i=2 -> mean over [10,0] => 5
    assert out[0][0] == pytest.approx(5.0)
    assert out[1][0] == pytest.approx(10.0 / 3.0)
    assert out[2][0] == pytest.approx(5.0)

    # dy, theta restent 0
    for i in range(3):
        assert out[i][1] == pytest.approx(0.0)
        assert out[i][2] == pytest.approx(0.0)


# =========================
# 2) Tests apply_drift_correction
# =========================

def test_apply_drift_correction_zero_strength_no_change():
    motions = [(1.0, 2.0, 0.1), (-1.0, 0.0, -0.1), (0.5, 1.0, 0.0)]
    out = apply_drift_correction(motions, correction_strength=0.0)
    assert out == motions


def test_apply_drift_correction_empty_list():
    assert apply_drift_correction([], correction_strength=0.5) == []


def test_apply_drift_correction_constant_drift_should_stay_same():
    # Si toutes les frames ont le même dy et theta,
    # avg_dy_per_frame = dy, cumulative_dy = dy*(i+1) => correction = 0
    motions = [(0.0, 2.0, 0.05)] * 8
    out = apply_drift_correction(motions, correction_strength=0.8)
    assert out == motions


def test_apply_drift_correction_alternating_drift_reduces_cumulative_bias():
    # dy alternant: +1, -1, +1, -1...
    # avg_dy_per_frame ~ 0 => la correction essaie de ramener cumulative_dy vers 0
    motions = [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
               (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]
    out = apply_drift_correction(motions, correction_strength=1.0)

    # Propriété testable : le dy corrigé du 1er element doit être plus petit que 1
    # (car cumulative_dy=1, avg=0 => correction = -(1-0)/1 = -1 => dy+correction=0)
    assert out[0][1] == pytest.approx(0.0)

    # Et la somme cumulée corrigée doit rester plus proche de 0 que l'originale
    cum_original = np.cumsum([m[1] for m in motions])
    cum_corrected = np.cumsum([m[1] for m in out])
    assert np.max(np.abs(cum_corrected)) <= np.max(np.abs(cum_original))


# =========================
# 3) Tests stabilize_sequence_optimized
# =========================

def test_stabilize_sequence_shapes_and_lengths():
    images = [np.zeros((10, 12, 3), dtype=np.uint8) for _ in range(6)]
    motions = [(1.0, 0.0, 0.0)] * (len(images) - 1)

    params, residuals = stabilize_sequence_optimized(
        images, motions, smooth_window=0, drift_correction=0.0
    )

    assert len(params) == len(images)
    assert len(residuals) == len(images) - 1
    assert params[0] == (0, 0, 0, 0)


def test_stabilize_sequence_cumulative_dx_and_residuals_integer_parts():
    # dx cumulatif: 0.6, 1.2, 1.8
    images = [np.zeros((10, 12, 3), dtype=np.uint8) for _ in range(4)]
    motions = [(0.6, 0.0, 0.0), (0.6, 0.0, 0.0), (0.6, 0.0, 0.0)]

    params, residuals = stabilize_sequence_optimized(
        images, motions, smooth_window=0, drift_correction=0.0
    )

    # i=1: cum=0.6 -> round=1, frac=-0.4
    assert params[1][3] == 1
    assert params[1][0] == pytest.approx(-0.4)

    # i=2: cum=1.2 -> round=1, frac=0.2
    assert params[2][3] == 1
    assert params[2][0] == pytest.approx(0.2)

    # i=3: cum=1.8 -> round=2, frac=-0.2
    assert params[3][3] == 2
    assert params[3][0] == pytest.approx(-0.2)

    # residuals = differences of dx_int
    # residual[0] = dx_int(1) = 1
    assert residuals[0][0] == 1
    # residual[1] = dx_int(2)-dx_int(1) = 0
    assert residuals[1][0] == 0
    # residual[2] = dx_int(3)-dx_int(2) = 1
    assert residuals[2][0] == 1

    # dy/theta cumulés doivent rester 0
    for i in range(1, 4):
        assert params[i][1] == pytest.approx(0.0)
        assert params[i][2] == pytest.approx(0.0)


def test_stabilize_sequence_cumulative_dy_theta():
    images = [np.zeros((10, 12, 3), dtype=np.uint8) for _ in range(5)]
    motions = [
        (0.0, 1.0, 0.01),
        (0.0, 2.0, 0.02),
        (0.0, -1.0, 0.00),
        (0.0, 0.5, -0.01),
    ]

    params, residuals = stabilize_sequence_optimized(
        images, motions, smooth_window=0, drift_correction=0.0
    )

    # cum dy: 1, 3, 2, 2.5
    assert params[1][1] == pytest.approx(1.0)
    assert params[2][1] == pytest.approx(3.0)
    assert params[3][1] == pytest.approx(2.0)
    assert params[4][1] == pytest.approx(2.5)

    # cum theta: 0.01, 0.03, 0.03, 0.02
    assert params[1][2] == pytest.approx(0.01)
    assert params[2][2] == pytest.approx(0.03)
    assert params[3][2] == pytest.approx(0.03)
    assert params[4][2] == pytest.approx(0.02)

    assert len(residuals) == 4


def test_stabilize_sequence_smoothing_reduces_variance_on_noisy_dx():
    images = [np.zeros((10, 12, 3), dtype=np.uint8) for _ in range(10)]
    # dx bruité autour de 1.0
    motions = [(1.0 + ((-1) ** i) * 0.5, 0.0, 0.0) for i in range(9)]

    params0, _ = stabilize_sequence_optimized(images, motions, smooth_window=0, drift_correction=0.0)
    paramsS, _ = stabilize_sequence_optimized(images, motions, smooth_window=5, drift_correction=0.0)

    dx_int0 = np.array([p[3] for p in params0[1:]])
    dx_intS = np.array([p[3] for p in paramsS[1:]])

    # le lissage devrait réduire les changements brusques de dx_int (heuristique)
    jumps0 = np.sum(np.abs(np.diff(dx_int0)))
    jumpsS = np.sum(np.abs(np.diff(dx_intS)))
    assert jumpsS <= jumps0
