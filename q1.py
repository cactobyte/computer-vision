import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt


def create_diagonal_edge_image(size: int = 9) -> np.ndarray:
    """
    Generates a grayscale image with a diagonal edge from top-left to bottom-right.
    Pixels on and above the diagonal (x >= y) are 255, below are 0.
    """
    img = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            if x >= y:
                img[y, x] = 255.0
    return img


def _apply_3x3_stencil_at(img: np.ndarray, x: int, y: int, k: np.ndarray) -> float:
    """
    Applies a 3x3 kernel centered at (x, y).
    Assumes (x, y) is not on the image border.
    """
    patch = img[y - 1:y + 2, x - 1:x + 2]
    return float(np.sum(patch * k))


# ── Part 2: Custom simple gradient (central differences) ──

def compute_custom_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float, float, float, float]:
    """
    Simple central-difference gradient estimation.
    Returns (gx, gy, magnitude, direction_in_degrees).
    """
    # Simple 1D central differences (no smoothing)
    kx = np.array([[0, 0, 0],
                   [-1, 0, 1],
                   [0, 0, 0]], dtype=np.float32)

    ky = np.array([[0, -1, 0],
                   [0,  0, 0],
                   [0,  1, 0]], dtype=np.float32)

    gx = _apply_3x3_stencil_at(img, x, y, kx)
    gy = _apply_3x3_stencil_at(img, x, y, ky)

    mag = (gx ** 2 + gy ** 2) ** 0.5
    ang = np.degrees(np.arctan2(gy, gx))
    return gx, gy, mag, ang


# ── Part 3: Standard Sobel ──

def compute_sobel_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float, float, float, float]:
    """
    Standard 3x3 Sobel gradient estimation.
    Returns (gx, gy, magnitude, direction_in_degrees).
    """
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    gx = _apply_3x3_stencil_at(img, x, y, kx)
    gy = _apply_3x3_stencil_at(img, x, y, ky)

    mag = (gx ** 2 + gy ** 2) ** 0.5
    ang = np.degrees(np.arctan2(gy, gx))
    return gx, gy, mag, ang


# ── Part 4: Diagonal-corrected gradient ──

def compute_diagonal_corrected_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float, float, float, float]:
    """
    Uses diagonal difference kernels (inspired by Roberts Cross)
    to better capture gradient along 45-degree and 135-degree directions,
    then converts back to horizontal/vertical components.

    Returns (gx, gy, magnitude, direction_in_degrees).
    """
    # Diagonal kernels: measure intensity change along 45° and 135°
    # k_45 detects change along the 45° direction (top-right to bottom-left)
    k_45 = np.array([[ 0, 0, 1],
                     [ 0, 0, 0],
                     [-1, 0, 0]], dtype=np.float32)

    # k_135 detects change along the 135° direction (top-left to bottom-right)
    k_135 = np.array([[ 1, 0, 0],
                      [ 0, 0, 0],
                      [ 0, 0,-1]], dtype=np.float32)

    g_45 = _apply_3x3_stencil_at(img, x, y, k_45)
    g_135 = _apply_3x3_stencil_at(img, x, y, k_135)

    # Convert diagonal components back to horizontal and vertical
    # The 45° axis is along (1, -1)/sqrt(2) and 135° axis is along (1, 1)/sqrt(2)
    # gx = (g_45 + g_135) / sqrt(2)
    # gy = (-g_45 + g_135) / sqrt(2)
    inv_sqrt2 = 1.0 / np.sqrt(2)
    gx = (g_45 + g_135) * inv_sqrt2
    gy = (-g_45 + g_135) * inv_sqrt2

    mag = (gx ** 2 + gy ** 2) ** 0.5
    ang = np.degrees(np.arctan2(gy, gx))
    return gx, gy, mag, ang


def main():
    img = create_diagonal_edge_image(size=9)
    x, y = 4, 4

    # Print the image for verification
    print("Diagonal edge image (9x9):")
    print(img.astype(int))
    print()

    # Print the 3x3 patch at (4,4) for clarity
    patch = img[y - 1:y + 2, x - 1:x + 2]
    print(f"3x3 patch centered at ({x},{y}):")
    print(patch.astype(int))
    print()

    # Compute all three gradient estimates
    cx, cy, c_mag, c_ang = compute_custom_gradient(img, x, y)
    sx, sy, s_mag, s_ang = compute_sobel_gradient(img, x, y)
    dx, dy, d_mag, d_ang = compute_diagonal_corrected_gradient(img, x, y)

    # True gradient direction for a diagonal edge (x = y boundary,
    # bright above/right, dark below/left) should be approximately
    # perpendicular to the edge, pointing into the bright region.
    # Edge runs along (1,1), so gradient points along (-1,1) direction
    # which is 135° (or equivalently the negative-x, positive-y quadrant).
    # With our sign conventions, expect around -45° or 135°.

    print(f"{'Method':<28} {'Gx':>10} {'Gy':>10} {'Magnitude':>12} {'Direction':>12}")
    print("-" * 75)
    print(f"{'Custom (central diff)':<28} {cx:>10.2f} {cy:>10.2f} {c_mag:>12.3f} {c_ang:>10.2f}°")
    print(f"{'Sobel (3x3)':<28} {sx:>10.2f} {sy:>10.2f} {s_mag:>12.3f} {s_ang:>10.2f}°")
    print(f"{'Diagonal-corrected':<28} {dx:>10.2f} {dy:>10.2f} {d_mag:>12.3f} {d_ang:>10.2f}°")
    print()

    # Expected: true edge is at 45°, gradient perpendicular = 135° (or -45°)
    true_direction = -45.0  # gradient direction perpendicular to diagonal edge
    print(f"True gradient direction (perpendicular to 45° edge): {true_direction}°")
    print(f"Custom direction error:            {abs(c_ang - true_direction):.2f}°")
    print(f"Sobel direction error:             {abs(s_ang - true_direction):.2f}°")
    print(f"Diagonal-corrected direction error: {abs(d_ang - true_direction):.2f}°")

    # ── Visualisation ──
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Diagonal Edge Image (9×9)")
    axes[0].plot(x, y, 'r+', markersize=12, markeredgewidth=2)
    axes[0].set_xticks(range(9))
    axes[0].set_yticks(range(9))

    axes[1].imshow(patch, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("3×3 Patch at (4,4)")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{int(patch[i, j])}", ha='center', va='center',
                         color='red', fontsize=12, fontweight='bold')

    # Bar chart comparing magnitudes
    methods = ['Central\nDiff', 'Sobel', 'Diagonal\nCorrected']
    magnitudes = [c_mag, s_mag, d_mag]
    bars = axes[2].bar(methods, magnitudes, color=['#4477AA', '#EE7733', '#228833'])
    axes[2].set_ylabel('Gradient Magnitude')
    axes[2].set_title('Magnitude Comparison at (4,4)')
    for bar, val in zip(bars, magnitudes):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('q1_results.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()