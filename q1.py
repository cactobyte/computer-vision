# q1.py
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

def create_diagonal_edge_image(size: int = 9) -> np.ndarray:
    """
    Generates a grayscale image with a diagonal edge from top-left to bottom-right.
    Pixels above the diagonal are 255, and on/below the diagonal are 0.
    """
    img = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            if x >= y:  # Diagonal included
                img[y, x] = 255.0
    return img

def _apply_3x3_stencil_at(img: np.ndarray, x: int, y: int, k: np.ndarray) -> float:
    """
    Applies a 3x3 stencil centered at (x, y). Assumes x,y are not on the border.
    """
    patch = img[y - 1:y + 2, x - 1:x + 2]
    return float(np.sum(patch * k))

def compute_custom_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """
    Computes gradient magnitude and direction (degrees) at (x, y)
    using 3x3 Sobel finite-difference stencils (implemented manually).
    """
    # Sobel (3x3) stencils
    kx = np.array([[-1, 0,  1],
                   [-2, 0,  2],
                   [-1, 0,  1]], dtype=np.float32)
    ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)

    gx = _apply_3x3_stencil_at(img, x, y, kx)
    gy = _apply_3x3_stencil_at(img, x, y, ky)

    mag = (gx * gx + gy * gy) ** 0.5
    ang = np.degrees(np.arctan2(gy, gx))  # [-180, 180]
    return mag, ang

def compute_diagonal_corrected_gradient(img: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """
    Modified gradient computation that compensates for diagonal bias.
    """

    # First compute standard Sobel gradient
    mag, ang = compute_custom_gradient(img, x, y)

    # If orientation is approximately diagonal (around ±45°),
    # apply correction factor
    if 30 <= abs(ang) <= 60:
        mag = mag * np.sqrt(2)

    return mag, ang

def main():
    img = create_diagonal_edge_image(size=9)

    x, y = 4, 4  # center pixel
    mag_sobel, ang_sobel = compute_custom_gradient(img, x, y)
    mag_corr, ang_corr = compute_diagonal_corrected_gradient(img, x, y)

    print("Q1: Diagonal edge 9x9 (above diagonal=255, else 0)")
    print(f"At (x={x}, y={y})")
    print(f"Custom (Sobel 3x3)    -> magnitude={mag_sobel:.3f}, direction={ang_sobel:.3f} degrees")
    print(f"Diagonal-corrected   -> magnitude={mag_corr:.3f}, direction={ang_corr:.3f} degrees")
    if mag_sobel > 0:
        print(f"Improvement factor   -> {mag_corr / mag_sobel:.3f}x")

    # Visualise (workshop-style quick plots)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title("Synthetic image")
    plt.xticks([]); plt.yticks([])

    plt.subplot(1, 2, 2)
    # show a 3x3 neighborhood around center
    patch = img[y-1:y+2, x-1:x+2]
    plt.imshow(patch, cmap="gray", vmin=0, vmax=255)
    plt.title("3x3 patch at (4,4)")
    plt.xticks([]); plt.yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
