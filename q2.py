# q2.py
import numpy as np
import cv2

def spatial_rgb_histogram(image_bgr: np.ndarray, num_bins: int = 16, grid: int = 2) -> np.ndarray:
    """
    Spatial RGB histogram: split image into grid x grid cells, compute per-cell RGB histograms,
    concatenate into one feature vector.

    - Preserves coarse spatial layout (which region has which colours)
    - Still tractable (vector length = grid*grid*3*num_bins)

    Returns: 1D float32 feature vector (L1-normalised).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image input")

    h, w = image_bgr.shape[:2]
    # Convert BGR -> RGB
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Bin edges for [0, 256)
    edges = np.linspace(0, 256, num_bins + 1, dtype=np.float32)

    feats = []
    cell_h = h // grid
    cell_w = w // grid

    for gy in range(grid):
        for gx in range(grid):
            y0 = gy * cell_h
            x0 = gx * cell_w
            # last cell takes the remainder
            y1 = (gy + 1) * cell_h if gy < grid - 1 else h
            x1 = (gx + 1) * cell_w if gx < grid - 1 else w

            cell = image[y0:y1, x0:x1, :]  # RGB

            # per-channel histograms
            for c in range(3):
                hist, _ = np.histogram(cell[:, :, c], bins=edges)
                feats.append(hist.astype(np.float32))

    feat = np.concatenate(feats, axis=0)

    s = float(np.sum(feat))
    if s > 0:
        feat /= s

    return feat

def main():
    img = cv2.imread("data/flower.jpg")  # BGR
    feat = spatial_rgb_histogram(img, num_bins=16, grid=2)
    print("Q2: spatial RGB histogram")
    print("feature length =", feat.size)
    print("sum (should be 1.0) =", float(np.sum(feat)))

if __name__ == "__main__":
    main()
