import numpy as np
from PIL import Image


def _manual_histogram(channel: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Computes a histogram for a single channel without using any histogram functions.
    """
    hist = np.zeros(num_bins, dtype=np.float32)
    bin_width = 256.0 / num_bins
    
    flat = channel.ravel()
    
    for val in flat:
        b = int(val / bin_width)
        if b >= num_bins:
            b = num_bins - 1
        hist[b] += 1.0
    
    return hist


def spatial_rgb_histogram(image_rgb: np.ndarray, num_bins: int = 16, grid: int = 2) -> np.ndarray:
    """
    Spatial RGB histogram: split image into grid x grid cells, 
    compute per-cell RGB histograms (manually), concatenate.
    Returns: 1D float32 feature vector (L1-normalised).
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Empty image input")

    h, w = image_rgb.shape[:2]

    feats = []
    cell_h = h // grid
    cell_w = w // grid

    for gy in range(grid):
        for gx in range(grid):
            y0 = gy * cell_h
            x0 = gx * cell_w
            y1 = (gy + 1) * cell_h if gy < grid - 1 else h
            x1 = (gx + 1) * cell_w if gx < grid - 1 else w

            cell = image_rgb[y0:y1, x0:x1, :]

            for c in range(3):
                hist = _manual_histogram(cell[:, :, c], num_bins)
                feats.append(hist)

    feat = np.concatenate(feats, axis=0)

    s = float(np.sum(feat))
    if s > 0:
        feat /= s

    return feat


def main():
    # Try loading an image, fall back to synthetic
    try:
        img = np.array(Image.open("data/flower.jpg").convert("RGB"), dtype=np.float32)
    except FileNotFoundError:
        print("No image file found, using synthetic 8x8 test image")
        img = np.zeros((8, 8, 3), dtype=np.float32)
        img[:4, :4] = [255, 0, 0]
        img[:4, 4:] = [0, 255, 0]
        img[4:, :4] = [0, 0, 255]
        img[4:, 4:] = [255, 255, 0]

    feat = spatial_rgb_histogram(img, num_bins=16, grid=2)

    print("Q2: Spatial RGB Histogram")
    print(f"  Feature vector length: {feat.size}")
    print(f"  Expected length: {2*2*3*16}")
    print(f"  Sum (should be 1.0): {float(np.sum(feat)):.6f}")
    print()
    print("Spatial information RETAINED:")
    print("  - Coarse colour distribution per region")
    print()
    print("Spatial information LOST:")
    print("  - Exact pixel positions within each cell")
    print("  - Fine-grained textures and edges within cells")
    print()
    print("Application: Scene classification where layout matters")
    print("  (e.g., sky=blue at top, grass=green at bottom)")


if __name__ == "__main__":
    main()