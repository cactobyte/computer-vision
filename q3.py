# q3.py
import time
from typing import Tuple

import cv2
import numpy as np

def compute_colour_histogram_np(im: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Vectorised NumPy implementation (no Python loops).
    Computes separate histograms for R, G, B channels.

    Returns:
        hist: shape (3, num_bins) float32
              hist[0] = R, hist[1] = G, hist[2] = B
    Constraints (per coursework):
        - No np.histogram, no np.bincount, no other histogram helpers.
        - No Python loops.
    """
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")

    # Values in [0,255]. Compute bin index in [0, num_bins-1]
    vals = im.reshape(-1, 3).astype(np.int32)  # (N, 3)
    bins = (vals * num_bins) // 256            # (N, 3)

    n = bins.shape[0]                          # N
    bins_flat = bins.T.reshape(-1)             # length 3N: R bins then G bins then B bins
    chan_ids = np.repeat(np.arange(3, dtype=np.int32), n)  # length 3N

    hist = np.zeros((3, num_bins), dtype=np.float32)
    np.add.at(hist, (chan_ids, bins_flat), 1.0)  # scatter-add counts

    return hist

def compute_colour_histogram_py(im: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Pure Python loop-based implementation.
    No numpy histogram helpers (np.histogram/np.bincount/etc).

    Returns:
        hist: shape (3, num_bins) float32
    """
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")

    # Convert to plain Python lists so counting is truly loop-based
    pix = im.tolist()

    hist_r = [0] * num_bins
    hist_g = [0] * num_bins
    hist_b = [0] * num_bins

    for row in pix:
        for (r, g, b) in row:
            br = (r * num_bins) // 256
            bg = (g * num_bins) // 256
            bb = (b * num_bins) // 256
            hist_r[br] += 1
            hist_g[bg] += 1
            hist_b[bb] += 1

    return np.array([hist_r, hist_g, hist_b], dtype=np.float32)

def time_fn(fn, im: np.ndarray, num_bins: int, repeats: int = 10) -> float:
    """
    Average runtime in milliseconds over `repeats` runs.
    """
    # Warm-up (esp. for NumPy)
    fn(im, num_bins)

    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(im, num_bins)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / repeats

def main():
    # Read image (OpenCV loads BGR)
    bgr = cv2.imread("data/flower.jpg", cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("Could not read data/flower.jpg")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    for num_bins in (8, 16, 32):
        t_np = time_fn(compute_colour_histogram_np, rgb, num_bins, repeats=30)
        t_py = time_fn(compute_colour_histogram_py, rgb, num_bins, repeats=3)  # pure python is slow

        h_np = compute_colour_histogram_np(rgb, num_bins)
        h_py = compute_colour_histogram_py(rgb, num_bins)

        # Quick correctness check: should match exactly
        same = np.allclose(h_np, h_py)

        print(f"num_bins={num_bins}")
        print(f"  numpy (vectorised)  : {t_np:.3f} ms")
        print(f"  python (loop-based) : {t_py:.3f} ms")
        print(f"  histograms match?   : {same}")

if __name__ == "__main__":
    main()