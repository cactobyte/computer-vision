import time
import numpy as np
from PIL import Image


def compute_colour_histogram_np(im: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Vectorised NumPy implementation (no Python loops, no histogram functions).
    
    Args:
        im: NumPy array of shape (H, W, 3), dtype uint8, RGB image
        num_bins: number of bins per channel
    
    Returns:
        hist: shape (3, num_bins) float32
    """
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")

    vals = im.reshape(-1, 3).astype(np.int32)  # (N, 3)
    bins = (vals * num_bins) // 256             # (N, 3)

    n = bins.shape[0]
    bins_flat = bins.T.reshape(-1)              # length 3N
    chan_ids = np.repeat(np.arange(3, dtype=np.int32), n)

    hist = np.zeros((3, num_bins), dtype=np.float32)
    np.add.at(hist, (chan_ids, bins_flat), 1.0)

    return hist


def compute_colour_histogram_py(im: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Pure Python loop-based implementation (no NumPy operations for counting).
    
    Args:
        im: NumPy array of shape (H, W, 3), dtype uint8, RGB image
        num_bins: number of bins per channel
    
    Returns:
        hist: shape (3, num_bins) float32
    """
    h, w = im.shape[0], im.shape[1]

    hist_r = [0] * num_bins
    hist_g = [0] * num_bins
    hist_b = [0] * num_bins

    for y in range(h):
        for x in range(w):
            r = int(im[y, x, 0])
            g = int(im[y, x, 1])
            b = int(im[y, x, 2])

            br = (r * num_bins) // 256
            bg = (g * num_bins) // 256
            bb = (b * num_bins) // 256

            # Safety clamp
            if br >= num_bins:
                br = num_bins - 1
            if bg >= num_bins:
                bg = num_bins - 1
            if bb >= num_bins:
                bb = num_bins - 1

            hist_r[br] += 1
            hist_g[bg] += 1
            hist_b[bb] += 1

    return np.array([hist_r, hist_g, hist_b], dtype=np.float32)


def time_fn(fn, repeats: int, *args) -> float:
    """Time a function over multiple repeats, return average in ms."""
    fn(*args)  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeats


def main():
    # Load image without cv2
    try:
        img = np.array(Image.open("data/flower.jpg").convert("RGB"), dtype=np.uint8)
    except FileNotFoundError:
        print("No image found, creating synthetic 100x100 test image")
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print()

    bin_values = [8, 16, 32]

    print(f"{'num_bins':>10} {'NumPy (ms)':>12} {'Python (ms)':>14} {'Speedup':>10} {'Match':>8}")
    print("-" * 60)

    for num_bins in bin_values:
        t_np = time_fn(compute_colour_histogram_np, 50, img, num_bins)
        t_py = time_fn(compute_colour_histogram_py, 3, img, num_bins)

        h_np = compute_colour_histogram_np(img, num_bins)
        h_py = compute_colour_histogram_py(img, num_bins)

        match = np.allclose(h_np, h_py)
        speedup = t_py / t_np if t_np > 0 else float('inf')

        print(f"{num_bins:>10} {t_np:>12.3f} {t_py:>14.3f} {speedup:>9.1f}x {str(match):>8}")

    print()
    print("=" * 60)
    print("TRADE-OFF ANALYSIS")
    print("=" * 60)
    print()
    print("NumPy (vectorised):")
    print("  + Much faster due to C-level array operations")
    print("  + Speed advantage grows with image size")
    print("  - Higher memory usage (creates intermediate arrays)")
    print("  - Requires NumPy; less portable")
    print()
    print("Python (loop-based):")
    print("  + Minimal memory overhead (processes one pixel at a time)")
    print("  + No dependencies beyond standard Python")
    print("  + Easier to understand and debug")
    print("  - Orders of magnitude slower for large images")
    print("  - Speed is roughly O(H * W) with large constant factor")
    print()
    print("Effect of num_bins:")
    print("  - Increasing num_bins has minimal effect on runtime for both methods")
    print("    (the dominant cost is iterating over pixels, not over bins)")
    print("  - More bins = finer colour resolution but sparser histogram")
    print("  - Fewer bins = coarser quantisation but more robust to noise")


if __name__ == "__main__":
    main()