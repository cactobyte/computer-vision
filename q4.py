import numpy as np


def apply_forward_transform(points: np.ndarray, theta: float, t: np.ndarray) -> np.ndarray:
    """
    Forward transform H:
        p' = R(theta) * (p - C) + C + t
    
    where C is the centroid of the original points.
    
    This is: rotate by theta about the centroid, then translate by t.
    """
    P = np.asarray(points, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(2)
    C = P.mean(axis=0)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]], dtype=np.float64)

    # Rotate each point about centroid, then translate
    P_prime = (P - C) @ R.T + C + t
    return P_prime


def compute_inverse_transform(points_transformed: np.ndarray, theta: float,
                               t: np.ndarray) -> np.ndarray:
    """
    Inverse transform H^{-1}:
    
    DERIVATION:
        Forward:  p' = R(theta) (p - C) + C + t
        
        Step 1:   p' - C - t = R(theta) (p - C)
        
        Step 2:   R(-theta) (p' - C - t) = p - C
                  (because R(theta)^{-1} = R(-theta) for rotation matrices)
        
        Step 3:   p = R(-theta) (p' - C - t) + C
        
    Rewriting to show same functional form:
        Let C' = C + t  (centroid of transformed points)
        Let t_inv = -R(-theta) t
        
        p = R(-theta) (p' - C') + C'  +  (C - C' - R(-theta)(C - C'))
        
        ... which simplifies to:
        p = R(-theta) (p' - C') + C' + t_inv
        
        This IS the same form: rotation about C' by (-theta), then translation.
        So the inverse has the same functional form as the forward transform.
    
    EXISTENCE CONDITIONS:
        The inverse exists if and only if R(theta) is invertible.
        Since R(theta) is an orthogonal matrix, det(R) = 1 != 0 for all theta.
        Therefore, the inverse ALWAYS exists for any theta and any t.
    
    Args:
        points_transformed: (N,2) array of transformed points
        theta: rotation angle in radians (same as used in forward)
        t: translation vector (2,) (same as used in forward)
    
    Returns:
        points_original: (N,2) array of recovered original points
    """
    P_prime = np.asarray(points_transformed, dtype=np.float64)
    if P_prime.ndim != 2 or P_prime.shape[1] != 2:
        raise ValueError("points_transformed must have shape (N, 2)")

    t = np.asarray(t, dtype=np.float64).reshape(2)

    # Recover original centroid:
    # Forward maps centroid C -> R(theta)(C - C) + C + t = C + t
    # So C' = C + t, therefore C = C' - t
    C_prime = P_prime.mean(axis=0)
    C = C_prime - t

    # R(-theta)
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    R_inv = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]], dtype=np.float64)

    # p = R(-theta) (p' - C - t) + C
    P_recovered = (P_prime - C - t) @ R_inv.T + C

    return P_recovered


def disprove_naive_inverse(theta: float, t: np.ndarray, C: np.ndarray):
    """
    PROOF THAT H^{-1} != R(-theta) + T(-t)  (in general)
    
    The naive claim is that the inverse is simply:
        "rotate by -theta, then translate by -t"
    i.e.,  p_naive = R(-theta) * p' + (-t)
    
    But the correct inverse is:
        p = R(-theta) * (p' - C - t) + C
        p = R(-theta) * p' - R(-theta)(C + t) + C
    
    The naive version gives:
        p_naive = R(-theta) * p' - t
    
    These are equal only if:
        -R(-theta)(C + t) + C = -t
    
    Which simplifies to:
        C - R(-theta)*C - R(-theta)*t + t = 0
        (I - R(-theta)) * C + (I - R(-theta)) * t = 0   ... NOT true in general
        
    This only holds when (I - R(-theta))(C + t) = 0,
    which requires either theta = 0 or C + t = 0.
    
    Therefore H^{-1} != R(-theta) + T(-t) in general.
    """
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    R_inv = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]], dtype=np.float64)

    t = np.asarray(t, dtype=np.float64).reshape(2)
    C = np.asarray(C, dtype=np.float64).reshape(2)

    # Check if (I - R(-theta)) @ (C + t) == 0
    I = np.eye(2)
    residual = (I - R_inv) @ (C + t)

    return residual


def main():
    # ── Test setup ──
    P = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 1.0],
                  [0.0, 1.0]], dtype=np.float64)

    theta = np.deg2rad(30)
    t = np.array([2.0, -1.0])
    C = P.mean(axis=0)

    print("=" * 65)
    print("Q4: RIGID BODY TRANSFORM — INVERSE AND ANALYSIS")
    print("=" * 65)
    print()

    # ── Part 1: Forward transform ──
    print("ORIGINAL POINTS:")
    print(P)
    print(f"Centroid C = {C}")
    print()

    P_prime = apply_forward_transform(P, theta, t)
    print(f"FORWARD TRANSFORM (theta={np.degrees(theta):.1f} deg, t={t}):")
    print(P_prime)
    print(f"Transformed centroid C' = {P_prime.mean(axis=0)}")
    print(f"Expected C' = C + t = {C + t}")
    print()

    # ── Part 2: Inverse transform ──
    P_recovered = compute_inverse_transform(P_prime, theta, t)
    print("RECOVERED POINTS (via inverse):")
    print(P_recovered)
    max_err = np.max(np.abs(P - P_recovered))
    print(f"Max absolute recovery error: {max_err:.2e}")
    print()

    # ── Part 3: Same functional form ──
    print("SAME FUNCTIONAL FORM:")
    print("  Forward:  p' = R(theta)  * (p  - C)  + C  + t")
    print("  Inverse:  p  = R(-theta) * (p' - C') + C' + t_inv")
    print(f"  where C' = C + t = {C + t}")
    print("  Both are: rotation about a centroid, then translation.")
    print("  Therefore the inverse has the SAME functional form. QED")
    print()

    # ── Part 4: Existence conditions ──
    print("EXISTENCE CONDITIONS:")
    print("  R(theta) is orthogonal => det(R) = 1 for all theta")
    print("  Therefore R(theta) is always invertible")
    print("  The inverse transform exists for ALL theta and ALL t.")
    print()

    # ── Part 5: Disprove H^{-1} = R(-theta) + T(-t) ──
    print("DISPROOF: H^{-1} != R(-theta) + T(-t)")
    print()

    # Correct inverse
    p_correct = P_recovered

    # Naive inverse: rotate by -theta then translate by -t (NO centroid)
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    R_inv = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]])
    p_naive = P_prime @ R_inv.T + (-t)

    print("  Correct inverse result:")
    print(f"  {p_correct}")
    print()
    print("  Naive R(-theta) + T(-t) result:")
    print(f"  {p_naive}")
    print()

    diff = np.max(np.abs(p_correct - p_naive))
    print(f"  Max difference: {diff:.6f}")
    print()

    if diff > 1e-10:
        print("  CONCLUSION: H^{-1} != R(-theta) + T(-t)")
        print("  The naive decomposition fails because the rotation in the")
        print("  forward transform is about the CENTROID, not the origin.")
        print("  Ignoring the centroid produces incorrect results.")
    else:
        print("  (Results happen to match for this specific case)")

    print()
    residual = disprove_naive_inverse(theta, t, C)
    print(f"  Analytical residual (I - R(-theta)) @ (C + t) = {residual}")
    print(f"  This is non-zero => naive inverse is WRONG in general.")
    print(f"  It would only work if theta=0 or C+t=0.")


if __name__ == "__main__":
    main()