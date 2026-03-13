import torch


def cox_deboor(t, c, p, x):
    """
    Evaluate B-spline at x using Cox-de Boor recursion.
    
    Parameters:
        t : knot vector (array-like)
        c : control point weights (array-like)
        p : degree of the spline
        x : evaluation point
    
    Returns:
        B-spline value at x
    """
    n = len(c)
    # Find knot span index k such that t[k] <= x < t[k+1]
    def find_span(x):
        for k in range(p, n):
            if t[k] <= x < t[k + 1]:
                return k
        return n - 1  # clamp to last valid span
    k = find_span(x)
    x_scalar = x.squeeze()
    # Initialize d as a list of scalar tensors (no inplace ops for autograd)
    d = [c[k - p + j] for j in range(p + 1)]
    # Cox-de Boor recursion
    for r in range(1, p + 1):
        new_d = list(d)
        for j in range(p, r - 1, -1):
            i = j + k - p
            denom = t[i + p - r + 1] - t[i]
            alpha = torch.where(denom == 0, torch.tensor(0.0), (x_scalar - t[i]) / denom)
            new_d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
        d = new_d
    
    return d[p]


def main():
    pass
if __name__ == "__main__":
    t = [0, 0, 0, 1, 2, 3, 3, 3]  # knot vector (clamped cubic)
    c = [0, 1, 2, 1, 0]            # control points
    p = 2                           # quadratic

    print(cox_de_boor(t, c, p, 1.5))  # → ~1.5
    print(cox_de_boor(t, c, p, 0.5))  # → ~0.625