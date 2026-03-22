import torch


def cox_deboor(c: torch.Tensor, t: torch.Tensor, order: int, x: float):
    n = len(c)
    
    # Degree-0 basis functions
    b = ((t[:n] <= x) & (x < t[1:n+1])).float()
    # Handle right endpoint: clamp last active basis
    if x == t[-1]:
        b[-1] = 1.0

    # Cox-de Boor recursion (vectorized over k, iterating over degree j)
    for j in range(1, order):
        t_k   = t[:n]          # t[k]
        t_kj  = t[j:n+j]      # t[k+j]
        t_k1  = t[1:n+1]      # t[k+1]
        t_kj1 = t[j+1:n+j+1]  # t[k+j+1]

        denom1 = t_kj - t_k
        denom2 = t_kj1 - t_k1

        w1 = torch.where(denom1 != 0, (x - t_k) / denom1, torch.zeros(n))
        w2 = torch.where(denom2 != 0, (t_kj1 - x) / denom2, torch.zeros(n))

        b = w1 * b + w2 * torch.cat([b[1:], torch.zeros(1)])

    # Weighted sum over control points
    return (b * c).sum()


