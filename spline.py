import torch


def cox_deboor(c: torch.Tensor, t: torch.Tensor, order: int, x: torch.Tensor):
    n = len(c)
    scalar = x.dim() == 0
    if scalar:
        x = x.unsqueeze(0)
    batch = x.shape[0]

    b = ((t[:n].unsqueeze(0) <= x.unsqueeze(1)) & (x.unsqueeze(1) < t[1:n+1].unsqueeze(0))).float()
    right_end = (x == t[-1])
    b[right_end, -1] = 1.0

    for j in range(1, order):
        t_k   = t[:n]
        t_kj  = t[j:n+j]
        t_k1  = t[1:n+1]
        t_kj1 = t[j+1:n+j+1]

        denom1 = t_kj - t_k
        denom2 = t_kj1 - t_k1

        w1 = torch.where(denom1 != 0, (x.unsqueeze(1) - t_k.unsqueeze(0)) / denom1.unsqueeze(0), torch.zeros(batch, n))
        w2 = torch.where(denom2 != 0, (t_kj1.unsqueeze(0) - x.unsqueeze(1)) / denom2.unsqueeze(0), torch.zeros(batch, n))

        b = w1 * b + w2 * torch.cat([b[:, 1:], torch.zeros(batch, 1)], dim=1)

    result = (b * c.unsqueeze(0)).sum(dim=1)
    if scalar:
        return result.squeeze(0)
    return result


