import torch
import torch.nn as nn
from KAN import KAN


def target(x):
    return torch.exp(x[:, 0]) * torch.sin(x[:, 1] + x[:, 2])


def generate_data(n, noise=0.0):
    x = torch.rand(n, 3) * 4 - 2
    y = target(x)
    if noise > 0:
        y = y + noise * torch.randn_like(y)
    return x, y.unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train(model, x_train, y_train, x_val, y_val, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if epoch % 100 == 0:
            print(f"  epoch {epoch:4d} | train loss {loss.item():.6f} | val loss {val_loss:.6f}")

    return train_losses, val_losses


def evaluate(model, x_test, y_test):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        pred = model(x_test)
        mse = criterion(pred, y_test).item()
        mae = (pred - y_test).abs().mean().item()
    return mse, mae


if __name__ == "__main__":
    torch.manual_seed(42)

    x_train, y_train = generate_data(2000)
    x_val,   y_val   = generate_data(1000)
    x_test,  y_test  = generate_data(1000)

    mlp = MLP(hidden=4)
    kan = KAN([3, 2, 1])

    print("=" * 50)
    print("Training MLP")
    print("=" * 50)
    mlp_train_losses, mlp_val_losses = train(mlp, x_train, y_train, x_val, y_val, epochs=500, lr=1e-3)

    print("=" * 50)
    print("Training KAN")
    print("=" * 50)
    kan_train_losses, kan_val_losses = train(kan, x_train, y_train, x_val, y_val, epochs=500, lr=1e-3)

    mlp_mse, mlp_mae = evaluate(mlp, x_test, y_test)
    kan_mse, kan_mae = evaluate(kan, x_test, y_test)

    mlp_params = sum(p.numel() for p in mlp.parameters())
    kan_params  = sum(p.numel() for p in kan.parameters())

    print("\n" + "=" * 50)
    print("Final Test Results")
    print("=" * 50)
    print(f"{'Model':<10} {'Params':>8} {'Test MSE':>12} {'Test MAE':>12}")
    print("-" * 50)
    print(f"{'MLP':<10} {mlp_params:>8} {mlp_mse:>12.6f} {mlp_mae:>12.6f}")
    print(f"{'KAN':<10} {kan_params:>8} {kan_mse:>12.6f} {kan_mae:>12.6f}")
    print("=" * 50)
