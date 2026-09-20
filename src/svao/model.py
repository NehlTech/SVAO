"""Single-layer LSTM regressor with sample-weighted training."""
import numpy as np
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """Many-to-one: a sequence of scalars in, one scalar out.

    Deliberately shallow. With roughly 250 training sequences per station a
    higher-capacity network would memorise the sample, and the comparison
    between weighting schemes would be masked by excess capacity.
    """

    def __init__(self, hidden_size=32, input_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def train_model(X_tr, y_tr, w_tr, X_va, y_va, seed,
                hidden_size=32, epochs=300, batch_size=16, lr=0.01,
                grad_clip=5.0, patience=30, device="cpu"):
    """Train one network and return it with its history.

    The seed fixes initialisation AND batch ordering, so two runs with the same
    seed but different weights differ only in the weights.

    Early stopping uses UNWEIGHTED validation loss. The weights encode a belief
    about which training examples are informative, not that errors in volatile
    months matter less; the model is selected on the criterion it is judged by.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = LSTMRegressor(hidden_size).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    wt = torch.as_tensor(w_tr, dtype=torch.float32, device=device)
    Xv = torch.as_tensor(X_va, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_va, dtype=torch.float32, device=device)

    gen = torch.Generator().manual_seed(seed)
    n = Xt.shape[0]
    best, best_state, bad = float("inf"), None, 0
    hist = {"train": [], "val": []}
    epoch = 0

    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=gen)
        total, seen = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            err = net(Xt[idx]) - yt[idx]
            loss = (wt[idx] * err ** 2).sum() / wt[idx].sum()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()
            total += float(loss) * len(idx); seen += len(idx)

        net.eval()
        with torch.no_grad():
            vl = float(((net(Xv) - yv) ** 2).mean())
        hist["train"].append(total / seen)
        hist["val"].append(vl)

        if vl < best - 1e-9:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if patience and bad >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    hist["best_val"] = best
    hist["epochs_run"] = epoch
    return net, hist


@torch.no_grad()
def predict(net, X, device="cpu"):
    if X.shape[0] == 0:
        return np.empty(0)
    net.eval()
    return net(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
