"""
PWTA: lightweight patch-based reconstruction anomaly detection with
kurtosis-weighted channel scoring.

Design decisions (each checked over 3-5 seeds):
  1. Scoring is the main performance lever: weight channels by kurtosis
     instead of averaging them.
  2. No re-normalization: the benchmark data is already scaled, and z-scoring
     near-constant channels amplifies tiny deviations and destroys the signal.
  3. window=100, patch=10: larger or mixed scales gave no gain (37+ settings).
  4. Projection: overlapping window scores are averaged; channel errors are
     averaged over each window before channel aggregation.
  5. Batched evaluation: avoids out-of-memory errors on long test series
     with bit-identical results.
  6. (Optional) Routing between PWTA and a channel-attention variant.

Evaluation: raw PR-AUC (threshold-free, no point adjustment). Point-adjusted F1
is avoided because even random scores reach a high PA-F1 (Kim et al., AAAI 2022).

Dependencies: torch>=2.0, numpy, scipy, scikit-learn
"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kurtosis
from sklearn.metrics import average_precision_score

# ----------------------------------------------------------------------
# Hyperparameters (performance was flat around these values)
# ----------------------------------------------------------------------
WINDOW   = 100   # sliding window length
PATCH    = 10    # patch length (10 tokens per window)
STRIDE   = 10    # window stride
D_MODEL  = 64
N_HEADS  = 4
N_LAYERS = 2
EPOCHS   = 50
BATCH    = 128
LR       = 1e-3
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
N_PATCH  = WINDOW // PATCH


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def _posenc(n, dm):
    pe = torch.zeros(n, dm)
    pos = torch.arange(0, n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dm, 2).float() * (-math.log(1e4) / dm))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class PWTA(nn.Module):
    """Patch-Window Transformer Autoencoder.

    Splits each window into patches, embeds each patch (mixing all channels),
    runs a transformer along the time axis and reconstructs the window.
    About 0.15M parameters; size barely grows with the number of channels d.
    """
    def __init__(self, d):
        super().__init__()
        self.id = d
        self.embed = nn.Linear(PATCH * d, D_MODEL)          # mixes all channels in a patch
        self.register_buffer('pe', _posenc(N_PATCH, D_MODEL))
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL * 4, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, N_LAYERS)
        self.recon = nn.Linear(D_MODEL, PATCH * d)

    def forward(self, x):                                    # x: (B, WINDOW, d)
        B = x.size(0)
        xp = x.reshape(B, N_PATCH, PATCH * self.id)
        h = self.enc(self.embed(xp) + self.pe)
        return self.recon(h).reshape(B, WINDOW, self.id)


# ----------------------------------------------------------------------
# Windowing and projection
# ----------------------------------------------------------------------
def make_windows(series):
    """(T, d) -> (n_win, WINDOW, d) sliding windows."""
    return np.stack([series[i:i + WINDOW]
                     for i in range(0, len(series) - WINDOW + 1, STRIDE)])


def avg_project(win_scores, cov):
    """Window scores (n_win,) -> point scores (cov,) by averaging overlapping windows."""
    s = np.zeros(cov); c = np.zeros(cov)
    for i, v in enumerate(win_scores):
        a = i * STRIDE
        s[a:a + WINDOW] += v
        c[a:a + WINDOW] += 1
    return s / np.maximum(c, 1)


# ----------------------------------------------------------------------
# Training and reconstruction error (batched evaluation avoids OOM)
# ----------------------------------------------------------------------
def train_and_error(model_cls, train_series, test_series, seed=0):
    """Train a model and return test reconstruction errors.

    Returns:
        cm:  (n_win, d) per-window, per-channel mean squared error
        cov: number of time points covered by the windows
    Inputs are used at their native scale (no normalization).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.tensor(make_windows(train_series).astype(np.float32))
    m = model_cls(train_series.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), LR)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr), batch_size=BATCH, shuffle=True)
    m.train()
    for _ in range(EPOCHS):
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss_fn(m(xb), xb).backward()
            opt.step()

    m.eval()
    Xte = torch.tensor(make_windows(test_series).astype(np.float32))
    errs = []
    with torch.no_grad():                                   # batched eval: never put the whole test set on the GPU
        for (xb,) in DataLoader(TensorDataset(Xte), batch_size=512):
            xb = xb.to(DEVICE)
            errs.append(((m(xb) - xb) ** 2).cpu().numpy())  # (b, WINDOW, d)
    re = np.concatenate(errs)                               # (n_win, WINDOW, d)
    cov = len(Xte) * STRIDE - STRIDE + WINDOW
    cm = re.mean(1)                                         # (n_win, d)
    return cm, cov


# ----------------------------------------------------------------------
# Scoring: the main lever
# ----------------------------------------------------------------------
def kurtosis_score(cm, cov, sharp='soft'):
    """Weight channels by kurtosis to produce point scores.

    Anomalies usually live in a few channels, so a plain average dilutes them
    with normal channels. Channels that are usually quiet and then spike
    (high kurtosis) get more weight, which keeps the anomaly signal.

    sharp='soft' : w = kurt^0.5   (recommended)
    sharp='base' : w = kurt
    sharp='mean' : plain average  (baseline for comparison)

    Assumes anomalies are sparse; with dense anomalies (e.g. PSM, 28%) the
    weighting stops helping.
    """
    if sharp == 'mean':
        w = np.ones(cm.shape[1])
    else:
        krt = np.nan_to_num(kurtosis(cm, axis=0, fisher=True), nan=0.0)
        kc = np.clip(krt, 0, None)
        w = (kc ** 0.5 + 0.05) if sharp == 'soft' else (kc + 0.1)
    w = w / w.sum()
    win = (cm * w).sum(1)                                   # weighted channel sum -> window score
    return avg_project(win, cov)


def anomaly_score(train_series, test_series, seed=0, sharp='soft'):
    """Convenience wrapper: train -> raw reconstruction error -> kurtosis score -> point scores."""
    cm, cov = train_and_error(PWTA, train_series, test_series, seed)
    return kurtosis_score(cm, cov, sharp), cov


# ----------------------------------------------------------------------
# Evaluation (raw PR-AUC)
# ----------------------------------------------------------------------
def pr_auc(score, label):
    """Raw PR-AUC with one 0/1 label per time point. Do not apply point adjustment."""
    L = min(len(score), len(label))
    return average_precision_score(label[:L], score[:L])


# ----------------------------------------------------------------------
# (Optional) routing: a small secondary gain
# ----------------------------------------------------------------------
class HybridPWTA(nn.Module):
    """PWTA plus channel attention (learns channel relations).

    Helps on truly multivariate data (SMD) and hurts on isolated-channel data
    (MSL). Sensitive to the seed, so use it only through routing.
    """
    def __init__(self, d):
        super().__init__()
        self.id = d
        self.eown = nn.Linear(PATCH, D_MODEL)               # per-channel embedding
        self.emix = nn.Linear(PATCH * d, D_MODEL)           # plus fixed linear channel mixing
        self.register_buffer('pe', _posenc(N_PATCH, D_MODEL))
        self.time_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL * 4, 0.1, batch_first=True), 1)
        self.chan_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL * 4, 0.1, batch_first=True), 1)
        self.recon = nn.Linear(D_MODEL, PATCH)

    def forward(self, x):
        B, d = x.size(0), self.id
        xp = x.permute(0, 2, 1).reshape(B, d, N_PATCH, PATCH)
        h = self.eown(xp) + self.emix(x.reshape(B, N_PATCH, PATCH * d)).unsqueeze(1) + self.pe.unsqueeze(0)
        h = self.time_attn(h.reshape(B * d, N_PATCH, D_MODEL)).reshape(B, d, N_PATCH, D_MODEL)
        hc = self.chan_attn(h.permute(0, 2, 1, 3).reshape(B * N_PATCH, d, D_MODEL))
        hc = hc.reshape(B, N_PATCH, d, D_MODEL).permute(0, 2, 1, 3)
        return self.recon(hc).reshape(B, d, WINDOW).permute(0, 2, 1)


def _peakedness(s):
    """Confidence of a score series: how peaked it is (kurtosis)."""
    s = np.asarray(s)
    return float(np.nan_to_num(kurtosis(s - s.min() + 1e-9), nan=0.0))


def routed_score(train_series, test_series, seed=0, mode='dead'):
    """Unsupervised per-entity routing between PWTA and HybridPWTA.

    mode='dead' : use PWTA if at least half of the channels are constant in
                  training, otherwise HybridPWTA. The 0.5 threshold matches
                  Otsu's automatic threshold (0.49); 0.3-0.7 all work.
    mode='conf' : pick the model whose score series is more peaked.

    Returns the selected point scores and cov.
    """
    cmP, cov = train_and_error(PWTA, train_series, test_series, seed)
    cmH, _   = train_and_error(HybridPWTA, train_series, test_series, seed)
    sP = kurtosis_score(cmP, cov, 'soft')
    sH = kurtosis_score(cmH, cov, 'soft')
    if mode == 'dead':
        dead_frac = (train_series.std(0) == 0).mean()
        use_pwta = dead_frac >= 0.5
    else:  # conf
        use_pwta = _peakedness(sP) >= _peakedness(sH)
    return (sP if use_pwta else sH), cov


# ----------------------------------------------------------------------
# Example: SMD
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import os, glob, statistics as st
    SMD = 'OmniAnomaly/ServerMachineDataset'   # set to your local path

    # Single machine. Note: on one entity kurtosis can score below the mean;
    # the gain shows up in the average over all entities (loop below).
    m = 'machine-1-1'
    tr = np.loadtxt(f'{SMD}/train/{m}.txt', delimiter=',').astype(np.float32)
    te = np.loadtxt(f'{SMD}/test/{m}.txt',  delimiter=',').astype(np.float32)
    lb = np.loadtxt(f'{SMD}/test_label/{m}.txt', delimiter=',').astype(int)
    score, cov = anomaly_score(tr, te, seed=0, sharp='soft')
    print(f'[{m}] PWTA+kurtosis PR-AUC = {pr_auc(score, lb):.3f}  (single entities vary a lot)\n')

    # Average over all SMD machines: kurtosis vs. plain mean
    machines = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{SMD}/train/*.txt'))
    kurt_aps, mean_aps = [], []
    for m in machines:
        tr = np.loadtxt(f'{SMD}/train/{m}.txt', delimiter=',').astype(np.float32)
        te = np.loadtxt(f'{SMD}/test/{m}.txt',  delimiter=',').astype(np.float32)
        lb = np.loadtxt(f'{SMD}/test_label/{m}.txt', delimiter=',').astype(int)
        if lb.sum() == 0:
            continue
        cm, cov = train_and_error(PWTA, tr, te, seed=0)     # train once, change only the scoring
        kurt_aps.append(pr_auc(kurtosis_score(cm, cov, 'soft'), lb))
        mean_aps.append(pr_auc(kurtosis_score(cm, cov, 'mean'), lb))
    print(f'=== SMD average (n={len(kurt_aps)}) ===')
    print(f'  PWTA + mean (baseline) : {st.mean(mean_aps):.3f}')
    print(f'  PWTA + kurtosis        : {st.mean(kurt_aps):.3f}')
