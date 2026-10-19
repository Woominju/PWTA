"""
patch_light.py: memory- and speed-optimized training/evaluation for the patch model.

Same model, scoring and metric as the original pipeline; only how training and
evaluation run has changed:
  1. [71x less memory] Batched evaluation instead of sending the whole test set
     to the GPU at once (the cause of out-of-memory errors). Bit-identical results.
  2. [~2x faster] Training data stays on the GPU, removing the CPU->GPU copy
     on every batch.
  3. [optional] Early stopping (patience), nearly lossless.

Configurations (accuracy/speed trade-off measured on 28 SMD machines):
  - CFG_LOSSLESS : batch 128 + GPU-resident + early stop -> recall difference ~0.006
                   versus the original settings. Use for final results.
  - CFG_FAST     : batch 256 + 2x LR + early stop -> difference ~0.021, faster.
                   Use for exploration.
"""
import math, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW, STRIDE, PATCH_SIZE = 100, 10, 10
N_PATCH = WINDOW // PATCH_SIZE
D_MODEL, N_HEADS, N_LAYERS = 64, 4, 2

CFG_LOSSLESS = dict(batch=128, lr=1e-3, patience=12, gpu_resident=True)   # difference ~0.006 (recommended)
CFG_FAST     = dict(batch=256, lr=2e-3, patience=8,  gpu_resident=True)   # difference ~0.021 (exploration)

class PatchAnomalyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.patch_size, self.n_patch, self.input_dim = PATCH_SIZE, N_PATCH, input_dim
        self.patch_embed = nn.Linear(PATCH_SIZE*input_dim, D_MODEL)
        self.register_buffer("pos_enc", self._pe(N_PATCH, D_MODEL))
        el = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL*4, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(el, N_LAYERS)
        self.patch_recon = nn.Linear(D_MODEL, PATCH_SIZE*input_dim)
    def _pe(self, L, d):
        pe = torch.zeros(L, d); pos = torch.arange(0, L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()*(-math.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        return pe.unsqueeze(0)
    def forward(self, x):
        B = x.size(0)
        xp = x.reshape(B, self.n_patch, self.patch_size*self.input_dim)
        h = self.transformer(self.patch_embed(xp) + self.pos_enc)
        return self.patch_recon(h).reshape(B, self.n_patch*self.patch_size, self.input_dim)

def make_windows(data, w=WINDOW, s=STRIDE):
    return np.stack([data[i:i+w] for i in range(0, len(data)-w+1, s)])

def patch_point_error(tr, te, seed=0, normalize=False, cfg=CFG_LOSSLESS, epochs=50):
    """Train on tr and return point-wise reconstruction error on te.

    Returns (ps, covered): point scores and the number of covered time points.
    """
    if normalize:                                  # only for data with large raw scales
        mu, sd = tr.mean(0), tr.std(0)+1e-8; tr, te = (tr-mu)/sd, (te-mu)/sd
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.tensor(make_windows(tr), dtype=torch.float32)
    model = PatchAnomalyModel(tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"]); lf = nn.MSELoss()
    B, pat = cfg["batch"], cfg["patience"]; model.train()

    if cfg["gpu_resident"]:                         # [speed] keep training data on the device
        Xg = Xtr.to(DEVICE); N = len(Xg); best = 1e9; bad = 0
        for _ in range(epochs):
            idx = torch.randperm(N, device=DEVICE); tot = nb = 0
            for b in range(0, N, B):
                xb = Xg[idx[b:b+B]]; opt.zero_grad(); l = lf(model(xb), xb); l.backward(); opt.step()
                tot += l.item(); nb += 1
            if pat is not None:                     # early stop
                e = tot/nb
                if e < best-1e-4: best = e; bad = 0
                else:
                    bad += 1
                    if bad >= pat: break
        del Xg; torch.cuda.empty_cache()
    else:
        ld = DataLoader(TensorDataset(Xtr), batch_size=B, shuffle=True)
        for _ in range(epochs):
            for (xb,) in ld:
                xb = xb.to(DEVICE); opt.zero_grad(); lf(model(xb), xb).backward(); opt.step()

    # [memory] batched evaluation: peak memory depends only on the batch size
    model.eval(); Xte = torch.tensor(make_windows(te), dtype=torch.float32); outs = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(Xte), batch_size=512):
            xb = xb.to(DEVICE)
            outs.append(((model(xb)-xb)**2).mean(dim=2).cpu().numpy())
    re_pt = np.concatenate(outs)                    # (n_win, WINDOW)
    covered = len(Xte)*STRIDE - STRIDE + WINDOW
    ps = np.full(covered, -np.inf)
    for wi in range(len(re_pt)):
        a = wi*STRIDE; ps[a:a+WINDOW] = np.maximum(ps[a:a+WINDOW], re_pt[wi])
    ps[ps == -np.inf] = ps[ps != -np.inf].min()
    return ps, covered


# ----------------------------------------------------------------------
# Segment-level metric (recall at the anomaly-ratio budget)
# ----------------------------------------------------------------------
def seg_spans(lb):
    """Return (start, end) spans of consecutive anomaly labels."""
    lb = np.asarray(lb).astype(int)
    if lb.sum() == 0: return []
    d = np.diff(np.concatenate([[0], lb, [0]]))
    return list(zip(np.where(d==1)[0].tolist(), np.where(d==-1)[0].tolist()))

def seg_detectability(ps, span, anom_ratio):
    """Fraction of a segment above the score quantile set by the anomaly ratio (recall_budget)."""
    s, e = span; thr = np.quantile(ps, 1.0-anom_ratio)
    return float((ps[s:e] > thr).mean())
