# PWTA: Patch-Window Transformer Autoencoder for Multivariate Time-Series Anomaly Detection

Code for my undergraduate thesis (Financial Engineering Lab, Hankuk University of Foreign Studies, advisor: Prof. Hyeongwoo Kong).

PWTA is a small reconstruction model (about 0.15M parameters) for multivariate time-series anomaly detection.
I tried 37+ architecture and hyperparameter settings and the accuracy hardly changed.
What did change the results was on the data side: how channel errors are combined, how the input is scaled, and which metric is used.

## Results

### 1. Weighting channels by kurtosis
An anomaly usually affects only a few channels, so taking the mean of the reconstruction error over all channels makes it weaker.
I weight each channel by the kurtosis of its error, which gives more weight to channels that are usually quiet and then suddenly spike.

| Scoring (same PWTA model) | Macro PR-AUC (SMD, SMAP, MSL) |
|---|---|
| Mean over channels | 0.470 |
| Kurtosis-weighted | 0.526 |

This only works when anomalies are rare. On PSM, where 28% of the points are anomalies, it doesn't help.

### 2. No re-normalization
On SMAP and MSL, 42% and 77% of the channels are almost constant.
z-score normalization divides these channels by a standard deviation close to 0, which blows up tiny changes (up to about 10^8) and hides the real signal.
On MSL, PR-AUC was 0.526 with the raw data and 0.393 after normalization.
I also kept the constant channels, because many MSL anomalies are a jump in one of them.

### 3. No point adjustment
Point-adjusted F1 counts a whole anomaly segment as detected if one point in it is flagged, so even random scores get 0.83 (Kim et al., AAAI 2022).
All results here use raw PR-AUC, and I only counted an improvement if it held on every seed.

### 4. Model size and speed
Batch size 128, forward + backward:

| Model | Parameters | Step time | Peak memory |
|---|---|---|---|
| PWTA | 0.149M | 4.8 ms | 0.042 GB |
| Anomaly Transformer | 4.83M | 23.2 ms | 1.95 GB |
| DCdetector | 0.90M | 104.4 ms | 9.63 GB |

## Files

| File | Description |
|---|---|
| `pwta.py` | PWTA model, training, kurtosis-weighted scoring, raw PR-AUC, and optional routing to a channel-attention variant (`HybridPWTA`) |
| `patch_light.py` | Faster training and evaluation for the same model, plus a segment-level recall metric |
| `notebooks/patch_light_demo.ipynb` | Runs `patch_light.py` on three SMD machines |

### patch_light.py
The model, scoring and metric are the same as before. I only changed how training and evaluation run:

1. Evaluation in batches. Before, the whole test set went to the GPU at once and long series ran out of memory. Peak memory on SWaT went from 2.88 GB to 0.04 GB and the outputs are identical.
2. Training data kept on the GPU, so it isn't copied every batch. About 2x faster.
3. Early stopping.

Segment recall before and after (`CFG_LOSSLESS`):

| Dataset | Entities | Original | Optimized | Mean abs. difference |
|---|---|---|---|---|
| SMD | 28 | 0.388 | 0.387 | 0.006 |
| SMAP + MSL | 81 | 0.257 | 0.256 | 0.003 |
| PSM | 1 | 0.582 | 0.591 | 0.009 |
| SWaT | 1 | 0.105 | 0.105 | 0.000 |

Retraining all 112 entities with 3 seeds takes 7.6 minutes, with 1.02 GB peak GPU memory.

## Usage

```bash
pip install -r requirements.txt
```

Download the Server Machine Dataset (SMD) from [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) and set the `SMD` path in `pwta.py`.

```python
import numpy as np
from pwta import anomaly_score, pr_auc

root = "OmniAnomaly/ServerMachineDataset"
tr = np.loadtxt(f"{root}/train/machine-1-1.txt", delimiter=",").astype(np.float32)
te = np.loadtxt(f"{root}/test/machine-1-1.txt", delimiter=",").astype(np.float32)
lb = np.loadtxt(f"{root}/test_label/machine-1-1.txt", delimiter=",").astype(int)

score, covered = anomaly_score(tr, te, seed=0, sharp="soft")
print(pr_auc(score, lb))
```

`python pwta.py` trains on every SMD machine and compares kurtosis weighting with the plain mean.

## Notes
- Kurtosis weighting and the routing rule use statistics of the test inputs (not the labels), so the setting is transductive.
- Scores only cover time points inside full windows, so up to 9 points at the end of each test series are left out. This is the same for every method.
- The kurtosis-weighted fusion is studied further in a manuscript in preparation: S. Park\*, M. Woo\*, and H. Kong, Adaptive Kurtosis-Weighted Channel Fusion for Multivariate Time-Series Anomaly Detection (\*equal contribution).
