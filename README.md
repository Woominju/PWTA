# PWTA: Patch-Window Transformer Autoencoder for Multivariate Time-Series Anomaly Detection

Undergraduate thesis code — Minju Woo, Financial Engineering Lab, Hankuk University of Foreign Studies (advisor: Prof. Hyeongwoo Kong).

PWTA is a ~0.15M-parameter reconstruction model for multivariate time-series anomaly detection.
The main finding of the thesis is that, for this kind of model, **accuracy is driven by how the data is scored and evaluated, not by the architecture**:
37+ architecture and hyperparameter settings gave nearly the same accuracy, while changes on the data side
(channel weighting, input scaling, the evaluation metric) moved results far more.

## Key findings

**1. Channel scoring is the main lever.**
Anomalies usually appear in only a few channels, so averaging reconstruction errors over all channels dilutes them.
Weighting each channel by the kurtosis of its error (channels that are usually quiet and then spike) restores the signal.

| Scoring (PWTA backbone) | Macro PR-AUC (SMD · SMAP · MSL) |
|---|---|
| Plain channel mean | 0.470 |
| Kurtosis-weighted | **0.526** |

The gain needs sparse anomalies. On PSM, where 28% of points are anomalous, the weighting stops helping, and this is reported as a limitation.

**2. Keep the native scale; don't re-normalize.**
On SMAP and MSL, 42% and 77% of channels are near-constant command flags.
z-scoring divides them by a standard deviation close to zero and amplifies tiny deviations by up to ~10⁸, burying the real signal
(MSL PR-AUC 0.526 raw vs. 0.393 normalized). Constant channels are kept rather than dropped,
because many MSL anomalies are exactly a jump in one of them.

**3. Evaluate without point adjustment.**
Point-adjusted F1 counts a whole anomaly segment as detected if a single point is flagged, so random scores reach 0.83 PA-F1 (Kim et al., AAAI 2022).
All results here use raw PR-AUC. A gain only counts if it holds on every seed.

**4. The model is small and cheap.**
Measured with batch 128, forward + backward:

| Model | Parameters | Step time | Peak memory |
|---|---|---|---|
| **PWTA (this repo)** | **0.149M** | **4.8 ms** | **0.042 GB** |
| Anomaly Transformer | 4.83M | 23.2 ms | 1.95 GB |
| DCdetector | 0.90M | 104.4 ms | 9.63 GB |

## Repository contents

| File | Description |
|---|---|
| `pwta.py` | Full pipeline: PWTA model, training, kurtosis-weighted scoring, raw PR-AUC, optional routing to a channel-attention variant (`HybridPWTA`) |
| `patch_light.py` | Memory- and speed-optimized training/evaluation for the same patch model, with a segment-level recall metric |
| `notebooks/patch_light_demo.ipynb` | Demo of `patch_light.py` on three SMD machines |

### `patch_light.py`: faster without changing results
Only how training and evaluation run has changed; the model, scoring and metric are the same.

1. **Batched evaluation**: peak memory no longer grows with test length (SWaT: 2.88 GB → 0.04 GB, 71× lower), with bit-identical outputs.
2. **GPU-resident training data**: removes the CPU→GPU copy on every batch, about 2× faster.
3. **Early stopping**: shorter training with almost no change in accuracy.

Accuracy before and after (segment recall at the anomaly-ratio budget, `CFG_LOSSLESS`):

| Dataset | Entities | Original | Optimized | Mean \|diff\| |
|---|---|---|---|---|
| SMD | 28 | 0.388 | 0.387 | 0.006 |
| SMAP + MSL | 81 | 0.257 | 0.256 | 0.003 |
| PSM | 1 | 0.582 | 0.591 | 0.009 |
| SWaT | 1 | 0.105 | 0.105 | 0.000 |

Retraining all 112 entities × 3 seeds takes 7.6 minutes with 1.02 GB peak GPU memory.

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

Running `python pwta.py` trains on every SMD machine and compares kurtosis-weighted scoring with the plain mean.

## Notes and limitations
- Kurtosis weighting and the routing rule use statistics of the test inputs (never the labels), so the setting is transductive.
- Scores are produced for the time points covered by full windows; up to 9 points at the end of each test series are excluded, for all methods alike.
- The kurtosis-weighted channel fusion is developed further in a manuscript in preparation:
  S. Park†, M. Woo†, and H. Kong, *Adaptive Kurtosis-Weighted Channel Fusion for Multivariate Time-Series Anomaly Detection* († equal contribution).
