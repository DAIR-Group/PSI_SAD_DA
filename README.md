# Post-Selection Inference for Semi-Supervised Anomaly Detection after Domain Adaptation (PSI_SAD_DA)

This package contains the experimental implementation for the paper
"Post-Selection Inference for Semi-Supervised Anomaly Detection after Domain
Adaptation".

## Reproducibility

The synthetic experiment results stored under `results/` can be reproduced from
the repository root with the scripts below. The current synthetic results use
two saved DA-DeepSAD models:

- `deepsad_da_delta2_independent`: independent Gaussian synthetic data
  (`rho=0.0`)
- `deepsad_da_delta2_correlated`: AR(1)-correlated Gaussian synthetic data
  (`rho=0.5`)

### Environment

Create a fresh Python environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Required Artifacts

To reproduce the existing synthetic results without retraining the models, keep
the following files:

```text
models/deepsad_da_delta2_independent_model.pth
models/deepsad_da_delta2_independent_c.pth
models/deepsad_da_delta2_independent_metadata.json
models/deepsad_da_delta2_independent_source.npy
covariances/deepsad_da_delta2_independent_cov.npy
covariances/deepsad_da_delta2_independent_source_cov.npy

models/deepsad_da_delta2_correlated_model.pth
models/deepsad_da_delta2_correlated_c.pth
models/deepsad_da_delta2_correlated_metadata.json
models/deepsad_da_delta2_correlated_source.npy
covariances/deepsad_da_delta2_correlated_cov.npy
covariances/deepsad_da_delta2_correlated_source_cov.npy
```


### Synthetic FPR

Run:

```powershell
.\run_synthetic_fpr.bat
```

This reproduces the false-positive-rate experiment for both covariance
settings:

- independent data: `rho=0.0`
- correlated data: `rho=0.5`
- source sample sizes: `100, 150, 200, 250`
- target test size: `50`
- anomaly rate during evaluation: `0.00`
- number of random seeds: `500`
- methods: `proposed`, `wo_ad`, `wo_da`, `oc`, `bonferroni`, `naive`

Outputs are written to:

```text
results/synthetic_fpr/independent_data/
results/synthetic_fpr/correlated_data/
```

The final figures are:

```text
results/synthetic_fpr/independent_data/final_fpr_plot.pdf
results/synthetic_fpr/independent_data/final_fpr_plot.png
results/synthetic_fpr/correlated_data/final_fpr_plot.pdf
results/synthetic_fpr/correlated_data/final_fpr_plot.png
```

### Synthetic TPR

Run:

```powershell
.\run_synthetic_tpr.bat
```

This reproduces the true-positive-rate experiment for both covariance settings:

- independent data: `rho=0.0`
- correlated data: `rho=0.5`
- total test size: `300`
- source test size: `200`
- target test size: `100`
- anomaly shifts: `0.5, 1.0, 1.5, 2.0`
- anomaly rate during evaluation: `0.05`
- number of random seeds: `500`
- methods: `proposed`, `oc`, `bonferroni`, `naive`

Outputs are written to:

```text
results/synthetic_tpr/independent_data/
results/synthetic_tpr/correlated_data/
```

The final figures are:

```text
results/synthetic_tpr/independent_data/final_tpr_plot.pdf
results/synthetic_tpr/independent_data/final_tpr_plot.png
results/synthetic_tpr/correlated_data/final_tpr_plot.pdf
results/synthetic_tpr/correlated_data/final_tpr_plot.png
```
