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
the following files in /models, /covariances.


### Synthetic FPR

Run:

```powershell
.\run_synthetic_fpr.bat
```

This reproduces the false-positive-rate experiments.

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

This reproduces the true-positive-rate experiments.

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
