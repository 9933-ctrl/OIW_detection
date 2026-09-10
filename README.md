# Reproduction package for the OIW detection method

This package contains multi-file Python code, configuration files, numerical figure data, and execution logs for reproducing the method described in the manuscript **"Deep Learning-based Oceanic Internal Wave Detection from SWOT Satellite Imaginary Observations"**.

## Important integrity note

The included scripts implement the stated hybrid CNN--Transformer detector with Spatial Reduction Attention (SRA), patch-level binary classification, augmentation, metric reporting, and figure recreation. 

## Main files

- `configs/method_config_article.json`: manuscript-aligned configuration.
- `src/oiw_repro/model.py`: hybrid CNN--Transformer model and SRA block.
- `src/oiw_repro/preprocess.py`: high-pass filtering, gradient channel construction, SNR noise injection, and rotations.
- `src/oiw_repro/dataset.py`: dataset loading and deterministic synthetic data generator.
- `src/oiw_repro/train.py`: training loop, metrics, checkpoints, and logs.
- `src/oiw_repro/evaluate.py`: evaluation from checkpoint.
- `scripts/run_demo_reproduction.py`: end-to-end deterministic smoke test.
- `scripts/recreate_published_figures.py`: recreates the numerical graph figures from CSV tables.
- `logs/environment_info.log`: Python/package/runtime information.
- `outputs/demo_train_metrics.csv`: metric log from the demo run.
- `outputs/demo_final_metrics.json`: final demo metrics.
- `outputs/checksums_sha256.txt`: SHA-256 checksums for verification.

## Expected input format for real data

Use one or more NumPy `.npz` files with:

- `x`: shape `(N, 3, H, W)`, float32. Channels should be SSH anomaly, high-pass filtered SSH, and local gradient magnitude.
- `y`: shape `(N,)`, int64 or int32. Label `1` means OIW-present patch and `0` means non-OIW/background/hard negative.
- optional metadata arrays: `patch_id`, `cycle`, `pass_id`, `region`, `source_file`, `timestamp`.

Run real training after replacing the paths in `configs/method_config_article.json`:

```bash
python -m oiw_repro.train --config configs/method_config_article.json
python -m oiw_repro.evaluate --config configs/method_config_article.json --checkpoint outputs/best_model.pt
```

## Recreate figures

The data is in `figure_plot` files


