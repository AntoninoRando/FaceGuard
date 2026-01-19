# FaceGuard Evaluation Results

This directory contains the evaluation metrics for the FaceGuard biometric system.

## Generating Metrics

To compute real system metrics, run the evaluation script:

```bash
python src/evaluate_metrics.py --gallery_path embeddings/gallery.pkl --test_path data/probes_for_test --output_dir results
```

### Arguments:
- `--gallery_path`: Path to the gallery file (default: `data/initial_samples`)
- `--test_path`: Path to test probe images (default: `data/probes_for_test`)
- `--output_dir`: Directory to save results (default: `results`)
- `--threshold`: Optional identification threshold override

## Output Files

After running the evaluation, the following files will be generated:

### `metrics_report.json`
Comprehensive JSON file containing:
- **Accuracy**: Overall identification accuracy
- **FAR/FRR**: False Acceptance/Rejection Rates
- **EER**: Equal Error Rate
- **ROC Curve Data**: Full ROC curve with AUC
- **DET Curve Data**: Detection Error Tradeoff curve
- **CMC Curve Data**: Cumulative Match Characteristic
- **Confusion Matrix**: TP, TN, FP, FN counts
- **Score Distributions**: Genuine vs impostor score histograms
- **Detailed Metrics**: All performance indicators with descriptions

### Visualization Plots
- `roc_curve.png`: ROC curve visualization
- `det_curve.png`: DET curve visualization
- `cmc_curve.png`: CMC curve visualization

## API Integration

The `/metrics` API endpoint automatically reads from `metrics_report.json` to provide real system performance data. If this file doesn't exist, the API will return a message instructing you to run the evaluation first.

## Notes

- Metrics are computed using the actual gallery and test probe images
- The test set should be separate from the gallery (enrollment) set
- For accurate metrics, ensure you have sufficient test samples per identity
- Re-run evaluation after making changes to the system or gallery
