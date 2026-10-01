# FraudGuard Model Card

- **Generated:** 2026-09-27T12:27:37+00:00
- **Version:** 1.0.0
- **Dataset:** Synthetic Kaggle-schema generator — 288,224 rows, 496 fraud (0.1721%)
- **Split:** 199,364 train / 28,481 valid / 56,962 test (stratified)
- **Balancing:** SMOTE on the training fold only, strategy 1.0 (398,042 rows after resampling)

## Primary model — XGBoost (SMOTE + scale_pos_weight)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8917 |
| PR-AUC (average precision) | 0.6530 |
| Operating threshold | 0.84 |
| Precision | 0.9483 |
| Recall | 0.5612 |
| F1 | 0.7051 |
| Specificity | 0.9999 |
| TP / FP / FN / TN | 55 / 3 / 43 / 56861 |

## All models (test fold, each at its own operating threshold)

| roc_auc | pr_auc | precision | recall | f1 | threshold |
| --- | --- | --- | --- | --- | --- |
| 0.8917 | 0.6530 | 0.9483 | 0.5612 | 0.7051 | 0.8400 |
| 0.9303 | 0.6219 | 0.8983 | 0.5408 | 0.6752 | 0.3100 |
| 0.9027 | 0.6328 | 0.7595 | 0.6122 | 0.6780 | 0.9500 |
| 0.8677 | 0.5130 | 0.9762 | 0.4184 | 0.5857 | 0.1400 |
| 0.8512 | 0.1453 | 0.0201 | 0.6633 | 0.0391 | 0.9500 |
| 0.7778 | 0.0167 | 0.0125 | 0.3469 | 0.0241 | 0.9500 |
| 0.8744 | 0.6778 | 0.9242 | 0.6224 | 0.7439 | 0.6900 |

## Intended use

Triage and rank incoming card transactions for manual review. The model is a decision
support signal, not an autonomous block/allow decision, and must be monitored for drift.

## Limitations

- The bundled dataset is a Kaggle-schema synthetic generator when `creditcard.csv` is absent;
  absolute metrics will differ from the published numbers on the real file.
- `Time` and `Amount` are scaled for distance-based detectors; the V1-V28 components are
  already PCA projections of the original anonymised features.
- SMOTE interpolates between fraud points; because real fraud is rare and non-compact,
  synthetic rows can drift off-manifold and cost precision.
- Class imbalance of ~0.17% means accuracy is uninformative; PR-AUC and recall at a fixed
  precision floor are the decision metrics.

## Environment

- **python**: 3.12.5
- **platform**: Windows-11-10.0.26200-SP0
- **scikit_learn**: 1.5.2
- **xgboost**: 3.4.1
- **numpy**: 2.1.3
- **pandas**: 2.2.3
