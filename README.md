# Fetal Health Classification

Multiclass classification of fetal health from cardiotocography (CTG) records, comparing five
scikit-learn classifiers, with a Streamlit app that loads the saved models and evaluates them on
an uploaded CSV, without retraining.

**Live app:** https://fetal-health-ashish.streamlit.app/

## Problem

Each CTG record is labelled by expert obstetricians as **Normal (1)**, **Suspect (2)** or
**Pathological (3)**. The goal is to predict this label from 20 numeric features such as baseline
heart rate, accelerations, decelerations and heart-rate variability.

## Dataset

- Source: [Fetal Health Classification on Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
  (the raw dataset is not included in this repository)
- 2,126 records, 21 numeric features, target `fetal_health`
- Class balance: 1,655 Normal (77.85%), 295 Suspect (13.88%), 176 Pathological (8.28%); imbalance ratio 9.4
- No missing values

## Preprocessing

- Target converted to integers 1, 2, 3
- 13 exact duplicate rows removed (2,113 rows remain), so no row can appear in both train and test
- `histogram_width` dropped: it equals `histogram_max − histogram_min` in every row (20 features remain)
- Stratified 80/20 train/test split (1,690 / 423 rows), `random_state = 42`
- Scaling (`StandardScaler`) or power transform (`PowerTransformer`, Gaussian NB only) is part of
  each saved pipeline and was fitted on the training data only
- `class_weight="balanced"` for Logistic Regression, Decision Tree and SVM to counter class imbalance

## Models

| Model | Preprocessing in pipeline |
|---|---|
| Logistic Regression | StandardScaler |
| Decision Tree | none (trees are scale-invariant) |
| K-Nearest Neighbors (k = 5) | StandardScaler |
| Gaussian Naive Bayes | PowerTransformer (Yeo–Johnson) |
| SVM (RBF kernel, calibrated probabilities) | StandardScaler |

## Results on the held-out test set (423 records)

Precision, recall and F1 are **macro-averaged** so that each class counts equally; AUC is
**one-vs-rest, macro-averaged**; MCC uses the full 3 × 3 confusion matrix.

| Model | Accuracy | AUC (OvR, macro) | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.8676 | 0.9637 | 0.7526 | 0.8402 | 0.7869 | 0.6914 |
| Decision Tree | 0.9314 | 0.8938 | 0.9042 | 0.8646 | 0.8820 | 0.8065 |
| KNN | 0.9007 | 0.9527 | 0.8654 | 0.7502 | 0.7981 | 0.7089 |
| Gaussian NB | 0.7470 | 0.8503 | 0.6540 | 0.7273 | 0.6461 | 0.5356 |
| SVM | 0.8983 | 0.9604 | 0.8109 | 0.8268 | 0.8177 | 0.7226 |

**Observations**

- The Decision Tree is best on five of six metrics, but its AUC is the lowest of the top four
  models because its unpruned leaves output only 0/1 probabilities, which gives the ROC curve
  little to rank.
- KNN has higher accuracy than Logistic Regression but lower macro recall: without class
  weighting, the majority Normal class dominates its neighbourhood votes.
- Gaussian NB performs worst, as expected from strongly correlated histogram features that
  violate its independence assumption.
- Limitation: the Kaggle CSV has no recording or patient ID, so segments from the same recording
  could fall on both sides of the split; scores may be slightly optimistic.

## Repository structure

```
fetal-health-classifier/
├── app.py                      Streamlit app
├── requirements.txt            Pinned dependencies (scikit-learn 1.8.0)
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_nb.joblib
│   ├── svm.joblib              Each file = full pipeline (preprocessing + model)
│   ├── metadata.json           Feature order, class names, training settings
│   ├── test_set.csv            Held-out test set (20 features + fetal_health)
│   └── metrics_project_a.csv   Metrics table above
└── notebooks/
    └── project_a_training.ipynb   EDA, preprocessing, training, evaluation, saving
```

## Using the app

1. Choose a model in the sidebar.
2. Upload a CSV with the 20 feature columns (optionally with `fetal_health`), or tick
   **Use the bundled test set**. The bundled `test_set.csv` can be downloaded from the sidebar.
3. With labels present, the app shows all six metrics, a confusion matrix and a classification
   report; without labels, it shows predictions only. Predictions can be downloaded as CSV.
4. The **Model comparison** tab shows the saved results for all five models.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The saved models require **scikit-learn 1.8.0**, the version used for training.
