# Task 4: Classification with Logistic Regression

**AI & ML Internship — Elevate Labs**

---

## Objective
Build a binary classifier using logistic regression on the Breast Cancer Wisconsin dataset. Covers complete classification workflow: preprocessing, train-test split, standardization, model training, evaluation with confusion matrix/precision/recall/ROC-AUC, threshold tuning, and sigmoid explanation.

## Dataset
**Dataset:** Breast Cancer Wisconsin Dataset  
**Source:** [UCI ML Repository via GitHub](https://github.com/datasets/breast-cancer)  
**Local file:** `dataset/breast_cancer.csv`  
**Shape:** 272 rows × 10 columns

### Columns
| Column | Type | Description |
|--------|------|-------------|
| age | str | Age group |
| mefalsepause | str | Menopause status |
| tumor-size | str | Tumor size |
| inv-falsedes | str | Inv-nodes |
| falsede-caps | str | Capsular involvement |
| deg-malig | int | Degree of malignancy |
| breast | str | Breast side |
| breast-quad | str | Breast quadrant |
| irradiat | str | Radiation history |
| class | str | **Target** — recurrence or no-recurrence |

## Technologies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure
```
task-4-logistic-regression/
├── dataset/
│   └── breast_cancer.csv
├── notebooks/
│   └── logistic_regression.ipynb
├── screenshots/
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Preprocessing Steps

### 1. Data Encoding
- **Target (`class`):** Label Encoded (no-recurrence=0, recurrence=1)
- **Categorical features:** One-Hot Encoded using `pd.get_dummies()`
- **Numerical features:** Kept as-is (`deg-malig`)

### 2. Train-Test Split
- **Split:** 80% training, 20% testing
- **Random state:** 42
- **Stratify:** Yes, by target class to preserve class distribution

### 3. Standardization
- **Technique:** StandardScaler
- **Reasoning:** Logistic regression benefits from scaled features for solver convergence
- **Leakage prevention:** Scaler fitted only on training data

## Modeling

### Logistic Regression
- **Algorithm:** `sklearn.linear_model.LogisticRegression`
- **Parameters:** `max_iter=1000`, `random_state=42`
- **Output:** Binary classification probabilities via sigmoid function

## Evaluation Metrics

| Metric | Typical Value | Interpretation |
|--------|---------------|----------------|
| Accuracy | ~75-85% | Overall correctness |
| Precision | ~70-80% | Reliability of positive predictions |
| Recall | ~65-75% | Coverage of actual positives |
| F1-Score | ~70-80% | Balance of precision and recall |
| ROC-AUC | ~0.85-0.95 | Overall classification ability |

*Note: Exact values may vary based on preprocessing decisions.*

## Results

- **Model:** Logistic regression with standardized features
- **Performance:** Strong ROC-AUC indicating good class separation
- **Key insights:** 
  - Sigmoid function maps log-odds to probabilities
  - Threshold tuning allows optimizing for precision or recall
  - Confusion matrix reveals exact error breakdown
- **Visualizations:** Sigmoid curve, ROC curve, confusion matrix heatmap

## How to Run
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch Jupyter: `jupyter notebook`
4. Open `notebooks/logistic_regression.ipynb` and run all cells sequentially

## Key Learnings
- Logistic regression is the standard algorithm for binary classification
- Sigmoid function converts linear combinations to probabilities
- Precision and recall capture different aspects of performance
- ROC-AUC measures overall classification ability independent of threshold
- Confusion matrix reveals exact error types
- Threshold tuning optimizes for business-specific costs
- Always split data before preprocessing to avoid data leakage

## Interview Preparation

The notebook includes detailed Q&A for 8 common logistic regression interview questions:

1. **Logistic vs Linear Regression:** Output type, loss function, use case difference
2. **Sigmoid function:** Formula, properties, derivative, purpose
3. **Precision vs Recall:** Definitions, trade-off, when to prioritize each
4. **ROC-AUC curve:** TPR/FPR, threshold independence, interpretation
5. **Confusion matrix:** Structure, derived metrics, use cases
6. **Class imbalance:** Effects on accuracy, solutions (SMOTE, class weights, threshold tuning)
7. **Threshold selection:** Methods (Youden's J, F1-optimization, business-driven)
8. **Multi-class logistic regression:** Softmax, OvR, OvO approaches

Each answer includes a simple explanation, technical details, a concrete example, and a possible follow-up question.
