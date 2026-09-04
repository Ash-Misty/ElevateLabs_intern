# Task 5: Decision Trees and Random Forests

**AI & ML Internship — Elevate Labs**

---

## Objective
Learn tree-based models for classification and regression using the Heart Disease dataset. Covers decision tree training, visualization, overfitting analysis, random forest comparison, feature importance interpretation, and cross-validation.

## Dataset
**Dataset:** Heart Disease Dataset (Cleveland)  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
**Local file:** `dataset/heart_disease.csv`  
**Shape:** 303 rows × 14 columns

### Columns
| Column | Type | Description |
|--------|------|-------------|
| age | int | Age in years |
| sex | int | Sex (1 = male, 0 = female) |
| cp | int | Chest pain type (0-3) |
| trestbps | int | Resting blood pressure |
| chol | int | Serum cholesterol |
| fbs | int | Fasting blood sugar > 120 mg/dl |
| restecg | int | Resting ECG results |
| thalach | int | Maximum heart rate achieved |
| exang | int | Exercise induced angina |
| oldpeak | float | ST depression induced by exercise |
| slope | int | Slope of peak exercise ST segment |
| ca | int | Number of major vessels colored by fluoroscopy |
| thal | int | Thalassemia type |
| target | int | **Target** — presence of heart disease (0 = no, 1 = yes) |

## Technologies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Graphviz
- Jupyter Notebook

## Project Structure
```
task-5-decision-trees/
├── dataset/
│   └── heart_disease.csv
├── notebooks/
│   └── decision_trees.ipynb
├── screenshots/
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Preprocessing Steps

### 1. Data Preparation
- **Target binarization:** Convert multi-class target (0-4) to binary (0 = no disease, 1 = disease)
- **Feature scaling:** StandardScaler applied after train-test split to avoid leakage

### 2. Train-Test Split
- **Split:** 80% training, 20% testing
- **Random state:** 42
- **Stratify:** Yes, by target class

## Modeling

### Decision Tree Classifier
- **Algorithm:** `sklearn.tree.DecisionTreeClassifier`
- **Parameters:** Default and tuned via GridSearchCV
- **Visualization:** `plot_tree` with max_depth=3 for clarity

### Random Forest Classifier
- **Algorithm:** `sklearn.ensemble.RandomForestClassifier`
- **Parameters:** `n_estimators=100`, `random_state=42`
- **Method:** Bagging with random feature selection

## Evaluation Metrics

| Metric | Decision Tree | Random Forest |
|--------|---------------|---------------|
| Accuracy | ~75-85% | ~80-90% |
| Precision | ~75-85% | ~80-90% |
| Recall | ~75-85% | ~80-90% |
| F1-Score | ~75-85% | ~80-90% |
| ROC-AUC | ~0.80-0.90 | ~0.85-0.95 |

*Note: Exact values may vary based on preprocessing and random state.*

## Results

- **Random Forest outperforms single Decision Tree** in all metrics
- **Feature Importance:** Key predictors typically include `thalach`, `cp`, `ca`, `oldpeak`
- **Overfitting:** Decision trees overfit without depth limiting; optimal depth ~5-7
- **Cross-validation:** Provides robust performance estimates with low variance

## How to Run
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch Jupyter: `jupyter notebook`
4. Open `notebooks/decision_trees.ipynb` and run all cells sequentially

## Key Learnings
- Decision trees split data recursively using entropy/Gini impurity
- Overfitting is controlled via max_depth, min_samples_split, min_samples_leaf
- Random Forest reduces variance through bagging and random feature selection
- Feature importance reveals which features drive predictions
- Cross-validation provides robust performance estimates
- Tree-based models are invariant to feature scaling

## Interview Preparation

The notebook includes detailed Q&A for 8 common decision tree and random forest interview questions:

1. **How decision trees work:** Recursive binary splitting, entropy/Gini, stopping criteria
2. **Entropy and information gain:** Impurity measure, split quality, Gini alternative
3. **Random Forest vs single tree:** Bagging, variance reduction, ensemble benefits
4. **Overfitting:** Symptoms, prevention via pruning, cross-validation, ensembles
5. **Bagging:** Bootstrap sampling, aggregation, variance reduction
6. **Visualizing trees:** plot_tree, Graphviz, interpretation
7. **Feature importance:** Gini importance, permutation importance, limitations
8. **Random Forest pros/cons:** Accuracy, robustness, interpretability trade-offs

Each answer includes a simple explanation, technical details, a concrete example, and a possible follow-up question.
