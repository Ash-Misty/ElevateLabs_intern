# Task 2: Exploratory Data Analysis (EDA)

**AI & ML Internship — Elevate Labs**

---

## Objective
Perform comprehensive Exploratory Data Analysis (EDA) on the Titanic dataset to uncover patterns, trends, relationships, and anomalies using descriptive statistics and visualizations. All insights are documented with interpretations to inform downstream preprocessing and modeling.

## Dataset
**Source:** [Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic/data)  
**Local file:** `dataset/titanic.csv`  
**Shape:** 891 rows × 12 columns

### Columns
| Column | Type | Description |
|--------|------|-------------|
| PassengerId | int | Unique passenger identifier |
| Survived | int | Target variable (0 = No, 1 = Yes) |
| Pclass | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Name | str | Passenger full name |
| Sex | str | Gender |
| Age | float | Age in years |
| SibSp | int | Siblings/Spouses aboard |
| Parch | int | Parents/Children aboard |
| Ticket | str | Ticket number |
| Fare | float | Passenger fare |
| Cabin | str | Cabin number |
| Embarked | str | Port of Embarkation (C, Q, S) |

## Technologies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Jupyter Notebook

## Project Structure
```
task-2-eda/
├── dataset/
│   └── titanic.csv
├── notebooks/
│   └── eda_notebook.ipynb
├── screenshots/
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```

## EDA Steps & Findings

### 1. Summary Statistics
- Generated descriptive statistics for all numerical and categorical features
- Identified missing values: Cabin (~77%), Age (~20%), Embarked (~0.2%)
- Observed skewness in Fare (highly right-skewed) and discrete distributions in SibSp/Parch

### 2. Univariate Analysis
- **Histograms:** Age is roughly normal; Fare is heavily right-skewed; SibSp/Parch are discrete
- **Boxplots:** Fare has extreme high outliers; SibSp/Parch have rare high values; Age has a few low outliers
- **Skewness:** Fare skewness ≈ 4.8; SibSp/Parch also highly skewed

### 3. Bivariate Analysis
- **Sex:** Females survived at ~74% vs males ~19% — strongest predictor
- **Pclass:** 1st class ~63% survival, 3rd class ~24% — strong class effect
- **Fare:** Higher fare correlates with higher survival (median survivor fare > median non-survivor)
- **Family size:** Large families (SibSp > 3, Parch > 2) had lower survival

### 4. Correlation Analysis
- **Fare vs Pclass:** r = -0.55 (strong negative) — higher class (lower number) means higher fare
- **Survived vs Fare:** r = +0.26 — positive correlation with survival
- **Survived vs Pclass:** r = -0.34 — negative correlation (higher class number = lower survival)
- **No severe multicollinearity** detected (|r| < 0.8 for all pairs)

### 5. Pairplot
- Fare shows clearest separation between survived and non-survived
- Age distributions overlap significantly between classes
- SibSp and Parch are discrete with limited separation power

### 6. Multivariate Analysis
- **Sex × Pclass interaction:** 1st class females had >90% survival; 3rd class males had ~13% survival
- This interaction is one of the most powerful predictors in the dataset

### 7. Anomaly Detection
- **IQR method:** Fare has most outliers; SibSp/Parch have rare high values
- **Z-score method:** Stricter; flags fewer Fare anomalies but more Age anomalies
- Decision: Fare outliers are real (luxury tickets) but may need transformation; Age outliers (elderly, infants) are meaningful and should be kept

### 8. Pattern Recognition
- **Key insight:** Gender and class are dominant survival predictors
- **Secondary insight:** Fare, family size, and embarkation port add predictive value
- **Actionable:** Preprocessing should preserve Sex and Pclass as strong features; consider log-transforming Fare; engineer FamilySize from SibSp/Parch

## Interview Preparation

The notebook includes detailed Q&A for 8 common EDA interview questions:

1. **Purpose of EDA:** Understand data structure, quality, and patterns before modeling
2. **Boxplots:** Show distributions, medians, quartiles, and outliers for quick comparison
3. **Correlation:** Measures linear relationships; used for feature selection and multicollinearity detection
4. **Skewness detection:** Histograms, statistical `skew()`, mean-vs-median comparison
5. **Multicollinearity:** High correlation between features causes unstable models; detected via correlation matrix or VIF
6. **EDA tools:** Pandas, Matplotlib, Seaborn, Plotly, Scipy, automated tools (Sweetviz, Pandas Profiling)
7. **Real-world EDA impact:** Example of finding negative tenure values that would have corrupted a churn model
8. **Visualization in ML:** Critical for EDA, model evaluation, interpretation, debugging, and stakeholder communication

Each answer includes a simple explanation, technical details, a concrete example, and a possible follow-up question.

## How to Run
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch Jupyter: `jupyter notebook`
4. Open `notebooks/eda_notebook.ipynb` and run all cells sequentially

## Key Learnings
- Always start with `df.info()` and `df.describe()` before visualization
- Histograms reveal distribution shape; boxplots reveal outliers and skewness
- Correlation matrices identify relationships and potential multicollinearity
- Pairplots show multi-feature interactions and class separation
- EDA findings must directly inform preprocessing and modeling decisions
- Visualization is essential for both technical analysis and stakeholder communication
