# Task 3: Linear Regression

**AI & ML Internship — Elevate Labs**

---

## Objective
Implement and understand simple and multiple linear regression using the California Housing dataset. This notebook covers the complete ML workflow: data loading, preprocessing, train-test splitting, model training, evaluation, visualization, and interpretation.

## Dataset
**Dataset:** California Housing Dataset  
**Source:** [Hands-On ML with Scikit-Learn, Keras & TensorFlow](https://github.com/ageron/handson-ml)  
**Local file:** `dataset/housing.csv`  
**Shape:** 20,640 rows × 10 columns

### Columns
| Column | Type | Description |
|--------|------|-------------|
| longitude | float | Longitude coordinate |
| latitude | float | Latitude coordinate |
| housing_median_age | float | Median age of houses in the block |
| total_rooms | float | Total number of rooms in the block |
| total_bedrooms | float | Total number of bedrooms in the block |
| population | float | Total population in the block |
| households | float | Total number of households in the block |
| median_income | float | Median income of households in the block |
| ocean_proximity | str | Proximity to the ocean (categorical) |
| median_house_value | float | **Target variable** — median house value |

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
task-3-linear-regression/
├── dataset/
│   └── housing.csv
├── notebooks/
│   └── linear_regression.ipynb
├── screenshots/
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Preprocessing Steps

### 1. Missing Value Handling
- **Column:** `total_bedrooms`
- **Strategy:** Median imputation
- **Reasoning:** Median is robust to outliers; preserves dataset size

### 2. Categorical Encoding
- **Column:** `ocean_proximity`
- **Technique:** One-Hot Encoding
- **Reasoning:** Nominal categories with no inherent order; one-hot avoids false ordinal assumptions

### 3. Train-Test Split
- **Split:** 80% training, 20% testing
- **Random state:** 42
- **Reasoning:** Prevents data leakage; ensures fair evaluation

## Modeling

### Simple Linear Regression
- **Feature:** `median_income`
- **Equation:** `median_house_value = β₀ + β₁ × median_income`
- **Purpose:** Build intuition with one predictor

### Multiple Linear Regression
- **Features:** All 9 features after preprocessing
- **Equation:** `median_house_value = β₀ + Σβᵢxᵢ`
- **Purpose:** Capture complex relationships with all available information

## Evaluation Metrics

| Metric | Simple LR (Test) | Multiple LR (Test) |
|--------|------------------|-------------------|
| MAE | ~$50,000 | ~$40,000 |
| MSE | ~$3.5B | ~$2.5B |
| R² | ~0.60 | ~0.65 |

*Note: Exact values may vary slightly due to randomness.*

## Results

- **Multiple regression outperforms simple regression** in all metrics
- **R² ≈ 0.65:** The model explains ~65% of house price variance
- **Key predictors:** `median_income`, `ocean_proximity_NEAR OCEAN/BAY`, `latitude`
- **Multicollinearity detected:** `total_rooms`, `total_bedrooms`, `population`, `households` are highly correlated
- **Residuals:** Approximately normal and centered at 0, supporting model assumptions

## How to Run
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch Jupyter: `jupyter notebook`
4. Open `notebooks/linear_regression.ipynb` and run all cells sequentially

## Key Learnings
- Linear regression models linear relationships between features and target
- Simple LR uses one feature; multiple LR uses many features
- R² measures explained variance; higher is better but not the only metric
- MAE and MSE measure prediction error; MSE penalizes large errors more
- Multicollinearity makes coefficients unstable but doesn't always hurt predictions
- Residual analysis validates model assumptions (linearity, normality, homoscedasticity)
- Always split data before preprocessing to avoid data leakage

## Interview Preparation

The notebook includes detailed Q&A for 8 common linear regression interview questions:

1. **Assumptions of linear regression:** Linearity, independence, homoscedasticity, normality, no multicollinearity, no endogeneity
2. **Interpreting coefficients:** β₁ = expected change in y per unit change in X₁, holding others constant
3. **R² score:** Proportion of variance explained; 1 = perfect, 0 = no explanatory power
4. **MSE vs MAE:** MSE squares errors (sensitive to outliers, differentiable); MAE uses absolute errors (robust)
5. **Detecting multicollinearity:** Correlation matrix, VIF > 5-10 indicates problem
6. **Simple vs multiple regression:** 1 feature vs many features; direct vs conditional relationships
7. **Linear regression for classification:** No — outputs continuous values; use logistic regression instead
8. **Violating assumptions:** Biased coefficients, wrong standard errors, poor predictions, unreliable inference

Each answer includes a simple explanation, technical details, a concrete example, and a possible follow-up question.
