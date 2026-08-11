# Data Professional Salary Prediction

**Project 8 — Stack Overflow Annual Developer Survey**

This repository implements the full salary prediction pipeline described in the project brief, covering data collection, cleaning, EDA, feature engineering, multi-model training, hyperparameter tuning, segment-level evaluation, and a Streamlit prediction app.

---

## Project Overview

| Item | Details |
|---|---|
| **Difficulty** | Intermediate |
| **Dataset** | Stack Overflow Annual Developer Survey 2023 (Salary subset) |
| **Dataset Source** | [Kaggle — Stack Overflow Developer Survey](https://www.kaggle.com/datasets/stackoverflow/stack-overflow-2023-developer-survey) |
| **Models** | Ridge Regression, Decision Tree Regressor, Random Forest Regressor |
| **Libraries** | pandas, numpy, matplotlib, seaborn, scikit-learn, plotly, joblib, streamlit |

### Business Objective

Predict annual salary as accurately as possible by comparing a linear baseline against tree-based and ensemble regressors, then recommend the model that performs most consistently across **junior, mid, and senior** experience segments.

---

## Project Workflow

```
Data Collection -> Data Cleaning -> EDA -> Feature Engineering -> Multi-Model Training -> Tuning -> Comparison & Recommendation
```

---

## Repository Structure

```
Data Professional Salary Prediction/
├── Notebook/
│   └── Notebook.ipynb          # Main analysis and modeling notebook (28 cells)
├── Dataset/
│   ├── survey_results_public.csv       # Raw Stack Overflow survey data
│   ├── survey_results_schema.csv       # Column descriptions
│   └── cleaned_salary_data.csv         # Cleaned, outlier-trimmed dataset
├── models/
│   ├── ridge.pkl                       # Base Ridge model
│   ├── decisiontree.pkl                # Base Decision Tree model
│   ├── randomforest.pkl                # Base Random Forest model
│   ├── best_ridge.pkl                  # Tuned Ridge (GridSearchCV)
│   ├── best_rf.pkl / best_model.pkl    # Tuned Random Forest (RandomizedSearchCV)
│   └── best_dt.pkl                     # Tuned Decision Tree (GridSearchCV)
├── images/
│   ├── salary_histogram.png            # Salary distribution (right-skew)
│   ├── salary_vs_yearscodepro.png      # Scatter: salary vs experience
│   ├── median_salary_by_yearsbin.png   # Binned bar chart: median salary per exp bin
│   ├── salary_by_top10_countries.png   # Boxplot: salary by top 10 countries
│   ├── salary_by_education.png         # Boxplot: salary by education level
│   └── correlation_heatmap.png         # Numeric feature correlation matrix
├── app.py                          # Streamlit salary prediction frontend
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Step-by-Step Implementation (Steps 81-90)

| Step | Description | Implementation |
|---|---|---|
| 81 | Load survey data, filter full-time + valid salary | Cell 1 loads CSV; Cell 6 applies full-time + salary filters |
| 82 | **Filter to data/AI-adjacent roles** | Cell 6: `is_data_ai_role()` filters DevType for Data Scientist, ML Engineer, Data Analyst, etc. |
| 83 | Remove extreme salary outliers (1st-99th percentile) | Cell 6: `quantile([0.01, 0.99])` cutoff |
| 84 | Explore salary vs YearsCodePro, Country, EdLevel | Cells 11-13: scatter, binned bar, boxplots |
| 85 | Bucket YearsCodePro -> experience_level (Junior/Mid/Senior) | Cell 15: `pd.cut(bins=[-0.1, 2.0, 7.0, 50.0])` |
| 86 | **Encode categorical columns incl. primary language** | Cell 15: one-hot encodes Country, EdLevel, Age, experience_level, **primary_language** |
| 87 | 80/20 train-test split stratified by experience_level | Cell 17: `train_test_split(..., stratify=experience_level)` |
| 88 | Train Ridge, Decision Tree, Random Forest | Cell 17: all three trained with `random_state=42` |
| 89 | Tune Random Forest with RandomizedSearchCV | Cell 19: tunes `n_estimators`, `max_depth`, `min_samples_leaf` |
| 90 | Model comparison table + deployment recommendation | Cell 24: per-model overall + segment table; Cell 23 recommendation |

---

## Data Cleaning Tasks

- Filter to **full-time employed** respondents only (`Employed, full-time` in Employment field)
- Remove rows with **missing or zero** `ConvertedCompYearly`
- Cap extreme salaries using the **1st-99th percentile** cutoff
- Filter to **data/AI-adjacent DevTypes**: Data Scientist, ML Engineer, Data Analyst, Data Engineer, Research Scientist, Engineer (data), Scientist
- Group rare countries as `"Other"` (top 20 countries retained)

## EDA (6 Visualizations)

1. `salary_histogram.png` — Overall salary distribution (right-skewed)
2. `salary_vs_yearscodepro.png` — Scatter: salary vs professional experience
3. `median_salary_by_yearsbin.png` — Binned bar chart: median salary per experience bin
4. `salary_by_top10_countries.png` — Boxplot: salary by top 10 countries
5. `salary_by_education.png` — Boxplot: salary by education level
6. `correlation_heatmap.png` — Numeric feature correlation matrix

**Key EDA Insights:**
- Salary is right-skewed; log-transformation is applied for modeling
- `YearsCodePro` is the strongest numeric predictor
- Geography (Country) is a critical signal — US/Western Europe salaries far exceed global median
- Higher education levels correlate with higher median salaries

## Feature Engineering Tasks

- **Primary Language** (Step 86): Extracted from `LanguageHaveWorkedWith` — priority-ranked (Python, SQL, JavaScript, R, Java, TypeScript, C++, Scala, Go, Julia → Other)
- **One-hot encoding**: Country (top 10 + Other), EdLevel, experience_level, Age, primary_language
- **Experience Level Bucket**: `Junior` (0-2 yrs), `Mid` (2-7 yrs), `Senior` (7+ yrs) from `YearsCodePro`
- **Log-transform target**: `log_salary = log1p(ConvertedCompYearly)` applied to reduce skewness
- **StandardScaler** applied to `YearsCodePro` for Ridge regression

## Model Building Tasks

| Model | Type | Tuning Method |
|---|---|---|
| Ridge | Linear baseline | GridSearchCV (alpha) |
| DecisionTreeRegressor | Tree-based | GridSearchCV (max_depth, min_samples_leaf, min_samples_split) |
| RandomForestRegressor | Ensemble (recommended) | RandomizedSearchCV (n_estimators, max_depth, min_samples_leaf) |

## Hyperparameter Tuning Tasks

- **Ridge**: `GridSearchCV` over `alpha=[0.1, 1.0, 10.0, 50.0, 100.0, 200.0]`, `cv=5`
- **Random Forest**: `RandomizedSearchCV`, `n_iter=12`, `cv=5`, `scoring='neg_root_mean_squared_error'`
- **Decision Tree**: `GridSearchCV` over `max_depth`, `min_samples_leaf`, `min_samples_split`, `cv=5`
- Best hyperparameters and cross-validated scores reported for each model

## Evaluation Tasks

- **Metrics**: R2, MAE, RMSE for all 5+ models on the full test set
- **Segmented Evaluation**: MAE & RMSE separately for Junior, Mid, Senior segments of test set
- **Predicted vs Actual** scatter plot colored by experience_level (interactive Plotly chart)
- **Feature Importance**: Ridge coefficients + Random Forest importances extracted and compared

## Model Comparison & Deployment Recommendation

- Comparison table: overall metrics + per-segment MAE/RMSE for all tuned models
- Final model recommendation: **RandomForest Tuned** (lowest RMSE, highest R2)
- Explicit discussion of segment-level error differences (Senior segment higher)
- Deployment caveat for hiring platform users documented

---

## Additional Requirement — Segment Fairness Check

> "Segment your evaluation by experience level (junior/mid/senior) and report whether model error is meaningfully different across segments."

The notebook computes per-segment MAE and RMSE for all tuned models (Cell 24) and explicitly discusses fairness/consistency in the markdown cell (Cell 25). The Senior segment shows meaningfully higher error — likely because salary variance grows with experience and senior roles are less homogeneous.

---

## Expected Outputs

| Output | Status |
|---|---|
| Cleaned, outlier-trimmed salary dataset | `Dataset/cleaned_salary_data.csv` |
| At least 5 EDA visualizations | 6 images in `images/` |
| Three or more tuned models with segment-level comparison | 3 tuned models in `models/` + full comparison table |
| Written recommendation addressing fairness/consistency | Cells 23 and 25 of notebook |

---

## Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download `survey_results_public.csv` from [Kaggle](https://www.kaggle.com/datasets/stackoverflow/stack-overflow-2023-developer-survey) and place it in the `Dataset/` folder.

### 3. Run the notebook

Open `Notebook/Notebook.ipynb` in Jupyter or VS Code and **Run All Cells** to:
- Clean and filter the data (including data/AI role filter)
- Generate all 6 EDA visualizations
- Engineer features including primary_language from LanguageHaveWorkedWith
- Train and tune all 3 models (Ridge, Decision Tree, Random Forest)
- Produce overall + segment-level comparison tables
- Save all model artifacts to `models/`

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open the local URL, select your Country, Years of Experience, Age, and Education Level, then click **Predict Salary**.

---

## Model Recommendation

**Recommended model: `RandomForest Tuned`**

- Lowest RMSE and highest R2 on the test set
- Better generalization than the base tree model
- Produces interpretable feature importances

**Deployment caveat**: This model is trained on self-reported survey data. Estimates are approximate, region-dependent, and may not capture employer-specific compensation components, stock/bonus packages, or sampling bias in the Stack Overflow developer community.

---

## License

Add a `LICENSE` file if you plan to publish this repository publicly.
