# Company A Telecom Customer Churn Analysis

## GCI World 2026 – Final Assignment

### Project Overview

This project analyses customer churn for Company A, a telecommunications company. The objective is to understand customer behaviour, identify factors associated with churn, develop machine learning models to predict customer churn, and propose a data-driven customer retention strategy.

The project uses exploratory data analysis (EDA), feature engineering, machine learning model comparison, model evaluation and business risk analysis.

The final objective is to help the business identify customers who may be at a higher risk of leaving and prioritise them for retention activities.

---

## 1. Project Objectives

The main objectives are:

1. Load, merge and validate the customer datasets.
2. Understand the distribution of churn and non-churn customers.
3. Identify missing values, unusual distributions and important customer characteristics.
4. Perform exploratory data analysis on numerical and categorical variables.
5. Create additional features to capture customer behaviour and usage patterns.
6. Train and compare six machine learning classification models.
7. Evaluate the selected model using appropriate classification metrics.
8. Identify important features associated with churn predictions.
9. Rank customers according to predicted churn risk.
10. Develop a business-oriented customer retention proposal.

---

## 2. Project Folder Structure

The project is organised into the following structure:

```text
Company_A_Telecom_Project/
│
├── analysis.ipynb
├── run_analysis.py
├── requirements.txt
├── README.md
│
├── telecom/
│   ├── Client.csv
│   └── Record.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── features.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── business.py
│   └── narrative.py
│
└── outputs/
    ├── figures/
    └── tables/
```

### Folder and File Descriptions

| File or folder | Purpose |
|---|---|
| `analysis.ipynb` | Interactive notebook containing the complete analysis workflow |
| `run_analysis.py` | Python script to execute the analysis from beginning to end |
| `requirements.txt` | Required Python libraries |
| `telecom/` | Contains the original Client and Record datasets |
| `src/config.py` | Stores project paths and configuration settings |
| `src/data.py` | Loads, merges and validates datasets |
| `src/eda.py` | Performs data profiling and exploratory data analysis |
| `src/features.py` | Creates engineered customer-behaviour features |
| `src/modeling.py` | Preprocesses data, trains and compares machine learning models |
| `src/evaluation.py` | Evaluates the selected classification model |
| `src/business.py` | Generates customer risk rankings and lift analysis |
| `src/narrative.py` | Generates business-oriented interpretations of model results |
| `outputs/figures/` | Stores generated visualisations |
| `outputs/tables/` | Stores generated analytical tables and summaries |

---

## 3. Dataset Description

The project uses two datasets:

### Client.csv

Contains customer-level information used to understand customer characteristics and account-related attributes.

### Record.csv

Contains additional customer information and variables used in the analysis.

The two datasets are loaded and merged using the customer identifier through the functions in `src/data.py`.

The merged dataset is used for exploratory analysis and machine learning.

**Target variable:** `churn`

The target variable represents whether a customer has churned or not. The modelling workflow uses this variable as the prediction target.

---

## 4. Tools and Technologies

The project is developed using Python and Jupyter Notebook in Visual Studio Code.

The main technologies and libraries include:

- Python
- Jupyter Notebook
- Visual Studio Code
- Pandas – data manipulation and analysis
- NumPy – numerical operations
- Matplotlib – data visualisation
- Seaborn – statistical visualisation
- Scikit-learn – preprocessing, machine learning and model evaluation

---

## 5. Project Workflow

### Step 1: Data Loading and Validation

The datasets are loaded using `src/data.py`.

The workflow includes:

- Loading Client.csv and Record.csv.
- Checking dataset dimensions.
- Merging the datasets.
- Validating the merged data.
- Examining the target variable distribution.

### Step 2: Data Profiling

The data profiling stage examines:

- Data types.
- Missing-value counts.
- Missing-value percentages.
- Number of unique values.
- Numerical descriptive statistics.

The data profile is saved in the output tables folder.

### Step 3: Exploratory Data Analysis

The EDA module examines customer behaviour and the relationship between customer characteristics and churn.

The analysis includes:

- Churn distribution.
- Missing-value analysis.
- Numerical feature distributions.
- Numerical feature comparisons by churn.
- Correlation analysis.
- Categorical feature comparisons.
- Scatter plots for selected numerical variables.

Selected original variables include:

- `rev_Mean`
- `mou_Mean`
- `custcare_Mean`
- `change_mou`
- `change_rev`
- `eqpdays`
- `months`

Only features available in the dataset are included in the relevant plots.

### Step 4: Feature Engineering

Additional customer-behaviour features are created in `src/features.py`.

The engineered features are intended to capture relationships and patterns that may not be directly represented by individual original variables.

The feature-engineering workflow includes:

- Revenue change relative to average revenue.
- Customer care usage relative to average usage.
- Usage change relative to average usage.
- Recent usage compared with six-month usage.
- Revenue per minute of usage.
- Equipment age groups.
- Customer tenure groups.

Ratios and derived variables are subject to data-quality checks, including handling zero denominators and invalid numerical values.

The engineered features are explored through additional distributions, churn comparisons and correlation analysis.

The modelling pipeline should apply the same feature-engineering logic when preparing the modelling data.

### Step 5: Machine Learning Model Development

The project compares six classification models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. AdaBoost
5. Gradient Boosting
6. HistGradientBoosting

The modelling workflow includes:

- Separating the target variable from predictor variables.
- Excluding the customer identifier from model features.
- Handling missing numerical and categorical values.
- Encoding categorical variables.
- Scaling numerical variables where required.
- Training and comparing classification models.

The model comparison table is generated using the evaluation metrics implemented in `src/modeling.py`.

### Step 6: Model Evaluation

The selected model is evaluated using the functions in `src/evaluation.py`.

The project considers the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision (PR-AUC)

The model comparison results are used to select the best-performing model according to the ranking implemented in the modelling module.

The decision threshold selected by the modelling workflow is also reported.

### Step 7: Feature Importance

Feature importance is examined when supported by the selected model and its fitted pipeline.

This analysis helps identify variables that contribute to the model's predictions.

Feature importance should be interpreted as a measure of predictive contribution, not as proof of causation.

### Step 8: Business Risk Analysis

The business analysis module uses model predictions to rank customers according to predicted churn risk.

The analysis includes:

- Customer risk ranking.
- Identification of the highest-risk customer group.
- Top 10% risk-group analysis.
- Lift calculation.
- Business interpretation of the results.

The objective is to help Company A prioritise retention resources towards customers with higher predicted churn risk.

---

## 6. Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or a compatible version supported by the installed dependencies.
- Visual Studio Code.
- Python extension for VS Code.
- Jupyter extension for VS Code.

### Step 1: Open the Project

Open the `Company_A_Telecom_Project` folder in Visual Studio Code.

### Step 2: Create a Virtual Environment

Open the VS Code terminal and run:

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

### Step 4: Select the Python Interpreter

In VS Code:

1. Open the Command Palette.
2. Select `Python: Select Interpreter`.
3. Choose the Python interpreter from the `.venv` environment.

---

## 7. Running the Project

There are two ways to execute the analysis.

### Option 1: Run the Python Script

From the project root directory, run:

```bash
python run_analysis.py
```

The script executes the complete workflow:

1. Data loading and validation.
2. Target distribution analysis.
3. Data profiling.
4. Original exploratory data analysis.
5. Engineered-feature analysis.
6. Six-model comparison.
7. Model evaluation.
8. Feature importance.
9. Customer risk ranking.
10. Business interpretation.

### Option 2: Run the Jupyter Notebook

Open:

```text
analysis.ipynb
```

in Visual Studio Code.

Select the correct Python kernel and execute the notebook cells sequentially.

The notebook contains the same major analytical stages as the Python script, with intermediate outputs displayed for easier inspection.

---

## 8. Expected Outputs

The analysis generates charts and tables under the `outputs/` directory.

### Figures

The figures folder contains visualisations generated by the EDA and evaluation modules, such as:

- Churn distribution.
- Missing-value analysis.
- Numerical feature distributions.
- Feature comparisons by churn.
- Correlation heatmaps.
- Categorical churn comparisons.
- Scatter plots.
- Engineered-feature visualisations.
- Model evaluation plots, where implemented.

### Tables

The tables folder contains analytical outputs generated by the project modules, such as:

- Data profile.
- Engineered-feature profile.
- Engineered-feature summary.
- Model comparison metrics.
- Model threshold results.
- Customer risk ranking, where implemented.

The exact filenames depend on the save functions implemented in the respective modules.

---

## 9. Business Objective and Proposed Application

The purpose of the churn prediction system is to support proactive customer retention.

The model can help Company A:

- Identify customers with higher predicted churn risk.
- Prioritise retention campaigns.
- Focus customer-service resources on high-risk segments.
- Develop targeted customer engagement strategies.
- Monitor customer retention performance over time.

Potential retention actions may include personalised offers, service-quality reviews, customer-care outreach and tailored engagement campaigns.

Any proposed intervention should be evaluated to determine whether it actually reduces churn.

---

## 10. Limitations and Considerations

The project has several limitations that should be considered when interpreting the results:

1. **Predictive versus causal analysis:** Associations between features and churn do not establish that a feature causes churn.
2. **Data quality:** Missing values, outliers and unusual distributions may affect model performance.
3. **Feature engineering:** Derived features depend on the accuracy and availability of their source variables.
4. **Model generalisation:** Holdout performance may differ from performance on future customers.
5. **Decision threshold:** The selected threshold involves trade-offs between precision and recall.
6. **Business implementation:** A high predicted churn risk does not guarantee that a customer will leave.
7. **Retention effectiveness:** The effectiveness of retention actions should be tested through an appropriate experiment or other credible evaluation method.

---

## 11. Conclusion

This project provides an end-to-end analytical workflow for understanding and predicting customer churn at Company A.

It combines data preparation, exploratory analysis, feature engineering, machine learning, model evaluation and customer risk ranking.

The final business objective is to translate predictive insights into actionable customer-retention priorities.

The results can support better allocation of retention resources and more informed customer engagement decisions.

---

## 12. Assignment Deliverables

The project is intended to support preparation of the following assignment deliverables:

- Business proposal and analytical findings.
- Exploratory data analysis.
- Feature engineering and machine learning implementation.
- Model comparison and evaluation.
- Business recommendations.
- Presentation/report in PDF format, following the assignment requirements.
- Jupyter Notebook and supporting Python source code.
- References and required generative AI conversation history URL.

**Note:** The final report and presentation should use the actual outputs generated by the completed analysis. Do not insert unverified model results or business claims.
