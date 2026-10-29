# Company A Telecom — modular churn analysis project

This project is designed to run locally in **VS Code with Jupyter Notebook**. It uses the two CSV files supplied with the GCI assignment and keeps the analysis split into reusable Python modules.

## Folder structure

```text
Company_A_Telecom_Project/
├── analysis.ipynb
├── run_analysis.py
├── requirements.txt
├── README.md
├── README_Assignment.docx
├── Company_A_Dataset_Overview.docx
├── original_tutorial.ipynb
├── final_assignment_tutorial.pdf
├── telecom/
│   ├── Client.csv
│   └── Record.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── business.py
│   └── narrative.py
└── outputs/
    ├── figures/
    └── tables/
```

## 1. Open in VS Code

1. Extract this ZIP.
2. In VS Code, select **File → Open Folder** and choose `Company_A_Telecom_Project`.
3. Install the Microsoft **Python** and **Jupyter** extensions if needed.
4. Open the VS Code terminal.

## 2. Create an environment and install packages

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, choose the `.venv` interpreter as the notebook kernel.

## 3. Run the notebook

Open `analysis.ipynb` and run the cells from top to bottom. The notebook imports functions from `src/`, while the modules contain the reusable logic.

## 4. Optional: run the complete pipeline from the terminal

```bash
python run_analysis.py
```

This runs data checks, exploratory summaries, two baseline models, evaluation plots, and a test-set risk ranking. Some model training may take a few minutes.

## Outputs

- `outputs/tables/data_profile.csv`: data types and missingness
- `outputs/tables/numeric_summary.csv`: numeric descriptive statistics
- `outputs/tables/model_comparison.csv`: model metrics
- `outputs/tables/feature_importance.csv`: Random Forest feature importance when selected
- `outputs/tables/test_set_risk_ranking.csv`: test-set risk ranking
- `outputs/figures/`: charts

## Method notes and cautions

- The primary target is `churn`, described in the supplied dataset overview as churn 31–60 days after the observation date. Verify its values and coding before interpreting class 1 as churn.
- `Customer_ID` is removed from the features; it is used only to merge the tables.
- Preprocessing is fit within each model pipeline, after the train/test split, to reduce data leakage.
- The notebook compares Logistic Regression and Random Forest. Use the holdout metrics to decide which model to discuss; don't claim one is best until you see the results.
- The top-risk-group analysis is retrospective. It does not prove that a retention campaign will prevent churn.
- Any financial impact calculator uses assumptions supplied by the analyst. Do not present illustrative assumptions as actual Company A financial data.
- Inspect feature definitions and consider leakage, proxy variables, privacy, and business actionability before making a final proposal.
- The assignment requires external market research and citations in the final slides. The supplied CSVs alone do not establish market size, customer lifetime value, intervention cost, or causal retention lift.
- The `narrative.py` functions draft plain-language summaries from measured results. Edit them to reflect your own interpretation and writing style; they are not an AI-detector bypass.

## Submission reminders from the supplied assignment README

The assignment asks for a PDF presentation of up to 15 slides, a reference list in Omnicampus, and a reproducible notebook or code ZIP. Include the model name, metric names and scores, explain the business implications, cite sources on slides, and include the chat URL if generative AI assistance was used. Confirm the latest instructions and deadline on Omnicampus.
