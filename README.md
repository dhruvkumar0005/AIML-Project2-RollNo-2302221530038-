# Data Professional Salary Prediction

This repository contains a Jupyter notebook and Streamlit app for predicting yearly compensation using cleaned Stack Overflow survey data. The project covers data cleaning, exploratory analysis, feature engineering, model training, hyperparameter tuning, evaluation, and deployment.

## Repository structure

- `Notebook/Notebook.ipynb`: Main analysis and modeling notebook.
- `Dataset/cleaned_salary_data.csv`: Cleaned dataset used for modeling and app preprocessing.
- `models/best_model.pkl`: Best tuned model saved by the notebook (`RandomForest Tuned`).
- `images/`: Saved EDA and evaluation figures.
- `requirements.txt`: Python dependencies.
- `app.py`: Streamlit prediction frontend.

## Setup

1. Open a terminal in the repository root.
2. Create and activate a Python environment.
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the notebook

1. Open `Notebook/Notebook.ipynb` in Jupyter or VS Code.
2. Run all cells in order to reproduce the analysis and saved outputs.
3. The notebook generates:
   - cleaned dataset
   - model training and tuning
   - comparison of tuned models
   - overall and segment-level evaluation
   - saved model artifact: `models/best_model.pkl`
   - saved figures in `images/`

## Run the Streamlit app

1. Ensure the virtual environment is active.
2. Start the app:
   ```bash
   streamlit run app.py
   ```
3. Open the local URL shown in the terminal.
4. Use the form to select `Country`, `YearsCodePro`, `Age`, and `Education Level`, then click `Predict Salary`.

## Model comparison and selection

The notebook evaluates multiple models and tuned versions, including:
- `Ridge`
- `DecisionTree`
- `RandomForest`
- `Ridge Tuned`
- `DecisionTree Tuned`
- `RandomForest Tuned`

The best model selected by the notebook is **`RandomForest Tuned`**, based on the lowest RMSE and highest R² on the test set.

## Segment-level evaluation

The notebook also compares performance across experience segments:
- `Junior`
- `Mid`
- `Senior`

This helps identify where the model performs well and where prediction error is larger.

## Notes

- `RandomForest Tuned` is the deployed model because it provides the best overall accuracy and generalization.
- The notebook includes segment-level performance to show reliability across experience groups.
- If you prefer not to publish survey data, remove `Dataset/cleaned_salary_data.csv` and add data download instructions.

## License

Add a `LICENSE` file if you plan to publish this repository publicly.
