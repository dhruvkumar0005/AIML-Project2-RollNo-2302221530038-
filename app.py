import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join('models', 'best_model.pkl')
CLEANED_PATH = os.path.join('Dataset', 'cleaned_salary_data.csv')
AGE_CATEGORIES = [
    'Under 18 years old',
    '18-24 years old',
    '25-34 years old',
    '35-44 years old',
    '45-54 years old',
    '55-64 years old',
    '65 years or older',
    'Prefer not to say',
]


def parse_experience(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value.lower().startswith('less than'):
            return 0.5
        if value.lower().startswith('more than'):
            return 50.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at: {model_path}')
    model = joblib.load(model_path)
    if not hasattr(model, 'feature_names_in_'):
        raise AttributeError('Model does not contain feature_names_in_.')
    return model


def get_model_metadata(model):
    feature_columns = list(model.feature_names_in_)
    countries = [c.replace('Country_', '') for c in feature_columns if c.startswith('Country_') and c != 'Country_Other']
    ed_levels = [c.replace('EdLevel_', '') for c in feature_columns if c.startswith('EdLevel_')]
    ages = [c.replace('Age_', '') for c in feature_columns if c.startswith('Age_')]
    return feature_columns, sorted(countries), sorted(ed_levels), sorted(ages)


def preprocess_input(input_data, feature_columns, scaler_mean, scaler_std, valid_countries):
    df = pd.DataFrame([input_data])
    df['YearsCodePro'] = df['YearsCodePro'].map(parse_experience)
    df['Country'] = df['Country'].fillna('Other')
    df['Country'] = df['Country'].where(df['Country'].isin(valid_countries), 'Other')
    df['EdLevel'] = df['EdLevel'].fillna('Missing')
    df['Age'] = df['Age'].fillna('Missing')
    df['experience_level'] = pd.cut(df['YearsCodePro'], bins=[-0.1, 2.0, 7.0, 50.0], labels=['Junior', 'Mid', 'Senior']).astype(str)

    X = pd.get_dummies(df[['Country', 'EdLevel', 'experience_level', 'Age']], prefix=['Country', 'EdLevel', 'experience_level', 'Age'], prefix_sep='_')
    X['YearsCodePro'] = df['YearsCodePro']

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    X['YearsCodePro'] = (X['YearsCodePro'] - scaler_mean) / scaler_std
    return X


def main():
    st.set_page_config(page_title='Salary Prediction App', page_icon='📊')
    st.title('Salary Prediction Frontend')
    st.write('This app uses the saved model from the notebook to predict salary based on survey-style inputs.')

    try:
        model = load_model()
    except Exception as exc:
        st.error(str(exc))
        return

    feature_columns, countries, ed_levels, ages = get_model_metadata(model)

    country = st.selectbox('Country', ['Other'] + countries)
    years_code_pro = st.slider('Years of Professional Coding Experience (YearsCodePro)', 0.0, 50.0, 5.0, 0.5)
    age = st.selectbox('Age', AGE_CATEGORIES)
    ed_level = st.selectbox('Education Level', ed_levels)

    input_data = {
        'Country': country,
        'YearsCodePro': years_code_pro,
        'EdLevel': ed_level,
        'Age': age,
    }

    if st.button('Predict Salary'):
        scaler_mean = 0.0
        scaler_std = 1.0
        # Estimate scaler from feature names if available in the notebook data
        if os.path.exists(CLEANED_PATH):
            cleaned = pd.read_csv(CLEANED_PATH)
            cleaned['YearsCodePro'] = cleaned['YearsCodePro'].map(parse_experience)
            cleaned = cleaned.dropna(subset=['YearsCodePro'])
            feature_df = cleaned.copy()
            feature_df['Country'] = feature_df['Country'].fillna('Other')
            feature_df['Country'] = feature_df['Country'].where(feature_df['Country'].isin(countries), 'Other')
            if 'EdLevel' not in feature_df.columns:
                feature_df['EdLevel'] = 'Missing'
            else:
                feature_df['EdLevel'] = feature_df['EdLevel'].fillna('Missing')
            feature_df['Age'] = feature_df['Age'].fillna('Missing')
            feature_df['experience_level'] = pd.cut(feature_df['YearsCodePro'], bins=[-0.1, 2.0, 7.0, 50.0], labels=['Junior', 'Mid', 'Senior']).astype(str)
            matrix = pd.get_dummies(feature_df[['Country', 'EdLevel', 'experience_level', 'Age']], drop_first=True)
            matrix['YearsCodePro'] = feature_df['YearsCodePro']
            scaler_mean = matrix['YearsCodePro'].mean()
            scaler_std = matrix['YearsCodePro'].std() + 1e-9

        X = preprocess_input(input_data, feature_columns, scaler_mean, scaler_std, countries)
        prediction_log = model.predict(X)
        prediction_salary = np.expm1(prediction_log)[0]
        st.metric('Predicted Annual Salary', f'${prediction_salary:,.0f}')
        st.success('Model prediction complete.')

    st.markdown('---')
    st.write('Model artifact used:')
    st.write(f'`{MODEL_PATH}`')
    st.write('The backend logic is aligned with the notebook preprocessing and feature engineering.')


if __name__ == '__main__':
    main()
