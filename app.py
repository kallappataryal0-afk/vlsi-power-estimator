import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

st.set_page_config(page_title="VLSI Power Estimator", layout="wide")
st.title("Digital Circuit Power Estimation using Machine Learning")
st.markdown("Upload your circuit dataset to predict power consumption using trained ML models.")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded: {df.shape[0]} circuits, {df.shape[1]} features")
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Select ML Model")
    model_choice = st.selectbox("Choose a model:", 
        ['Ridge Regression', 'Linear Regression', 'Random Forest', 'MLP Regressor', 'XGBoost']
    )
    
    target_col = st.selectbox("Select target power metric:", ['Total_Power_mW', 'Dynamic_Power_mW', 'Leakage_Power_mW'])
    
    if st.button("Train Model & Predict"):
        with st.spinner("Training model..."):
            df_proc = df.copy()
            
            if 'Circuit_Type' in df_proc.columns:
                le = LabelEncoder()
                df_proc['Circuit_Type'] = le.fit_transform(df_proc['Circuit_Type'].astype(str))
            
            if 'Circuit_ID' in df_proc.columns:
                df_proc = df_proc.drop('Circuit_ID', axis=1)
            
            power_cols = ['Dynamic_Power_mW', 'Leakage_Power_mW', 'ShortCircuit_Power_mW', 'Glitch_Power_mW', 'Total_Power_mW']
            X = df_proc.drop([c for c in power_cols if c in df_proc.columns], axis=1)
            y = df_proc[target_col]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if model_choice == 'Ridge Regression':
                model = Ridge(alpha=0.05)
            elif model_choice == 'Linear Regression':
                model = LinearRegression()
            elif model_choice == 'Random Forest':
                model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
            elif model_choice == 'XGBoost':
                model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
            elif model_choice == 'MLP Regressor':
                model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
            
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.4f} mW")
            col2.metric("R2 Score", f"{r2:.6f}")
            col3.metric("Accuracy", f"{r2*100:.4f}%")
            
            st.subheader("Actual vs Predicted Plot")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y, y_pred, alpha=0.5, s=20, color='steelblue')
            min_val, max_val = y.min(), y.max()
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel(f'Actual {target_col}')
            ax.set_ylabel(f'Predicted {target_col}')
            ax.set_title(f'{model_choice}: Actual vs Predicted')
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
            st.pyplot(fig)
            
            st.subheader("Prediction Results")
            results_df = pd.DataFrame({
                f'Actual_{target_col}': y,
                f'Predicted_{target_col}': y_pred,
                'Error': y - y_pred
            })
            st.dataframe(results_df)
