import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os


st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme-aware CSS
st.markdown("""
<style>

/* ===== Global Background & Text Colors ===== */
[data-theme="light"] .stApp {
    background-color: #f0f2f6 !important;
    color: #1a1a1a !important;
}

[data-theme="dark"] .stApp {
    background-color: #0e1117 !important;
    color: #f0f0f0 !important;
}

/* ===== Sidebar Styling ===== */
[data-testid="stSidebar"] {
    padding-top: 1rem;
}

[data-theme="light"] [data-testid="stSidebar"] {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}

[data-theme="dark"] [data-testid="stSidebar"] {
    background-color: #262730 !important;
    color: #f0f0f0 !important;
}

/* ===== Buttons ===== */
.stButton>button {
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    border: none;
}

[data-theme="light"] .stButton>button {
    background-color: #4CAF50;
    color: white;
}

[data-theme="dark"] .stButton>button {
    background-color: #6fbf73;
    color: black;
}

/* ===== Titles & Headers ===== */
h1, h2, h3, h4 {
    font-weight: 600;
}

[data-theme="light"] h1, [data-theme="light"] h2, [data-theme="light"] h3 {
    color: #1a1a1a !important;
}

[data-theme="dark"] h1, [data-theme="dark"] h2, [data-theme="dark"] h3 {
    color: #f0f0f0 !important;
}

/* ===== Table Styling ===== */
[data-theme="light"] .stDataFrame {
    background-color: white;
}

[data-theme="dark"] .stDataFrame {
    background-color: #1c1e24;
}

</style>
""", unsafe_allow_html=True)



# --- Verify current directory (Good practice, keep it) ---
st.write("Current Directory:", os.getcwd())

# --- Load data and model with error handling ---
try:
    df = pd.read_csv('data/diabetes.csv')
    with open('notebooks/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('notebooks/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error: {e}. Please ensure 'notebooks/model.pkl', 'notebooks/scaler.pkl', and 'data/diabetes.csv' are in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# --- Main application UI layout ---
st.title("Diabetes Prediction App")
st.markdown("""
This app predicts diabetes risk using a machine learning model trained on the Pima Indians Diabetes Dataset.
Navigate through the sections to explore the data, visualize patterns, and make predictions.
""")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Exploration", "Visualizations", "Predictions", "Model Performance"])

if page == "Data Exploration":
    st.header("Data Exploration")
    st.markdown("---")
    st.info("Dataset Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Shape")
        st.write(df.shape)
    with col2:
        st.subheader("Columns and Data Types")
        st.write(df.dtypes)
    
    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Filter Data")
    age_range = st.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (20, 80))
    filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
    st.dataframe(filtered_df)

elif page == "Visualizations":
    st.header("Interactive Visualizations")
    st.markdown("---")
    
    # Using a container for a clean, bordered section
    with st.container():
        st.subheader("Feature Distribution")
        feature = st.selectbox("Select Feature for Histogram", df.columns[:-1])
        fig = px.histogram(df, x=feature, color='Outcome', title=f"{feature} Distribution by Outcome")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    with st.container():
        st.subheader("Scatter Plot")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis Feature", df.columns[:-1], index=1)
        with col2:
            y_axis = st.selectbox("Y-axis Feature", df.columns[:-1], index=5)
        fig2 = px.scatter(df, x=x_axis, y=y_axis, color='Outcome', title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Predictions":
    st.header("Make a Prediction")
    st.markdown("---")
    st.markdown("Enter patient details below to predict diabetes risk.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.slider("Glucose", 0, 200, 100)
            blood_pressure = st.slider("Blood Pressure", 0, 150, 70)
        with col2:
            skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
            insulin = st.slider("Insulin", 0, 900, 100)
            bmi = st.slider("BMI", 0.0, 70.0, 30.0)
        with col3:
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
            age = st.slider("Age", 0, 100, 30)
        
        st.markdown("---")
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            with st.spinner("Making prediction..."):
                try:
                    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0][1]
                    
                    if prediction == 1:
                        st.error(f"Prediction: Diabetes")
                        st.write(f"Probability of Diabetes: **{prob:.2%}**")
                    else:
                        st.success(f"Prediction: No Diabetes")
                        st.write(f"Probability of Diabetes: **{prob:.2%}**")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

elif page == "Model Performance":
    st.header("Model Performance")
    st.markdown("---")
    
    st.markdown("The following metrics are calculated on the entire dataset to give a general overview of the model's performance.")
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Accuracy")
        st.write(f"**{accuracy_score(y, y_pred):.2f}**")
    with col2:
        st.subheader("Precision")
        st.write(f"**{precision_score(y, y_pred):.2f}**")
    with col3:
        st.subheader("Recall")
        st.write(f"**{recall_score(y, y_pred):.2f}**")
    with col4:
        st.subheader("F1-Score")
        st.write(f"**{f1_score(y, y_pred):.2f}**")
        
    st.subheader("Confusion Matrix")
    st.markdown("A confusion matrix showing the number of correct and incorrect predictions.")
    
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

    st.pyplot(fig)
