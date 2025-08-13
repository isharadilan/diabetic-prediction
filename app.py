import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

# --- Set page config for a wider layout and cleaner look ---
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for modern styling ---
# We use st.markdown with unsafe_allow_html=True to inject CSS.
# This gives us full control over the app's appearance.
st.markdown(
    """
    <style>
    /* General body styling for a clean, light theme */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Style the sidebar for a more integrated feel */
    .st-emotion-cache-16txte5 {
        background-color: #ffffff;
        padding-top: 1rem;
        padding-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .st-emotion-cache-16txte5 h2 {
        color: #333333;
        font-weight: 600;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }

    /* Main app container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }

    /* Main Title Styling */
    h1 {
        color: #1a1a1a;
        text-align: left;
        font-weight: 800;
        font-size: 2.5rem;
    }
    
    /* Header Styling */
    h2, h3 {
        color: #4CAF50;
        font-weight: 600;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }

    /* Markdown Styling */
    .st-emotion-cache-1g8i73 {
        color: #555555;
        font-size: 1.1rem;
    }
    
    /* Styling for buttons with rounded corners and hover effect */
    .st-emotion-cache-l33f8h {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .st-emotion-cache-l33f8h:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Style for input widgets (sliders, number inputs) */
    .stSlider > label, .stNumberInput > label {
        font-weight: 500;
        color: #333333;
    }

    /* Style the dataframes for better readability */
    .stDataFrame > div {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Style plotly charts for a cleaner look */
    .st-emotion-cache-16x881k {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1rem;
    }
    
    /* Fix for matplotlib figures to have a background */
    .st-emotion-cache-f1g0i0 img {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    </style>
    """,
    unsafe_allow_html=True
)

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