"""
Streamlit Web Application
=========================
This script creates a web interface for the Student Performance Prediction System.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="centered"
)

def load_model():
    """Load the trained model."""
    model_path = 'models/model.pkl'
    
    if not os.path.exists(model_path):
        st.error("Model not found! Please run train.py first to train the model.")
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_performance(model, study_hours, attendance, internal_marks, previous_score):
    """Make prediction using the loaded model."""
    # Create input array
    input_data = np.array([[study_hours, attendance, internal_marks, previous_score]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    return prediction, probability

def main():
    """Main function to run the Streamlit app."""
    
    # Title and header
    st.title("🎓 Student Performance Prediction System")
    st.markdown("---")
    
    # Sidebar with information
    st.sidebar.header("About")
    st.sidebar.info(
        "This system predicts whether a student will PASS or FAIL "
        "based on their study habits and academic performance."
    )
    
    st.sidebar.header("Features Used")
    st.sidebar.markdown(
        """
        - 📚 Study Hours
        - 📊 Attendance Percentage