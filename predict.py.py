"""
Prediction Script
=================
This script loads the trained model and makes predictions on new student data.
"""

import joblib
import numpy as np
import pandas as pd

def load_model(model_path='models/model.pkl'):
    """Load the trained model from file."""
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("❌ Error: Model file not found. Please train the model first.")
        return None

def predict_student_performance(model, study_hours, attendance, internal_marks, previous_score):
    """
    Predict whether a student will pass or fail.
    
    Parameters:
    -----------
    model : trained model
        The loaded machine learning model
    study_hours : float
        Number of study hours per day
    attendance : float
        Attendance percentage (0-100)
    internal_marks : float
        Internal assessment marks (0-100)
    previous_score : float
        Previous exam score (0-100)
    
    Returns:
    --------
    tuple
        (prediction_result, probability)
    """
    # Create input array
    input_data = np.array([[study_hours, attendance, internal_marks, previous_score]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Get probability scores
    probabilities = model.predict_proba(input_data)[0]
    
    return prediction, probabilities

def predict_from_csv(model, csv_path):
    """
    Make predictions for multiple students from a CSV file.
    
    Parameters:
    -----------
    model : trained model
        The loaded machine learning model
    csv_path : str
        Path to CSV file with student data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions added
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Separate features
    X = df.drop('result', axis=1, errors='ignore')
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # Probability of passing
    
    # Add results to dataframe
    df['predicted_result'] = predictions
    df['pass_probability'] = np.round(probabilities * 100, 2)
    
    return df

def main():
    """Main function to demonstrate prediction functionality."""
    print("=" * 60)
    print("   STUDENT PERFORMANCE PREDICTION")
    print("   Prediction Demo")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    if model is None:
        return
    
    # Example predictions
    print("\n📝 Example Predictions:")
    print("-" * 60)
    
    test_cases = [
        {
            'study_hours': 8,
            'attendance': 95,
            'internal_marks': 85,
            'previous_score': 80,
            'description': "High performing student"
        },
        {
            'study_hours': 2,
            'attendance': 50,
            'internal_marks': 35,
            'previous_score': 40,
            'description': "At-risk student"
        },
        {
            'study_hours': 5,
            'attendance': 75,
            'internal_marks': 65,
            'previous_score': 60,
            'description': "Average student"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        prediction, probabilities = predict_student_performance(
            model,
            case['study_hours'],
            case['attendance'],
            case['internal_marks'],
            case['previous_score']
        )
        
        result = "PASS ✅" if prediction == 1 else "FAIL ❌"
        pass_prob = probabilities[1] * 100
        
        print(f"\nTest Case {i}: {case['description']}")
        print(f"   Study Hours: {case['study_hours']}")
        print(f"   Attendance: {case['attendance']}%")
        print(f"   Internal Marks: {case['internal_marks']}")
        print(f"   Previous Score: {case['previous_score']}")
        print(f"   🎯 Prediction: {result}")
        print(f"   📊 Pass Probability: {pass_prob:.1f}%")
    
    print("\n" + "=" * 60)
    print("   ✅ PREDICTION DEMO COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()