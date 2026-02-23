"""
Model Training Script
=====================
This script trains and evaluates multiple ML models for student performance prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath='data/student_data.csv'):
    """Load the student performance dataset."""
    print("📂 Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df

def preprocess_data(df):
    """Preprocess the data for training."""
    print("\n🔧 Preprocessing data...")
    
    # Separate features and target
    X = df.drop('result', axis=1)
    y = df['result']
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target distribution:\n{y.value_counts()}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    """Train multiple machine learning models."""
    print("\n🤖 Training models...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"   ✓ {name} trained successfully")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models and compare performance."""
    print("\n📊 Evaluating models...")
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"\n   {name}:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Update best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name
        
        # Print classification report
        print(f"   Classification Report:")
        report = classification_report(y_test, y_pred, target_names=['Fail', 'Pass'])
        for line in report.split('\n'):
            print(f"      {line}")
    
    print(f"\n🏆 Best Model: {best_model} with {best_accuracy*100:.2f}% accuracy")
    
    return results, best_model

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot and save confusion matrix."""
    print(f"\n📈 Generating confusion matrix for {model_name}...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Pass'],
                yticklabels=['Fail', 'Pass'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150)
    plt.close()
    
    print("   ✓ Confusion matrix saved to models/confusion_matrix.png")

def plot_accuracy_comparison(results):
    """Plot accuracy comparison between models."""
    print("\n📊 Generating accuracy comparison chart...")
    
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=12)
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('models/accuracy_comparison.png', dpi=150)
    plt.close()
    
    print("   ✓ Accuracy comparison chart saved to models/accuracy_comparison.png")

def save_best_model(models, X_test, y_test, results):
    """Save the best performing model."""
    print("\n💾 Saving best model...")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = models[best_model_name]
    
    # Save model
    model_path = 'models/model.pkl'
    joblib.dump(best_model, model_path)
    print(f"   ✓ Best model ({best_model_name}) saved to {model_path}")
    
    return best_model_name, best_model

def main():
    """Main function to run the complete training pipeline."""
    print("=" * 60)
    print("   STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("   Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Step 3: Train models
    models = train_models(X_train, y_train)
    
    # Step 4: Evaluate models
    results, best_model_name = evaluate_models(models, X_test, y_test)
    
    # Step 5: Generate visualizations
    best_predictions = results[best_model_name]['predictions']
    plot_confusion_matrix(y_test, best_predictions, best_model_name)
    plot_accuracy_comparison(results)
    
    # Step 6: Save best model
    best_model_name, best_model = save_best_model(models, X_test, y_test, results)
    
    print("\n" + "=" * 60)
    print("   ✅ TRAINING COMPLETE!")
    print(f"   Best Model: {best_model_name}")
    print(f"   Model saved at: models/model.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()