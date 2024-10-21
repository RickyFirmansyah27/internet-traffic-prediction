import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import cat_pipe, num_pipe
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def load_data(file):
    """Load data from an Excel file."""
    df = pd.read_excel(file)
    return df

def data_training(df):
    """Train the model using the provided DataFrame."""
    X = df.drop(columns="traffic")
    y = df.traffic

    # Preprocessing (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', num_pipe(), ["jam", "ping", "jitter", "packet_loss", "speed_mbps"]),
        ]
    )
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('algo', GaussianNB())
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'traffic_model.pkl')

    return pipeline, X_train, X_test, y_train, y_test

def display_results(X_train, y_train, X_test, y_test, y_pred_train, y_pred_test):
    """Display training and testing results in Streamlit."""
    st.subheader('Data Training')
    st.write(X_train)

    st.subheader('Data Testing')
    st.write(X_test)

    st.subheader('Hasil Prediksi Training')
    results_train_df = pd.DataFrame({'Aktual': y_train, 'Prediksi': y_pred_train})
    st.write(results_train_df)

    st.subheader('Hasil Prediksi Testing')
    results_test_df = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred_test})
    st.write(results_test_df)

def show_confusion_matrices(y_train, y_pred_train, y_test, y_pred_test):
    """Display confusion matrices for training and testing results."""
    st.write('')
    st.subheader('Plot Confusion Matrices')

    # Create confusion matrices
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training Confusion Matrix
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Low Traffic', 'High Traffic'], yticklabels=['Low Traffic', 'High Traffic'])
    axes[0].set_xlabel('Prediksi')
    axes[0].set_ylabel('Aktual')
    axes[0].set_title('Data Training')

    # Testing Confusion Matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=['Low Traffic', 'High Traffic'], yticklabels=['Low Traffic', 'High Traffic'])
    axes[1].set_xlabel('Prediksi')
    axes[1].set_ylabel('Aktual')
    axes[1].set_title('Data Testing')

    # Display the plot
    st.pyplot(fig)
    plt.clf()

def show_nb_testing(y_test, y_pred_test):
    """Display confusion matrix for testing results."""
    st.write('')
    st.subheader('Confusion Matrix')

    # Create confusion matrix
    cm_test = confusion_matrix(y_test, y_pred_test)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Testing Confusion Matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Low Traffic', 'High Traffic'], yticklabels=['Low Traffic', 'High Traffic'])
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    ax.set_title('Confusion Matrix - Data Testing')

    # Display the plot
    st.pyplot(fig)
    plt.clf()

def predict_traffic(model, jam, ping, jitter, packet_loss, speed_mbps):
    """Predict traffic based on input features."""
    input_data = pd.DataFrame(
        {'jam': [jam],
         'ping': [ping],
         'jitter': [jitter],
         'packet_loss': [packet_loss],
         'speed_mbps': [speed_mbps]}
    )
    prediction = model.predict(input_data)
    return prediction
