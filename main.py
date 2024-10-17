import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from jcopml.pipeline import cat_pipe, num_pipe
from jcopml.plot import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_data(file):
    df = pd.read_excel(file)
    return df

def data_training(df):
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
    input_data = pd.DataFrame(
        {'jam': [jam],
         'ping': [ping],
         'jitter': [jitter],
         'packet_loss': [packet_loss],
         'speed_mbps': [speed_mbps]}
    )
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("Internet Traffic Prediction with Naive Bayes")
    st.sidebar.subheader('Data Training')
    file = st.sidebar.file_uploader(label='Pilih data training', type=('xlsx'))
    
    if st.sidebar.button('Training Model'):
        if file is not None:
            st.sidebar.write('File Uploaded')
            try:
                df_train = pd.read_excel(file)
                pipeline, X_train, X_test, y_train, y_test = data_training(df_train)
                st.success("Model telah dilatih dan disimpan.")
                st.subheader('Dataset untuk Training')
                st.write(df_train)

                y_pred_train = pipeline.predict(X_train)
                y_pred_test = pipeline.predict(X_test)
                
                show_confusion_matrices(y_train, y_pred_train, y_test, y_pred_test)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        else:
            st.warning("Silakan unggah file data training terlebih dahulu.")

    st.sidebar.subheader('Testing Model')
    test_file = st.sidebar.file_uploader(label='Pilih data testing', type=('xlsx'), key='test_file')

    if st.sidebar.button('Tampilkan Hasil Prediksi'):
        if test_file is not None:
            df_test = pd.read_excel(test_file)
            model = joblib.load('traffic_model.pkl')
            y_pred = model.predict(df_test)

            df_test['Prediksi'] = y_pred
            df_test['Label'] = df_test['Prediksi'].apply(lambda x: 'Tinggi' if x == 1 else 'Rendah')
            st.write("Hasil Prediksi:")
            st.write(df_test)

            test_file = st.session_state.get('test_file')
            df_test = pd.read_excel(test_file)
            X_test = df_test.drop(columns="traffic")
            y_test = df_test.traffic
            y_pred_test = model.predict(X_test)
            show_nb_testing(y_test, y_pred_test)

        else:
            st.warning("Silakan unggah file data testing terlebih dahulu.")

    st.subheader('Form Prediksi Traffic')
    form = st.form(key='prediction_form')
    jam = form.selectbox('Jam', [9, 13, 17])
    ping = form.slider('Ping (ms)', min_value=10, max_value=80, value=50)
    jitter = form.slider('Jitter (ms)', min_value=0, max_value=15, value=5)
    packet_loss = form.slider('Packet Loss (%)', min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    speed_mbps = form.slider('Kecepatan (Mbps)', min_value=10, max_value=80, value=40)

    submit = form.form_submit_button('Submit untuk Prediksi')
    
    if submit:
        model = joblib.load('traffic_model.pkl')
        prediction = predict_traffic(model, jam, ping, jitter, packet_loss, speed_mbps)
        st.write("Hasil Prediksi:")
        st.write("Traffic Tinggi" if prediction[0] == 1 else "Traffic Rendah")

if __name__ == "__main__":
    main()
