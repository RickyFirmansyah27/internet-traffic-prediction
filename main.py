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
    # Split data into training and testing (75% train, 25% test)
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

    # Save the model
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
    axes[0].set_title('Confusion Matrix - Data Training')

    # Testing Confusion Matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=['Low Traffic', 'High Traffic'], yticklabels=['Low Traffic', 'High Traffic'])
    axes[1].set_xlabel('Prediksi')
    axes[1].set_ylabel('Aktual')
    axes[1].set_title('Confusion Matrix - Data Testing')

    # Display the plot
    st.pyplot(fig)
    plt.clf()  # Clear the figure after showing it

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
    st.title("Prediksi Traffic Berdasarkan Kecepatan Mbps")

    # Upload file
    st.sidebar.subheader('Data Training')
    file = st.sidebar.file_uploader(label='Pilih data training', type=('xlsx'))
    
    # Button for training
    if st.sidebar.button('Training Model'):
        if file is not None:
            st.sidebar.write('File Uploaded')
            try:
                df_train = pd.read_excel(file)
                pipeline, X_train, X_test, y_train, y_test = data_training(df_train)
                st.success("Model telah dilatih dan disimpan.")
                
                # Menampilkan DataFrame dari dataset
                st.subheader('Dataset untuk Training')
                st.write(df_train)

                # Melakukan prediksi pada data training dan testing
                y_pred_train = pipeline.predict(X_train)
                y_pred_test = pipeline.predict(X_test)
                
                # Menampilkan confusion matrices
                show_confusion_matrices(y_train, y_pred_train, y_test, y_pred_test)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        else:
            st.warning("Silakan unggah file data training terlebih dahulu.")

    # Button for testing with a new file
    st.sidebar.subheader('Testing Model')
    test_file = st.sidebar.file_uploader(label='Pilih data testing', type=('xlsx'), key='test_file')

    if st.sidebar.button('Tampilkan Hasil Prediksi'):
        if test_file is not None:
            df_test = pd.read_excel(test_file)
            model = joblib.load('traffic_model.pkl')
            # Lakukan prediksi
            y_pred = model.predict(df_test)
            # Tambahkan kolom prediksi ke DataFrame
            df_test['Prediksi'] = y_pred
            df_test['Label'] = df_test['Prediksi'].apply(lambda x: 'Tinggi' if x == 1 else 'Rendah')
            st.write("Hasil Prediksi:")
            st.write(df_test)
        else:
            st.warning("Silakan unggah file data testing terlebih dahulu.")

    # Input prediksi
    st.subheader('Form Prediksi Traffic')
    form = st.form(key='prediction_form')
    jam = form.selectbox('Jam', [9, 13, 17])
    ping = form.slider('Ping (ms)', min_value=10, max_value=80, value=50)
    jitter = form.slider('Jitter (ms)', min_value=0, max_value=15, value=5)
    packet_loss = form.slider('Packet Loss (%)', min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    speed_mbps = form.slider('Kecepatan (Mbps)', min_value=10, max_value=80, value=40)

    submit = form.form_submit_button('Submit untuk Prediksi')
    
    if submit:
        # Memuat model yang telah ada
        model = joblib.load('traffic_model.pkl')
        prediction = predict_traffic(model, jam, ping, jitter, packet_loss, speed_mbps)
        st.write("Hasil Prediksi:")
        st.write("Traffic Tinggi" if prediction[0] == 1 else "Traffic Rendah")

if __name__ == "__main__":
    main()
