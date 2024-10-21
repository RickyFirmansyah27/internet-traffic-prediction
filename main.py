import streamlit as st
from utils import load_data, data_training, display_results, show_confusion_matrices, show_nb_testing, predict_traffic
import joblib

def main():
    st.title("Internet Traffic Prediction with Naive Bayes")
    st.sidebar.subheader('Data Training')
    file = st.sidebar.file_uploader(label='Pilih data training', type=('xlsx'))

    if st.sidebar.button('Training Model'):
        if file is not None:
            st.sidebar.write('File Uploaded')
            try:
                df_train = load_data(file)
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
            df_test = load_data(test_file)
            model = joblib.load('traffic_model.pkl')
            y_pred = model.predict(df_test)

            df_test['Prediksi'] = y_pred
            df_test['Label'] = df_test['Prediksi'].apply(lambda x: 'Tinggi' if x == 1 else 'Rendah')
            st.write("Hasil Prediksi:")
            st.write(df_test)

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
