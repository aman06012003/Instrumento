import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('Intru_model.h5')

# Function to preprocess a custom audio file
def preprocess_custom_audio(file_path):
    y, sr = librosa.load(str(file_path), sr=22050, mono=True)  # Adjust parameters as needed
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128, hop_length=512)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed, mel_spectrogram

# Streamlit App
st.title("Audio Classification App")

# Upload file through Streamlit
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    # Preprocess the audio file
    feature_extracted, _ = preprocess_custom_audio(uploaded_file)
    feature_extracted = np.array(feature_extracted)
    feature_extracted = feature_extracted.reshape(1, feature_extracted.shape[0], 1)

    # Make predictions using the model
    predictions = model.predict(feature_extracted)

    # Get the predicted label
    pseudo_label = np.argmax(predictions)

    st.success(f"Predicted Label: {pseudo_label}")
