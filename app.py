import librosa
import streamlit as st
import numpy as np
import pandas  as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('Intru_model.h5')


def preprocess_custom_audio(file_path):
    y,sr = librosa.load(str(file_path),sr = None ,mono=True)  # Adjust parameters as needed
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=None, n_mels=128, hop_length=512)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=None, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed,  mel_spectrogram

custom_file_path = st.file_uploader("Upload a custom audio file", type=["wav", "mp3"])

# Preprocess the audio file
if custom_file_path is not None:
    feature_extracted, mel_spectrogram = preprocess_custom_audio(custom_file_path)
    feature_extracted = np.array(feature_extracted)
    feature_extracted = feature_extracted.reshape(1,feature_extracted.shape[0],1)
# Make predictions using the model
    predictions = model.predict(feature_extracted)

#  Get the predicted label
    pseudo_label = np.argmax(predictions)
    output = ["Accordion","Alto Saxophone","Bass Tuba","Bassoon","Cello","Clarinet","Contrabass","Flute","French Horn","Oboe","Trombone","Trumpet","Viola","Violin"]
    st.write(f"The give instrument is {output[int(pseudo_label)]}")
else:
    st.write("Please enter a valid Audio file")
