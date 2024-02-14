import os
import librosa
import numpy as np

# Set the path to the directory containing the audio files
audio_dir = 'C:/Users/91701/OneDrive/Desktop/Research Project/Music Genre/Data/genres_original'

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    return mfcc

# List to store the MFCC features
mfcc_features = []

# List to store the corresponding genre labels
labels = []

# Iterate over the audio files in the directory
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            mfcc = extract_mfcc(file_path)
            mfcc_features.append(mfcc)
            labels.append(os.path.basename(root))

# Convert the lists to NumPy arrays
mfcc_features = np.array(mfcc_features)
labels = np.array(labels)

# Print the shape of the extracted MFCC features array
print("MFCC features shape:", mfcc_features.shape)
print("Labels shape:", labels.shape)
