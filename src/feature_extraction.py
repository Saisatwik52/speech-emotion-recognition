import librosa
import soundfile as sf
import numpy as np

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    """
    Extracts MFCC, chroma, and mel spectrogram features from an audio file.
    
    Parameters:
    file_name (str): Path to the audio file
    mfcc (bool): Whether to extract MFCC features
    chroma (bool): Whether to extract chroma features
    mel (bool): Whether to extract mel spectrogram features
    
    Returns:
    np.array: Extracted features as a numpy array
    """
    with sf.SoundFile(file_name) as sound_file:
        audio_data = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        result = np.array([])

        if chroma:
            stft = np.abs(librosa.stft(audio_data))

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_features))

        if mel:
            mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_spectrogram))

    return result

