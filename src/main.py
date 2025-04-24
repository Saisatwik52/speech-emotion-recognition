
import os
from sklearn.model_selection import train_test_split
from src.feature_extraction import extract_feature
from src.model import train_model, evaluate_model, save_model
import glob
import numpy as np

# Emotion dictionary
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("data/ravdess_data/Actor_*/*.wav"):  # Update path to your dataset location
        file_name = os.path.basename(file)
        emotion_code = file_name.split("-")[2]  # Emotion is in the 3rd position
        emotion = emotions.get(emotion_code)
        if emotion not in observed_emotions:
            continue
        
        features = extract_feature(file)
        x.append(features)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Load and prepare the data
x_train, x_test, y_train, y_test = load_data()

# Train the model
model = train_model(x_train, y_train)

# Evaluate the model
accuracy = evaluate_model(model, x_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
save_model(model)
