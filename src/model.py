from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_model(x_train, y_train):
    """
    Train a neural network classifier using MLPClassifier.
    
    Parameters:
    x_train (numpy.ndarray): Training data
    y_train (list): Training labels
    
    Returns:
    model: Trained model
    """
    model = MLPClassifier(alpha=0.01, batch_size=256, hidden_layer_sizes=(300,), max_iter=500)
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model using accuracy score.
    
    Parameters:
    model: Trained model
    x_test (numpy.ndarray): Test data
    y_test (list): Test labels
    
    Returns:
    float: Accuracy score
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    return accuracy

def save_model(model, filename="emotion_recognition_model.pkl"):
    """
    Save the trained model to a file.
    
    Parameters:
    model: Trained model
    filename (str): Path to save the model
    """
    with open(filename, "wb") as f:
        pickle.dump(model, f)

