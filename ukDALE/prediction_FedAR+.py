import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ast

def clean_and_convert_footprint(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []  

def load_data(file_path, max_length=30):
    data = pd.read_csv(file_path)
    data['footprint'] = data['footprint'].apply(clean_and_convert_footprint)
    data = data[data['footprint'].map(len) > 0]
    X = pad_sequences(data['footprint'].tolist(), maxlen=max_length, padding='post', truncating='post')
    y = data['appliance'].values 
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"Length of X: {len(X)}, Length of y: {len(y_encoded)}")

    return X, y_encoded, label_encoder

def estimate_label_distributions(model, X):
    probabilities = model.predict(X)
    return probabilities

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv1D(128, 1, activation='relu'),
        tf.keras.layers.Conv1D(128, 1, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_with_noise_handling(X, y_encoded, num_iterations=10):
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    model = create_model((X.shape[1], 1), len(np.unique(y_encoded)))
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=5)
    for iteration in range(num_iterations):
        label_distributions = estimate_label_distributions(model, X_train.reshape(X_train.shape[0], X_train.shape[1], 1))
        y_train = np.argmax(label_distributions, axis=1)
        model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=5)
    
    return model, X_val, y_val

def evaluate_model(model, X_val, y_val):
    y_pred_probs = model.predict(X_val.reshape(X_val.shape[0], X_val.shape[1], 1))
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

file_path = 'train_data.csv'
X, y_encoded, label_encoder = load_data(file_path)
model, X_val, y_val = train_with_noise_handling(X, y_encoded)
evaluate_model(model, X_val, y_val)
