import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical

data = pd.read_csv("train_data.csv")
le = LabelEncoder()
data['appliance'] = le.fit_transform(data['appliance'])

# Set fixed length for footprint sequences
fixed_length = 100

# Function to parse and pad/truncate footprints
def parse_footprint(footprint, fixed_length):
    array = np.fromstring(footprint[1:-1], sep=',')
    if len(array) < fixed_length:
        array = np.pad(array, (0, fixed_length - len(array)), 'constant')
    elif len(array) > fixed_length:
        array = array[:fixed_length]
    return array

data['footprint_array'] = data['footprint'].apply(lambda x: parse_footprint(x, fixed_length))
data.dropna(subset=['footprint_array'], inplace=True)
X = np.stack(data['footprint_array'].values)
y = data['appliance']
X = X.reshape((X.shape[0], fixed_length, 1))
y = to_categorical(y, num_classes=len(le.classes_))

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
)

# LSTM model
model = Sequential([
    Masking(mask_value=0, input_shape=(fixed_length, 1)), 
    LSTM(128, activation='relu', return_sequences=False),
    Dense(len(le.classes_), activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print(f"LSTM Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))
