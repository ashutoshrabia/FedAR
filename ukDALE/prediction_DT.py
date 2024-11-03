import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("train_data.csv")
le = LabelEncoder()
data['appliance'] = le.fit_transform(data['appliance'])

# Set fixed length for footprint sequences
fixed_length = 50 

# Function to parse and pad/truncate footprints
def parse_footprint(footprint, fixed_length):
    try:
        array = np.fromstring(footprint[1:-1], sep=',')
        # Pad if shorter, truncate if longer
        if len(array) < fixed_length:
            array = np.pad(array, (0, fixed_length - len(array)), 'constant')
        elif len(array) > fixed_length:
            array = array[:fixed_length]
        return array
    except ValueError:
        return None

data['footprint_array'] = data['footprint'].apply(lambda x: parse_footprint(x, fixed_length))
data.dropna(subset=['footprint_array'], inplace=True)
X = np.stack(data['footprint_array'].values)
y = data['appliance']

# Test-train split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# RF
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"RF Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\n")

# LightGBM
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.3,
    'num_leaves': 31,
    'verbose':-1,
    'random_state': 42
}

model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval])
y_pred = np.argmax(model.predict(X_test, num_iteration=model.best_iteration), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\n")