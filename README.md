# fedAR+
This repository contains code and resources for a machine learning model to predict which home appliance is consuming power based on time series data. The project aims to classify power consumption patterns into one of five appliance categories.
#Project Overview
The goal of this project is to predict the appliance category from power consumption data collected over time. The dataset includes multiple time-based features and power readings, enabling us to build a model that accounts for seasonal and daily usage patterns.

#Dataset
Source: https://github.com/areinhardt/tracebase/tree/master
Format: The dataset is in CSV format with the following key columns:
datetime: Date and time of the reading (dd/mm/yyyy hh:mm:ss).
info1 and info2: Power consumption readings from a smart plug.
Appliance label: Encoded as 0, 1, 2, 3, and 4, representing five distinct appliance categories.
Features: The dataset has additional engineered features like year, month, day, hour, hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, along with lagged and rolling statistical features:
info1_lag_1, info2_lag_1: Lagged values for power readings.
info1_rolling_mean_3, info2_rolling_std_3: Rolling mean and standard deviation for window size 3.
Model and Training
Architecture: [Specify model type, e.g., a neural network, random forest, etc.]
Implementation: The model is implemented using [framework or libraries, e.g., TensorFlow, PyTorch, Scikit-Learn].
Target Variable: Appliance label (0, 1, 2, 3, 4) representing distinct appliances.
Challenge: One class had significantly lower recall in the model’s predictions, indicating potential difficulty in accurately classifying it. This could be due to class imbalance or similar power consumption patterns.
Evaluation
Overall Accuracy: 86.54%
Classification Report:
Precision, recall, and F1-score are provided for each of the five classes.
High precision and recall for certain classes, but relatively lower recall for one specific class.
Macro Average: Precision - 0.86, Recall - 0.81, F1-score - 0.82
Weighted Average: Precision - 0.88, Recall - 0.87, F1-score - 0.87
This indicates the model is fairly robust, with good predictive power, but could improve further on certain classes.
