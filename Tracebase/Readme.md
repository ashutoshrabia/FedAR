# Appliance Prediction using Tracebase Dataset

## Overview

This project focuses on predicting the energy consumption of household appliances using the Tracebase dataset. The dataset contains detailed appliance-level energy consumption data. This README outlines the steps taken to preprocess, analyze, and model the data using various machine learning techniques.

## Workflow

The project workflow consists of several key steps, each handled by specific Python scripts:

### 1. Data Extraction

- **Tracebase Data**: The relevant data is accessed from the Tracebase dataset and loaded into a suitable format for preprocessing.

### 2. Data Preprocessing

Data preprocessing is essential to ensure the quality and usability of the dataset. The following concepts are defined:

#### 2.1 Terminology

- **Switch Point**: For a time series collection \( X = \{x_1, x_2, \ldots, x_n\} \), a time instance \( t \) is a switch point if:
  1. The difference \( \delta(t) = |X(t) - X(t - 1)| > \phi_1 \), a predefined threshold.
  2. The rate of change \( \delta_r(t) = \delta(t) / X(t) > \phi_2 \), another threshold.

  Empirical thresholds used are:
  - \( \phi_1 = 30 \) watts
  - \( \phi_2 = 0.2 \) (20%)

- **Steady Point**: A time point \( t \) is considered steady if \( \delta_r(t) < \phi_2 \).

- **Steady Period**: A steady period is a subsequence \( X_{t:m} \) where all time points are steady, starting from a switch point \( t \).

#### 2.2 Appliance Footprint Definition

For a steady period \( X_{t:m} \) corresponding to the ON state of the appliance, the appliance footprint is:

\[
X_{af} = \{X_{t:m}(i) - X_{t:m}(i - 1) \mid 1 \leq i \leq m\}
\]

This captures the first-order differences between consecutive data points, highlighting identifiable patterns in power consumption.

#### 2.3 Footprint Extraction Steps

1. **Identify Switch Points**: Detect switch points using thresholds \( \phi_1 \) and \( \phi_2 \).
2. **Identify Steady Periods**: 
   - Search for a steady period \( X_{t:m} \) following each switch point \( t \).
   - Classify \( X_{t:m} \) as an ON state if \( X(t) - X(t + m) < 0 \); otherwise, it is OFF.
3. **Store Appliance Footprints**: Save the ON state footprints with labels for analysis.

### 3. Preparing Training Data

The preprocessed data is split into training and testing datasets to accurately evaluate model performance.

### 4. Model Training

A Deep Neural Network (DNN) is employed to predict appliance energy consumption.

#### 4.1 DNN Architecture

- **Input**: Features \( \mathcal{X} \) (appliance footprints) and labels \( \mathcal{Y} \) from the dataset \( \mathcal{D} \).
- **Convolutional Layers**: Three 1D convolutional layers with 128 filters each.
- **Flatten Layer**: Converts the 3D output to 1D for the fully connected layer.
- **Fully Connected Layer**: Contains \( C \) neurons, where \( C \) is the number of class labels.
- **Output**: A softmax function provides class probabilities.

#### 4.2 Model Training Script

To train the model, use the following command:

```bash
python file_name.py

